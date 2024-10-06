import logging
import os
from dataclasses import field, dataclass, asdict
from typing import Optional, Iterable

from transformers import HfArgumentParser
from vllm import LLM

from inference.llama import llama_forward
from inference.retrieval import LangChainRetriever, get_retrieval_prompts
from tasks.quality import QuALITY
from tasks.task_abc import Task
from utils.io_utils import jdump

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@dataclass
class EvalConfigs:
    eval_func: str = field(default='eval_quality_qa', metadata={'help': 'Task to run'})
    model_path: str = field(default='meta-llama/Meta-Llama-3-8B')
    model_name: Optional[str] = None
    eval_temperature: float = field(default=0.1)

    # Retrieval args
    embedding_model_path: Optional[str] = field(default='text-embedding-3-large')
    text_split_strategy: Optional[str] = field(default='recursive')
    chunk_size: Optional[int] = field(default=1024, metadata={'help': 'Chunk size in chars for text split strategy.'})
    chunk_overlap: Optional[int] = field(default=0, metadata={'help': 'Overlap size in chars for text split strategy.'})
    retrieval_max_k: Optional[int] = field(
        default=128,
        metadata={'help': 'Upper bound on the number of chunks to retrieve per query. '
                          'Used to pre-embed and cache rerank results.'})
    retrieval_top_k: Optional[int] = field(
        default=128,
        metadata={'help': 'Top k chunks to retrieve per query. Used to construct prompts for LM evaluation.'})
    rerank_model_path: Optional[str] = field(default='rerank-english-v3.0"')
    rerank_top_k: Optional[int] = field(
        default=16,
        metadata={'help': 'Top k chunks to rerank per query. Used to construct prompts for LM evaluation.'})
    retrieved_chunk_order: Optional[str] = field(default='best_first')

    def __post_init__(self):
        if not self.model_name:
            if self.model_path == 'meta-llama/Meta-Llama-3-8B':
                self.model_name = 'llama3-base'
            else:
                self.model_name = self.model_path.split('/')[-1]


def _write_outputs(
    task: Task,
    savename: str,
    outputs: Iterable,
    prompts: Optional[Iterable] = None
):
    i = 0
    for document in task.documents:
        for question in document.questions:
            output = outputs[i]
            if question.attempts == [{}]:
                attempts = []
            else:
                attempts = question.attempts
            for attempt in output:
                attempts.append(question.llama_parse_answer(attempt))

            question.attempts = attempts

            if prompts is not None:
                question.formatted_prompt = prompts[i]

            i += 1

    jdump(task.asdict(), savename)


def eval_quality_qa(model_path: str,
              model_name: str,
              **kwargs):
    task = QuALITY('all')
    savename = f'out/qualityqa-{model_name}.json'
    system_message = task.llama_cot_prompt
    print(savename)

    if os.path.exists(savename):
        task.load_attempts_json(savename)

    outputs = llama_forward(
        model_path=model_path,
        prefix_or_prefixes=system_message,
        prompts=task.all_questions(
            add_document_context=True,
            add_thought_process=True,
            sep_after_question='\n'),
    )
    _write_outputs(task=task, savename=savename, outputs=outputs)


def eval_quality_qa_with_rag(
    model_path: str,
    model_name: str,
    embedding_model_path: str,
    text_split_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    retrieval_max_k: int,
    retrieval_top_k: int,
    rerank_model_path: str,
    rerank_top_k: int,
    retrieved_chunk_order: str,
    eval_temperature: float,
    **kwargs
):
    savename = f'out/retrieval/qualityqa-rag-{model_name}/'
    savename += (
        f"embedding_model_path-{embedding_model_path}/"
        f"text_split_strategy-{text_split_strategy}/"
        f"chunk_size-{chunk_size}/"
        f"chunk_overlap-{chunk_overlap}/"
        f"retrieval_top_k-{retrieval_top_k}/"
        f"rerank_model_path-{rerank_model_path}/"
        f"rerank_top_k-{rerank_top_k}/"
        f"retrieved_chunk_order-{retrieved_chunk_order}/"
        f"eval_temperature-{eval_temperature}/"
        f"out.json"
    )
    print(savename)
    if os.path.exists(savename):
        print('File already exists. Terminating job.')
        return

    task = QuALITY('all')

    # We cache embeddings for both document chunks and queries,
    #   so the below won't do any embedding if these two caches exist
    retriever = LangChainRetriever(
        task=task,
        embedding_model_path=embedding_model_path,
        text_split_strategy=text_split_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rerank_model_path=rerank_model_path,
    )
    logging.info(f"Setup retriever index.")

    # Run retrieval for all queries, using top_k=retrieval_max_k.
    #   We just run retrieval once, rerank all the retrieved docs, and cache these results.
    logging.info(f"Retrieving {retrieval_max_k} chunks per query for QuALITY QA.")

    # Each of the retrieval and rerank results has type: Dict[str, List[Tuple[str, float]]]
    # For each query, we have reranked `retrieval_max_k` chunks
    # This allows us to sweep over different top_k values for both retrieval and rerank, posthoc
    # In each Tuple, the str is the text chunk, and the float is the reranker score
    retrieval_and_maybe_rerank_results = retriever.retrieve_chunks_for_all_queries(
        top_k=retrieval_max_k,
        rerank_model_path=rerank_model_path,
    )

    logging.info(f"Retrieval top_k: {retrieval_top_k}, Rerank top_k: {rerank_top_k}")
    if rerank_top_k > retrieval_top_k:
        raise ValueError(
            f"Rerank top_k ({rerank_top_k}) must be less than or equal to retrieval top_k ({retrieval_top_k}).")

    logging.info(f"Constructing prompts for LM evaluation.")

    # Get retrieval and rerank results based on (retrieval_top_k, rerank_top_k)
    list_of_retrieved_chunks = retrieval_and_maybe_rerank_results.get_rerank_results_for_top_k(
        retrieval_top_k=retrieval_top_k,
        rerank_top_k=rerank_top_k,
    )
    prompts = get_retrieval_prompts(
        task=task,
        list_of_retrieved_chunks=list_of_retrieved_chunks,
        retrieved_chunk_order=retrieved_chunk_order,
    )

    # Now, we run the LM evaluation for all prompts
    logging.info(f"Running LM evaluation for savename: {savename}")
    model = LLM(model=model_path, tokenizer="meta-llama/Meta-Llama-3-8B", swap_space=16)
    outputs = llama_forward(
        prefix_or_prefixes=None,
        prompts=prompts,
        model=model,
        max_length=300,  # of generation
        temperature=eval_temperature
    )

    # Means we are in the master process if distributed evaluation is enabled
    if outputs:
        _write_outputs(task=task, savename=savename, outputs=outputs, prompts=prompts)


if __name__ == '__main__':
    parser = HfArgumentParser(EvalConfigs)
    configs = parser.parse_args_into_dataclasses()[0]
    globals()[configs.eval_func](**asdict(configs))
