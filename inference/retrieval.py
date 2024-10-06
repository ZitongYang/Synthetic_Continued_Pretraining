import logging
import os
from typing import List, Iterable, Optional, Union, cast, Tuple, Dict

import cohere
import pandas as pd
from langchain.embeddings.cache import (
    CacheBackedEmbeddings, Embeddings, ByteStore, EncoderBackedStore, batch_iterate,  # noqa
    _create_key_encoder, _value_serializer, _value_deserializer)  # noqa
from langchain.storage import LocalFileStore
from langchain_community.document_loaders import DataFrameLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from inference.retry_wrapper import rerank_with_backoff
from tasks.quality import QuALITY
from utils import python_utils
from utils.io_utils import set_openai_key, set_cohere_private_key, jload, jdump
from utils.prompt_utils import format_name

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DOCUMENT_EMBEDDING_CACHE_DIR = 'cache/retrieval/doc_embeddings'
QUERY_EMBEDDING_CACHE_DIR = 'cache/retrieval/query_embeddings'
RETRIEVAL_CACHE_DIR = 'cache/retrieval/retrieval_cache'
RERANK_CACHE_DIR = 'cache/retrieval/rerank_cache'

RETRIEVAL_CONTEXT_CHUNK_PREFIX = "*Context {i}*"
RETRIEVAL_CONTEXT_CHUNK_DELIMITER = "\n\n"


"""Embeddings."""


class DocumentAndQueryCacheBackedEmbeddings(CacheBackedEmbeddings):
    """Two overloaded methods. This code is copied almost entirely from the `langchain` library (Chase et al., 2022).

    Overload method .from_bytes_store(): We add the ability to cache query embeddings in a separate namespace.
        This is done because the embedding model entirely determines the query embedding namespace,
        while the document embedding namespace is determined by many hyperparameters.

    New method .embed_queries(): Batched embedding of queries. Can be used when a large set of queries are
        available a priori and need to be embedded.
    """
    @classmethod
    def from_bytes_store(
            cls,
            underlying_embeddings: Embeddings,
            document_embedding_cache: ByteStore,
            *,
            document_namespace: str = "",
            query_namespace: str = "",
            batch_size: Optional[int] = None,
            query_embedding_cache: Union[bool, ByteStore] = False,
    ) -> CacheBackedEmbeddings:
        """On-ramp that adds the necessary serialization and encoding to the store.

        Args:
            underlying_embeddings: The embedder to use for embedding.
            document_embedding_cache: The cache to use for storing document embeddings.
            *,
            document_namespace: The namespace to use for document cache.
                       This namespace is used to avoid collisions with other caches.
                       For example, set it to the name of the embedding model used.
            query_namespace: The namespace to use for query cache.
                      This namespace is used to avoid collisions with other caches.
                      For example, set it to the name of the embedding model used.
            batch_size: The number of documents to embed between store updates.
            query_embedding_cache: The cache to use for storing query embeddings.
                True to use the same cache as document embeddings.
                False to not cache query embeddings.
        """
        document_key_encoder = _create_key_encoder(document_namespace)
        document_embedding_store = EncoderBackedStore[str, List[float]](
            document_embedding_cache,
            document_key_encoder,
            _value_serializer,
            _value_deserializer,
        )
        if query_embedding_cache is True:
            query_embedding_store = document_embedding_store
        elif query_embedding_cache is False:
            query_embedding_store = None
        else:
            query_key_encoder = _create_key_encoder(query_namespace)
            query_embedding_store = EncoderBackedStore[str, List[float]](
                query_embedding_cache,
                query_key_encoder,
                _value_serializer,
                _value_deserializer,
            )

        return cls(
            underlying_embeddings,
            document_embedding_store,
            batch_size=batch_size,
            query_embedding_store=query_embedding_store,
        )

    def embed_queries(self, queries: List[str]) -> List[List[float]]:
        """Embed a list of queries.

        The method first checks the cache for the embeddings.
        If the embeddings are not found, the method uses the underlying embedder
        to embed the queries and stores the results in the cache.

        Args:
            queries: A list of queries to embed. These should already have metadata prepended, if that's desired.

        Returns:
            A list of embeddings for the given queries.
        """
        vectors: List[Union[List[float], None]] = self.query_embedding_store.mget(
            queries
        )
        all_missing_indices: List[int] = [
            i for i, vector in enumerate(vectors) if vector is None
        ]

        for missing_indices in batch_iterate(self.batch_size, all_missing_indices):
            missing_queries = [queries[i] for i in missing_indices]

            # This is the key change. We use .embed_documents(), which will batch
            #     request to OpenAI (or whatever the underlying embedding is).
            missing_vectors = self.underlying_embeddings.embed_documents(missing_queries)
            assert len(missing_vectors) == len(missing_queries)
            self.query_embedding_store.mset(
                list(zip(missing_queries, missing_vectors))
            )
            for index, updated_vector in zip(missing_indices, missing_vectors):
                vectors[index] = updated_vector

        return cast(
            List[List[float]], vectors
        )  # Nones should have been resolved by now


def load_langchain_documents_from_quality_dataset(task: QuALITY) -> List[Document]:
    df = pd.DataFrame(
        data={
            'text': [document.text for document in task.documents],
            'title': [document.title for document in task.documents],
            'year': [document.year for document in task.documents],
            'author': [document.author for document in task.documents]
        })

    # Deduplicate on all columns
    df = df.drop_duplicates()
    logging.info(f"Loaded {len(df)} documents from QuALITY dataset, after deduplication.")

    return DataFrameLoader(df).load()


def get_metadata_prefix_for_quality_document(document):
    """Returns a prefix with document metadata, which is prepended to each
            text chunk that we index for retrieval."""
    metadata = document.metadata
    title, author, year = metadata['title'], metadata['author'], metadata['year']
    formatted_author = format_name(author)
    return f'Title: {title}\nAuthor: {formatted_author}\nYear Written: {year}\n\n'


def split_langchain_documents(
    documents: Iterable[Document],
    text_split_strategy: str,
    chunk_size: int,
    overlap_size: int,
):
    if text_split_strategy == 'recursive':
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_size)
    else:
        raise ValueError(f"Unsupported text split strategy: {text_split_strategy}")

    # For each document, split the texts and prepend with metadata (title, author, year)
    texts = []
    for document in documents:
        per_document_texts = text_splitter.split_text(document.page_content)
        metadata_prefix = get_metadata_prefix_for_quality_document(document)
        texts.extend([f'{metadata_prefix}{text}' for text in per_document_texts])

    return texts


def get_document_embedding_namespace(
    embedding_model_path: str,
    text_split_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    return (
        f"task_name-quality--"
        f"embedding_model_path-{embedding_model_path}--"
        f"text_split_strategy-{text_split_strategy}--"
        f"chunk_size-{chunk_size}--"
        f"chunk_overlap-{chunk_overlap}--"
    )


def get_query_embedding_namespace(
    embedding_model_path: str,
):
    return (
        f"task_name-quality--"
        f"embedding_model_path-{embedding_model_path}--"
    )


def get_cached_langchain_embedding_model(
    embedding_model_path: str,
    text_split_strategy: str = 'recursive',
    chunk_size: int = 1024,
    chunk_overlap: int = 0,
    document_embedding_cache_dir: str = DOCUMENT_EMBEDDING_CACHE_DIR,
    query_embedding_cache_dir: str = QUERY_EMBEDDING_CACHE_DIR
) -> CacheBackedEmbeddings:
    underlying_embeddings = OpenAIEmbeddings(
        model=embedding_model_path,
        deployment=embedding_model_path,
        show_progress_bar=True,
        max_retries=10000
    )
    document_cache = LocalFileStore(document_embedding_cache_dir)
    document_cache_namespace = get_document_embedding_namespace(
        embedding_model_path=embedding_model_path,
        text_split_strategy=text_split_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    query_cache = LocalFileStore(query_embedding_cache_dir)
    query_cache_namespace = get_query_embedding_namespace(
        embedding_model_path=embedding_model_path,
    )
    logging.info(
        f"Caching document embeddings to directory {document_embedding_cache_dir} "
        f"with namespace prefix {document_cache_namespace}")
    logging.info(
        f"Caching query embeddings to directory {query_embedding_cache_dir} "
        f"with namespace prefix {query_cache_namespace}")

    # Caches both document embeddings and query embeddings, in separate caches with separate namespaces
    # See DocumentAndQueryCacheBackedEmbeddings for more details
    cached_embeddings = DocumentAndQueryCacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=underlying_embeddings,
        document_embedding_cache=document_cache,
        document_namespace=document_cache_namespace,
        query_embedding_cache=query_cache,
        query_namespace=query_cache_namespace,
    )
    return cached_embeddings


"""Performing retrieval and reranking."""


class RetrievalAndRerankResults:
    """Allow the user to specify posthoc the retrieval_top_k and rerank_top_k."""
    def __init__(
            self,
            retrieval_results: List[List[Tuple[str, float]]],
            rerank_results: List[List[Tuple[str, float]]]
    ):
        self.retrieval_results = retrieval_results
        self.base_rerank_results = rerank_results

        self.rerank_list_of_chunk_to_rerank_score = []
        for rerank_results_for_query in rerank_results:
            chunk_to_rerank_score = {}
            for chunk, rerank_score in rerank_results_for_query:
                chunk_to_rerank_score[chunk] = rerank_score

            self.rerank_list_of_chunk_to_rerank_score.append(chunk_to_rerank_score)

    def get_retrieval_results_for_top_k(self, retrieval_top_k: int) -> List[List[Tuple[str, float]]]:
        return [retrieval_results[:retrieval_top_k] for retrieval_results in self.retrieval_results]

    def get_rerank_results_for_top_k(self, retrieval_top_k: int, rerank_top_k: int) -> List[List[Tuple[str, float]]]:
        # First, we filter to the top k retrieval results
        retrieval_results_for_top_k = self.get_retrieval_results_for_top_k(retrieval_top_k)

        # Then, for each query, we:
        #   Filter the rerank results to only include the top k retrieval results
        #   Sort those results according to the rerank score
        #   Return the top k rerank results
        rerank_results_for_top_k = []
        for i, (retrieval_results_for_query, rerank_results_for_query) in enumerate(python_utils.zip_(
                retrieval_results_for_top_k, self.base_rerank_results)):
            rerank_results_for_query = [  # List[Tuple[float, str]]
                (chunk, self.rerank_list_of_chunk_to_rerank_score[i][chunk]) for chunk, _ in retrieval_results_for_query]

            # Sort by rerank score, higher is better
            rerank_results_for_query = sorted(rerank_results_for_query, key=lambda x: x[1], reverse=True)
            rerank_results_for_top_k.append(rerank_results_for_query[:rerank_top_k])

        return rerank_results_for_top_k


class LangChainRetriever:
    def __init__(
        self,
        task: QuALITY,
        embedding_model_path: str,
        text_split_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        rerank_model_path: str
    ):
        self.task = task
        self.task_name = 'quality'
        self.embedding_model_path = embedding_model_path
        self.text_split_strategy = text_split_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.rerank_model_path = rerank_model_path

        set_openai_key()

        # Parse into LangChain documents
        self.docs = load_langchain_documents_from_quality_dataset(task)

        # Split docs into text chunks, prepending metadata to each text chunk
        self.texts = split_langchain_documents(
            documents=self.docs,
            text_split_strategy=text_split_strategy,
            chunk_size=chunk_size,
            overlap_size=chunk_overlap,
        )

        # Get embedding model
        self.embeddings = get_cached_langchain_embedding_model(
            embedding_model_path=embedding_model_path,
            text_split_strategy=text_split_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Set up the rerank model
        if self.rerank_model_path == 'rerank-english-v3.0':
            logging.info("Setting up Cohere reranker.")
            set_cohere_private_key()
            self.rerank_model = cohere.Client()
            self.rerank_wrapper_fn = self.rerank_with_cohere
        else:
            raise NotImplementedError(f"Unknown rerank model path: {self.rerank_model_path}")

        # Actually perform doc embedding (or retrieves from the cache, if available) and build the FAISS vector store
        logging.info("Starting document embedding (or getting document embeddings from cache) and indexing.")
        self.vector_store = FAISS.from_texts(self.texts, self.embeddings)
        logging.info("Document embedding and indexing completed.")

        # Embed queries
        logging.info("Starting query embedding (or getting query embeddings from cache).")

        all_task_questions_for_query_embedding = self.get_all_task_questions_for_query_embedding()
        self.query_embeddings = self.embeddings.embed_queries(all_task_questions_for_query_embedding)
        logging.info("Query embedding completed.")

    def get_all_task_questions_for_query_embedding(self):
        return self.task.all_questions(
            add_document_context=True,
            # The thought process suffix doesn't make sense to put in the query embeddings;
            #   it is the same for all questions.
            add_thought_process=False,
            sep_after_question='\n\n'
        )

    def get_rerank_cache_path(
            self,
            retrieval_max_k: int,
            rerank_model_path: str,
            rerank_cache_dir: str = RERANK_CACHE_DIR
    ) -> str:
        return (
            f"{rerank_cache_dir}/"
            f"task_name-{self.task_name}/"
            f"embedding_model_path-{self.embedding_model_path}/"
            f"text_split_strategy-{self.text_split_strategy}/"
            f"chunk_size-{self.chunk_size}/"
            f"chunk_overlap-{self.chunk_overlap}/"
            f"retrieval_max_k-{retrieval_max_k}/"
            f"rerank_model_path-{rerank_model_path}/"
            f"cache.json"
        )

    def get_retrieval_cache_path(
            self,
            retrieval_cache_path: str = RETRIEVAL_CACHE_DIR
    ):
        return (
            f"{retrieval_cache_path}/"
            f"task_name-{self.task_name}/"
            f"embedding_model_path-{self.embedding_model_path}/"
            f"text_split_strategy-{self.text_split_strategy}/"
            f"chunk_size-{self.chunk_size}/"
            f"chunk_overlap-{self.chunk_overlap}/"
            f"cache.json"
        )

    def retrieve_chunks_for_all_queries(
            self,
            top_k: int,
            rerank_model_path: str,
    ):
        retrieval_cache_path = self.get_retrieval_cache_path()

        if not os.path.exists(retrieval_cache_path):
            os.makedirs(os.path.dirname(retrieval_cache_path), exist_ok=True)

        try:
            retrieval_cached_results = jload(retrieval_cache_path)  # type: Dict[Tuple[float], List[Tuple[str, float]]]
            logging.info(f"Loaded retrieval cache from {retrieval_cache_path}.")
        except FileNotFoundError:
            retrieval_cached_results = {}
            logging.info(f"No retrieval cache found at {retrieval_cache_path}. Creating a new one.")

        should_overwrite_cache = False

        # List of List[Tuple[str, float]]
        # The str is the text chunk, and the float is the similarity score (lower is better)
        chunks_for_all_queries = []
        all_task_questions_for_query_embedding = self.get_all_task_questions_for_query_embedding()
        for query_str, embedding in python_utils.zip_(
                all_task_questions_for_query_embedding, self.query_embeddings):
            if query_str in retrieval_cached_results.keys():
                chunks_for_query = retrieval_cached_results[query_str]
            else:
                chunks_for_query = self.vector_store.similarity_search_with_score_by_vector(embedding, k=top_k)
                chunks_for_query = [(doc.page_content, score) for doc, score in chunks_for_query]
                retrieval_cached_results[query_str] = chunks_for_query
                should_overwrite_cache = True

            chunks_for_all_queries.append(chunks_for_query)

        # Cache the retrieval results
        if should_overwrite_cache:
            jdump(retrieval_cached_results, retrieval_cache_path)
            logging.info(f"Overwrote retrieval cache at {retrieval_cache_path}.")

        logging.info(f"Retrieval completed. Cached results to {retrieval_cache_path}.")

        return RetrievalAndRerankResults(
            retrieval_results=chunks_for_all_queries,
            rerank_results=self.rerank_chunks_for_all_queries(
                chunks_for_all_queries=chunks_for_all_queries,
                rerank_model_path=rerank_model_path,
                retrieval_max_k=top_k
        ))

    @staticmethod
    def rerank_with_cohere(
            client,
            query,
            documents,
            model
    ):
        """See documentation at https://docs.cohere.com/reference/rerank."""
        # By leaving top_n unspecified, we default to returning all documents with their reranker scores
        cohere_output = rerank_with_backoff(
            client,
            query=query,
            documents=documents,
            model=model,
            return_documents=True
        )
        # The above returns an ordered list of {index, text, relevance score}
        # We convert this to a list of tuples (text, relevance score)
        results = []

        for results_item in cohere_output.results:
            results.append((results_item.document.text, results_item.relevance_score))

        return results

    def rerank_chunks_for_all_queries(
            self,
            chunks_for_all_queries: List[List[Tuple[str, float]]],
            rerank_model_path: str,
            retrieval_max_k: int
    ) -> List[List[Tuple[str, float]]]:
        logging.info(f"Starting reranking with {rerank_model_path}.")

        rerank_cache_path = self.get_rerank_cache_path(
            retrieval_max_k=retrieval_max_k,
            rerank_model_path=rerank_model_path)

        # The str is the text chunk, and the float is the reranker score (higher is better)
        if not os.path.exists(rerank_cache_path):
            os.makedirs(os.path.dirname(rerank_cache_path), exist_ok=True)

        try:
            rerank_cached_results = jload(rerank_cache_path)  # type: Dict[str, List[Tuple[str, float]]]
            logging.info(f"Loaded rerank cache from {rerank_cache_path}.")
        except FileNotFoundError:
            rerank_cached_results = {}
            logging.info(f"No rerank cache found at {rerank_cache_path}. Creating a new one.")

        should_overwrite_cache = False

        query_strs = self.get_all_task_questions_for_query_embedding()
        all_reranking_results = []  # type: List[List[Tuple[str, float]]]
        for query_str, retrieved_chunks in python_utils.zip_(query_strs, chunks_for_all_queries):
            if query_str in rerank_cached_results.keys():
                all_reranking_results.append(rerank_cached_results[query_str])
            else:
                retrieved_chunk_strs = [retrieved_chunks[i][0] for i in range(len(retrieved_chunks))]
                rerank_results = self.rerank_wrapper_fn(
                    self.rerank_model,
                    query=query_str,
                    documents=retrieved_chunk_strs,
                    model=rerank_model_path
                )
                all_reranking_results.append(rerank_results)
                rerank_cached_results[query_str] = rerank_results
                should_overwrite_cache = True

        # Cache the reranking results
        if should_overwrite_cache:
            jdump(rerank_cached_results, rerank_cache_path)
            logging.info(f"Overwrote rerank cache at {rerank_cache_path}.")

        logging.info(f"Reranking completed. Cached results to {rerank_cache_path}.")

        return all_reranking_results


"""Few-shot prompt formatting."""


def format_retrieval_icl_prompt(
    prompt_template: str,
    question_and_choices_str: str,
    retrieved_chunks: List[Tuple[str, float]],
    chunk_prefix: str = RETRIEVAL_CONTEXT_CHUNK_PREFIX,
    chunk_delimiter: str = RETRIEVAL_CONTEXT_CHUNK_DELIMITER
) -> str:
    retrieved_chunks_str = ""
    for i, (doc, _) in enumerate(retrieved_chunks):
        chunk_prefix_i = chunk_prefix.format(i=i+1)
        retrieved_chunks_str += f"{chunk_prefix_i}\n"
        retrieved_chunks_str += f"{doc}"
        retrieved_chunks_str += f"{chunk_delimiter}"

    return prompt_template.format(
        formatted_question_and_choices=question_and_choices_str,
        k_icl_retrieval_chunks=len(retrieved_chunks),
        contexts=retrieved_chunks_str
    )


def get_best_first_retrieved_chunks(
    retrieved_chunks: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    # Sort based on the score, higher is better
    return sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)


def get_best_last_retrieved_chunks(
    retrieved_chunks: List[Tuple[str, float]]
) -> List[Tuple[str, float]]:
    # We use the approach of sorting in descending order and then reversing, because there are sometimes ties
    return sorted(retrieved_chunks, key=lambda x: x[1], reverse=True)[::-1]


RETRIEVED_CHUNK_ORDER_SETTING_TO_SORT_FN = {
    'best_first': get_best_first_retrieved_chunks,
    'best_last': get_best_last_retrieved_chunks
}


def get_retrieval_prompts(
    task: QuALITY,
    list_of_retrieved_chunks: List[List[Tuple[str, float]]],
    retrieved_chunk_order: str,
) -> List[str]:
    retrieval_icl_prompt_template = python_utils.read("utils/prompts/quality_retrieval_icl.txt")
    all_task_formatted_questions_and_choices = task.all_questions(
        add_document_context=True,
        add_thought_process=True,
        sep_after_question='\n\n'
    )
    retrieval_prompts = []
    for formatted_question_and_choices, retrieved_chunks in python_utils.zip_(
            all_task_formatted_questions_and_choices,
            list_of_retrieved_chunks
    ):
        retrieved_chunks = RETRIEVED_CHUNK_ORDER_SETTING_TO_SORT_FN[retrieved_chunk_order](retrieved_chunks)
        formatted_prompt = format_retrieval_icl_prompt(
            prompt_template=retrieval_icl_prompt_template,
            question_and_choices_str=formatted_question_and_choices,
            retrieved_chunks=retrieved_chunks
        )
        retrieval_prompts.append(formatted_prompt)

    return retrieval_prompts
