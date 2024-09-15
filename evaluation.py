import logging
import os
from dataclasses import field, dataclass, asdict
from typing import Optional, Iterable
from transformers import HfArgumentParser

from inference.llama import llama_forward
from tasks.quality import QuALITY
from tasks.task_abc import Task
from utils.io_utils import jdump

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class EvalConfigs:
    model_path: str = field(default='meta-llama/Meta-Llama-3-8B')
    model_name: Optional[str] = None

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


if __name__ == '__main__':
    parser = HfArgumentParser(EvalConfigs)
    configs = parser.parse_args_into_dataclasses()[0]
    eval_quality_qa(**asdict(configs))