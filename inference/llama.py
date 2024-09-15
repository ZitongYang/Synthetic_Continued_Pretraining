import torch
from typing import Sequence, Optional, Union, Iterable
from vllm import LLM, SamplingParams

from utils import python_utils
from utils.prompt_utils import generate_all_answer_strings


def llama_forward(
    prefix_or_prefixes: Optional[Union[str, Iterable[str]]],
    prompts: Sequence[str],
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[str] = None,
    max_length: int = 300,  # Generation length
    temperature: float = 0.1,
    n_samples: int = 8,
    n_gpus: int = 8,
) -> Optional[Sequence[str]]:
    assert model is not None or model_path is not None, "model or model_path must be provided"

    if isinstance(prefix_or_prefixes, str):
        prompts = [prefix_or_prefixes + prompt for prompt in prompts]
    elif isinstance(prefix_or_prefixes, Sequence):
        prompts = [prefix + prompt for prefix, prompt in python_utils.zip_(prefix_or_prefixes, prompts)]
    elif prefix_or_prefixes is not None:
        raise ValueError(
            f"prefix_or_prefixes must be None, a string, or a sequence of strings, not {type(prefix_or_prefixes)}")

    if temperature == 0.0:
        # best_of must be 1 when using greedy decoding
        n_samples = 1

    sampling_params = SamplingParams(n=n_samples,
                                     temperature=temperature,
                                     max_tokens=max_length,
                                     stop=generate_all_answer_strings())

    # Create an LLM.
    if model is None:
        model = LLM(
            model=model_path, tokenizer="meta-llama/Meta-Llama-3-8B", tensor_parallel_size=n_gpus)

    outputs = model.generate(prompts=prompts, sampling_params=sampling_params)

    result = []
    for output in outputs:
        attempts = []
        for ith_output in output.outputs:
            answer = ith_output.stop_reason
            if answer:
                attempts.append(ith_output.text + answer)
        result.append(attempts)

    return result
