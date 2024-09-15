from dataclasses import dataclass, field, asdict
from typing import Optional
import transformers
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from data.cptdata import get_task_data_module


@dataclass
class TrainingConfig:
    task_name: str
    block_size: int
    rehersal_rate: float
    model_name: str
    subsample_ratio: float
    wandb_project: Optional[str] = field(default="synthetic-continued-pretraining")

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project

def train():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, transformers.TrainingArguments))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}")

    # loading model
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.model_name)
    # loading dataset
    data_module = get_task_data_module(**asdict(config))

    # setting up trainer
    trainer = transformers.Trainer(model=model, args=args, **data_module)
    trainer.train()
    trainer.save_model(output_dir=args.output_dir)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()