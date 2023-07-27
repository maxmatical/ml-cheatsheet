from transformers import HfArgumentParser, TrainingArguments
from transformers import AutoTokenizer, GPT2Model
from datasets import load_dataset 

from dataclasses import dataclass, field
from rlhf.reward_model.reward_trainer import RewardTrainer
from rlhf.reward_model.reward_model import RewardModel
from rlhf.data.data import RewardDataCollatorWithPadding, preprocess_function


@dataclass
class RMTrainerArguments(TrainingArguments):
    """
    Adapt transformers.TrainingArguments
    """
    use_lion: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the LION optimizer instead of AdamW "
                "uses same hyperparam args (betas, wd, lr) as AdamW, except eps)."
            )
        },
    )
def main():
    parser = HfArgumentParser(RMTrainerArguments)

    trainer_args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    tokenizer.pad_token = tokenizer.eos_token

    collator = RewardDataCollatorWithPadding(tokenizer=tokenizer)

    lm_model = GPT2Model.from_pretrained("gpt2")
    lm_model.config.pad_token_id = tokenizer.eos_token_id

    model = RewardModel(lm_model)

    dataset = load_dataset("Anthropic/hh-rlhf")
    train_ds = dataset["train"]

    train_dataset = train_ds.map(
        preprocess_function, 
        batched=True, num_proc=1, 
        fn_kwargs={"tokenizer": tokenizer}
    )


    trainer = RewardTrainer(
        model=model,
        args = trainer_args,
        train_dataset = train_dataset,
        data_collator = collator
    )

    trainer.train()

if __name__ == '__main__':
    main()