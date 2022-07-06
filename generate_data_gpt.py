"""
take `intent_sample_generation_data_path`
sample k queries per intent to use as prompts
given prompt, generate n new queries for that intent
optionally: dedupe
concat with intent_training_data_path eg kajabi small/kajabi large
write new training file of train data + synthetic data

example usage: note will need at least a T4 gpu

python -m scripts.intent.gpt_generate_intent_queries \
    --intent_training_data_path {data-dir}/kajabi_train_large.jsonl \
    --intent_sample_generation_data_path {data-dir}/kajabi_train_large.jsonl \
    --synthetic_data_csv_fname kajabi_synthetic_data_from_large_50_samples.csv \
    --prompt_sample_size 8 \
    --generate_samples 50 \
    --dedupe

"""

import argparse
import json
import logging
import random

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPTJForCausalLM

from mlds.utils.utils import from_jsonl

logger = logging.getLogger(__name__)


class BadGeneratedTexError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__()
        self.message = message


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--intent_training_data_path",
        type=str,
        help="Path to intent data json file for training,\
             eg kajabi_small, kajabi_large",
        required=True,
    )
    parser.add_argument(
        "--intent_sample_generation_data_path",
        type=str,
        help="Path to intent data json file used to \
            generate synthetic queries from intents",
        required=True,
    )
    parser.add_argument(
        "--synthetic_data_csv_fname",
        type=str,
        help="name for generated synthetic data csv",
        required=True,
    )
    parser.add_argument(
        "--prompt_sample_size",
        type=int,
        help="Number of query/intent pairs to sample for generating prompt for GPT",
        required=False,
        default=8,
    )

    parser.add_argument(
        "--generate_samples",
        type=int,
        help="Number of synthetic queries you want to generate from prompt",
        required=False,
        default=10,
    )
    parser.add_argument(
        "--dedupe", action="store_true", help="Whether to remove duplicate generated queries"
    )

    args = parser.parse_args()

    NUM_QUESTIONS_TO_SAMPLE = args.prompt_sample_size
    NUM_QUESTIONS_TO_GENERATE = args.generate_samples
    synthetic_data_csv_fname = args.synthetic_data_csv_fname
    DEDUPE = args.dedupe

    # need a GPU with at least 16gb mem + tensorcores for fp16
    # otherwise will be too slow, luckily a single T4 will suffice
    print("loading model and tokenizer")
    model = GPTJForCausalLM.from_pretrained(
        "EleutherAI/gpt-j-6B", revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
    ).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
    print("loaded")

    # load data for generating prompts
    data = from_jsonl(args.intent_sample_generation_data_path)

    # get data in a format we want
    intents_dict = {}
    for d in data:
        intent = d["output"]["label"]
        question = d["input"]["sequence_features"][0]["text"]
        if not intent:
            continue
        if intents_dict.get(intent):
            intents_dict[intent].append(question)
        else:
            intents_dict[intent] = [question]

    # generate synthetic queries from intent
    generated_data = []
    for intent, questions in tqdm(intents_dict.items()):
        # generate prompt for GPT
        prompt = """Generate query from the following intent"""
        # sample questions
        sampled_questions = (
            random.sample(questions, NUM_QUESTIONS_TO_SAMPLE)
            if len(questions) > NUM_QUESTIONS_TO_SAMPLE
            else questions
        )
        for question in sampled_questions:
            prompt += f"\nintent: {intent} "
            prompt += f"query: {question}"
        # add final prompt
        prompt += f"\nintent: {intent} "
        prompt += "query:"
        # generate text
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        input_len = input_ids.shape[1]
        for _ in range(NUM_QUESTIONS_TO_GENERATE):
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                top_k=50,
                top_p=0.5,
                temperature=1.0,
                num_return_sequences=3,
                max_length=input_len + 64,
                pad_token_id=tokenizer.eos_token_id,
            )
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            try:
                # sometimes text generation might fail
                generated_query = gen_text.split(prompt)[1].split("\n")[0].strip()
                generated_data.append({"intent": intent, "query": generated_query})
            except BadGeneratedTexError:
                error_messaige = f"generated query not found, generated text=\n\n {gen_text}"
                logging.exception(error_messaige)
                continue

    # generate for analysis
    synthetic_data_df = pd.DataFrame(generated_data)
    synthetic_data_df.to_csv(synthetic_data_csv_fname, index=False)

    # adding synthetic data to real training data
    if DEDUPE:
        synthetic_data_df = synthetic_data_df.drop_duplicates()
    generated_data = synthetic_data_df.to_dict("records")

    # concatenate with train file
    train_intents_fname = args.intent_training_data_path
    train_data = from_jsonl(train_intents_fname)
    for d in generated_data:
        intent = d["intent"]
        query = d["query"]
        mlds_format = {
            "input": {"sequence_features": [{"text": query}]},
            "output": {"label": intent},
        }
        train_data.append(mlds_format)

    # write to jsonls
    out_fname = (
        "kajabi_data_large/kajabi_train_small_synthetic_data_from_large_more_samples_dedupe.jsonl"
    )
    with open(out_fname, "w") as outfile:
        for d in train_data:
            json.dump(d, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    main()
