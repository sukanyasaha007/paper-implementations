"""IMDb prompts: the first few tokens of a review, which the policy learns to continue positively."""
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def create_prompt(review, tokenizer, prompt_length):
    tokens = tokenizer.tokenize(review)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    prompt_ids = token_ids[:prompt_length]

    prompt = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(prompt_ids))
    return {
        'input_ids': prompt_ids,
        'query': prompt,
    }


def build_prompt_dataset(dataset_name, split, tokenizer, prompt_length, num_prompts, seed):
    # IMDb splits are sorted by label (all negative reviews first), so shuffle before taking a subset
    dataset = load_dataset(dataset_name, split=split).shuffle(seed=seed).select(range(num_prompts))
    prompt_dataset = dataset.map(lambda sample: create_prompt(sample['text'], tokenizer, prompt_length))
    prompt_dataset = prompt_dataset.remove_columns(['text', 'label'])
    prompt_dataset.set_format(type='torch', columns=['input_ids'], output_all_columns=True)
    return prompt_dataset


def collator(prompts):
    # Keep variable-length prompts as a list of tensors instead of padding them into one tensor
    out = defaultdict(list)
    for prompt in prompts:
        for k, v in prompt.items():
            out[k].append(v)
    return dict(out)
