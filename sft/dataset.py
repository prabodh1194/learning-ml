import logging

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoTokenizer, TokenizersBackend

from sft.load_tinyllama import MODEL_DIR

PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Response:
{output}</s>"""

PROMPT_TEMPLATE_WITH_INPUT = """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}</s>"""

def tokenize_with_mask(
    example: dict, tokenizer: TokenizersBackend, max_length: int = 512
):
    prompt = format_example(example, skip_response=True)
    response = f"{example['output']}</s>"

    # tokenise separately
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)

    # combine
    input_ids = prompt_ids + response_ids

    labels = [-100] * len(prompt_ids) + response_ids

    input_ids = input_ids[:max_length]
    labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "labels": labels,
    }



def format_example(example: dict, skip_response: bool = False) -> str:
    has_input = example.get("input", "").strip()

    if skip_response:
        # Prompt only - no response
        if has_input:
            return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n"
        else:
            return f"### Instruction:\n{example['instruction']}\n\n### Response:\n"
    else:
        # Full template with response + </s>
        example["output"] = f"{example['output']}"
        if has_input:
            pr = PROMPT_TEMPLATE_WITH_INPUT.format(**example)
        else:
            pr = PROMPT_TEMPLATE.format(**example)
        # Restore original (remove </s> we added)
        example["output"] = example["output"][:-4]
        return pr


class AlpacaDataset(Dataset):
    def __init__(
        self,
        tokenizer: TokenizersBackend,
        max_length: int = 512,
        split: str = "train",
    ) -> None:
        raw = load_dataset("tatsu-lab/alpaca", split="train")

        self.examples = []
        for ex in raw:
            tokens = tokenize_with_mask(ex, tokenizer, max_length)
            self.examples.append(tokens)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


if __name__ == "__main__":
    loggers = [
        logging.getLogger(name).setLevel(level=logging.INFO)
        for name in logging.root.manager.loggerDict
    ]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")
    ds = AlpacaDataset(tokenizer, max_length=512, split="train")

    print(f"Dataset size: {len(ds)}")
    print(f"Sample tokens: {ds[0][:20]}...")
    print(f"Decoded: {tokenizer.decode(ds[0])[:200]}...")
