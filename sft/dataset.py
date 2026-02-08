import logging

from datasets import load_dataset
from sympy import trunc
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


def format_example(example: dict) -> str:
    if example.get("input", "").strip():
        return PROMPT_TEMPLATE_WITH_INPUT.format(**example)
    return PROMPT_TEMPLATE.format(**example)


class AlpacaDataset(Dataset):
    def __init__(
        self, tokenizer: TokenizersBackend, max_length: int = 512 * 3, split: str = "train"
    ) -> None:
        raw = load_dataset("tatsu-lab/alpaca", split="train")

        self.examples = []
        for ex in raw:
            text = format_example(ex)
            tokens = tokenizer.encode(text, truncating=True, max_length=max_length)
            self.examples.append(tokens)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


if __name__ == "__main__":
    loggers = [logging.getLogger(name).setLevel(level=logging.INFO) for name in logging.root.manager.loggerDict]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR / "models/tinyllama-1.1b")
    ds = AlpacaDataset(tokenizer, max_length=512, split="train")

    print(f"Dataset size: {len(ds)}")
    print(f"Sample tokens: {ds[0][:20]}...")
    print(f"Decoded: {tokenizer.decode(ds[0])[:200]}...")
