from llama.model import LLaMA
from sft.load_tinyllama import load


def load_teacher() -> LLaMA:
    model = load(freeze=True)

    return model