from sft.load_tinyllama import load


def load_teacher():
    model = load(freeze=True)

    return model