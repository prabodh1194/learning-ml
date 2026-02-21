from llama.model import LLaMA


def create_student(
    *,
    vocab_size: int,
):
    """
    Small LLaMA (~20M params vs teacher's 1.1B)
    """
    model = LLaMA(
        n_layers=4,
        vocab_size=vocab_size,
        dim=512,
        hidden_dim=1408,
        context_length=512,
        num_head=8,
        num_kv_head=2,
    )

    return model
