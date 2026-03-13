"""
Zero-shot classification with CLIP on CIFAR-10.

No labeled training data needed — just text descriptions of classes.
The model classifies images by finding the most similar text prompt.
"""

import logging

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import CLIPTokenizer

from clip.model import CLIP
from config import DATASETS_CACHE

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def build_model(checkpoint_path: str, device: str) -> CLIP:
    model = CLIP(
        input_C=3,
        patch_size=4,
        image_d_model=64,
        image_seq_len=65,
        image_mlp_dim=256,
        image_n_heads=4,
        image_n_layers=4,
        vocab_size=49152,
        text_d_model=64,
        text_n_heads=4,
        text_mlp_dim=256,
        embed_dim=64,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def encode_class_prompts(model: CLIP, device: str) -> torch.Tensor:
    """
    Encode all class names as text embeddings using the prompt template.
    Returns: (num_classes, embed_dim) — precomputed once, reused for every image.
    """
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    prompts = [f"a photo of a {cls}" for cls in CIFAR10_CLASSES]

    tokens = tokenizer(
        prompts,
        max_length=77,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    token_ids = tokens["input_ids"].to(device)  # (10, 77)

    with torch.no_grad():
        text_embeds = model.text_projection(
            model.text_encoder(token_ids)
        )  # (10, embed_dim)

    return text_embeds


def zero_shot_classify(
    model: CLIP, images: torch.Tensor, text_embeds: torch.Tensor
) -> torch.Tensor:
    """
    images:      (B, C, H, W)
    text_embeds: (num_classes, embed_dim)  ← precomputed, reused for every image

    returns: predicted class indices (B,)
    """
    with torch.no_grad():
        image_embeds = model.image_projection(
            model.image_encoder(images)
        )  # (B, embed_dim)

    similarities = image_embeds @ text_embeds.T  # (B, num_classes)
    return similarities.argmax(dim=-1)  # (B,)


def evaluate():
    device = "mps"
    checkpoint_path = "clip/checkpoints/clip_step_9600.pt"

    log.info(f"loading checkpoint: {checkpoint_path}")
    model = build_model(checkpoint_path, device)

    log.info("encoding class prompts...")
    text_embeds = encode_class_prompts(model, device)

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    test_set = datasets.CIFAR10(
        root=DATASETS_CACHE, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    correct, total = 0, 0

    for images, labels in test_loader:
        images = images.to(device)
        preds = zero_shot_classify(model, images, text_embeds)
        correct += (preds == labels.to(device)).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    log.info(f"zero-shot accuracy on CIFAR-10: {acc:.2f}% ({correct}/{total})")
    log.info("random baseline: 10.00%")


if __name__ == "__main__":
    evaluate()
