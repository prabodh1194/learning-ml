"""
Layers:

1) RoPE
2) RMSNorm
3) GQA
4) SwiGLU FFN
5) LLaMABlock
"""

import logging

logging.basicConfig(
    level=logging.DEBUG,
    format="%(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("llama")
