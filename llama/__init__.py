"""
Layers:

1) RoPE
2) RMSNorm
3) GQA
4) SwiGLU FFN
5) LLaMABlock
"""

import logging
from datetime import datetime


class MicrosecondsFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created)
        return ct.strftime("%Y-%m-%dT%H:%M:%S.%f")


handler = logging.StreamHandler()
handler.setFormatter(
    MicrosecondsFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logging.root.addHandler(handler)
logging.root.setLevel(logging.INFO)

logger = logging.getLogger("llama")
