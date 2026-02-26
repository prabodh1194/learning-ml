"""
Serve the merged TinyLlama model with vLLM + Ray Serve.

Run: serve run sft_hf.serve:app
"""

from ray import serve
from fastapi import FastAPI
from pydantic import BaseModel
from vllm import LLM, SamplingParams

app = FastAPI()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128


class GenerateResponse(BaseModel):
    text: str


@serve.deployment
@serve.ingress(app)
class VLLMService:
    def __init__(self, model_path: str = "/Users/pbd/personal/learning-ml/sft_hf/sft-merged-hf", tensor_parallel_size: int = 1):

        self.llm = LLM(model=model_path, tensor_parallel_size=tensor_parallel_size)
        self.SamplingParams = SamplingParams

    @app.post("/generate")
    async def generate(self, req: GenerateRequest) -> GenerateResponse:
        params = self.SamplingParams(max_tokens=req.max_tokens)
        outputs = self.llm.generate([req.prompt], params)
        return GenerateResponse(text=outputs[0].outputs[0].text)


app = VLLMService.bind()
