#!/bin/python3
"""
SD 2.0
"""
import io
import sys
import modal
from http.client import PROCESSING

CACHE_PATH = "/root/model_cache"
GPU = modal.gpu.A100()
volume = modal.SharedVolume().persist("just-testing")

stub = modal.Stub(
    image=modal.Image.debian_slim()
    .apt_install(["git"])
    .pip_install(
        [
            "git+https://github.com/huggingface/diffusers.git",
            "transformers",
            "accelerate",
            "scipy",
        ]
    )
)


@stub.function(gpu=GPU,
	shared_volumes={CACHE_PATH: volume},
)
def run(prompt):
    import torch
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

    model_id = "stabilityai/stable-diffusion-2"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    image = pipe(prompt, height=768, width=768).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()

PROMPT = "Beautiful character art of the Pokemon Eevee, digital art"

if __name__ == "__main__":
    with stub.run():
        png_data = run(PROMPT)
        with open("output.png", "wb") as f:
            f.write(png_data)