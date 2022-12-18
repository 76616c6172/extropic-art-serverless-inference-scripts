#!/bin/python3
import argparse, io, sys, modal
from http.client import PROCESSING

CACHE_PATH = "/root/model_cache"
GPU = modal.gpu.A100()
volume = modal.SharedVolume().persist("sd-testing")

stub = modal.Stub(
    "serverless-gpu-worker-2",
    image=modal.Image.debian_slim()
    .apt_install(["git"])
    .pip_install(
        [
            "git+https://github.com/huggingface/diffusers.git",
            "transformers",
            "accelerate",
            "scipy",
        ]
    ),
)


@stub.function(
    gpu=GPU,
    shared_volumes={CACHE_PATH: volume},
)
async def run_sd2(prompt, seed, width, height, steps, scale):
    import torch
    from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

    model_id = "stabilityai/stable-diffusion-2-1"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        width=width,
        height=height,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", help="text prompt for the model")
    parser.add_argument("seed", help="seed used by noise generator")
    parser.add_argument("width", help="width of the final image in pixels")
    parser.add_argument("height", help="height of the final image pixels")
    parser.add_argument("steps", help="number of denoising steps")
    parser.add_argument("scale", help="guidance scale")
    parser.add_argument("jobid", help="the id for so output is saved as <jobid>.png")

    args = parser.parse_args()

    prompt = args.prompt
    seed = int(args.seed)
    width = int(args.width)
    height = int(args.height)
    steps = int(25)
    scale = int(args.scale)
    file_name = args.jobid + ".png"

    with stub.run():
        png_data = run_sd2(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            scale=scale,
        )
        with open("output.png", "wb") as file:
            file.write(png_data)
