#!/bin/python3
import modal, argparse, io, os, time

VOLUME = modal.SharedVolume().persist("worker-volume-2")
MODEL_ID = "stabilityai/stable-diffusion-2-1"
CACHE_PATH = "/root/model_cache"

stub = modal.Stub(
    "serverless-worker-2",
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
    gpu=modal.gpu.A100(),
    shared_volumes={CACHE_PATH: VOLUME},
    secret=modal.Secret.from_name("my-huggingface-secret"),
)
async def run_sd2_1(prompt, seed, width, height, steps, scale):
    import torch as torch
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

    # EulerDiscreteScheduler
    # scheduler = EulerDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    image = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=scale,
        width=width,
        height=height,
        generator=torch.Generator("cuda").manual_seed(seed),
    ).images[0]

    # Save and return the final image
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


if __name__ == "__main__":
    timeAtStartOfRun = time.monotonic()

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
    steps = int(args.steps)
    scale = int(args.scale)
    file_name = args.jobid + ".png"

    with stub.run():
        png_data = run_sd2_1(
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            scale=scale,
        )
        with open("output.png", "wb") as file:
            file.write(png_data)

    timeAtCompletion = time.monotonic() - timeAtStartOfRun
    print(f"finished in: {timeAtCompletion:.2f} seconds")
