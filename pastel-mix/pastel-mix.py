#!/bin/python3
from sched import scheduler
import modal, io, os, time, argparse

VOLUME = modal.SharedVolume().persist("volume-6-pastel-mix")
CACHE_PATH = "/root/model_cache"
MODEL_ID = "andite/pastel-mix"


stub = modal.Stub(
    "worker-6-pastel-mix",
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
    gpu="any",
    shared_volumes={CACHE_PATH: VOLUME},
    secret=modal.Secret.from_name("my-huggingface-secret"),
)
async def run_pastel_mix(prompt, seed, width, height, steps, scale):
    import torch as torch
    from torch import float16
    from diffusers import StableDiffusionPipeline
    from diffusers import EulerAncestralDiscreteScheduler

    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
        torch_dtype=float16,
        cache_dir=CACHE_PATH,
        local_files_only=True,
        device_map="auto",
        safety_checker=None,
    )

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_pretrained(MODEL_ID, subfolder="scheduler")

    image = pipe(
        prompt,
        negative_prompt="lowres, ((bad anatomy)), ((bad hands)), text, missing finger, extra digits, fewer digits, blurry, ((mutated hands and fingers)), (poorly drawn face), ((mutation)), ((deformed face)), (ugly), ((bad proportions)), ((extra limbs)), extra face, (double head), (extra head), ((extra feet)), monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, ((jpeg artifacts))",
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

    # Run serverless inference job
    with stub.run():

        img_bytes = run_pastel_mix.call(prompt, seed, width, height, steps, scale)
        output_path = os.path.join("./", file_name)

        with open(output_path, "wb") as f:
            f.write(img_bytes)

    timeAtCompletion = time.monotonic() - timeAtStartOfRun
    print(f"finished in: {timeAtCompletion:.2f} seconds")
