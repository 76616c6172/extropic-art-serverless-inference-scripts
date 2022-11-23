import io
import os
import time

import modal

from typing import Optional

stub = modal.Stub("serverless-worker")
volume = modal.SharedVolume().persist("serverless-worker-volume")

CACHE_PATH = "/root/model_cache"
OUTPUT_DIR = "/tmp/"
GPU = modal.gpu.A100()


#"runwayml/stable-diffusion-v1-5",

@stub.function(
	# gpu=True,
	gpu = GPU,
	image =(
		modal.Image.debian_slim()
		.run_commands(["pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu117"])
		.pip_install(["diffusers", "transformers", "scipy", "ftfy", "accelerate"])
	),
	shared_volumes={CACHE_PATH: volume},
	secret=modal.Secret.from_name("my-huggingface-secret"),
)
async def run_serverless_inference(prompt: str, seed, width, height, steps, scale):
	from diffusers import StableDiffusionPipeline
	from diffusers import DDIMScheduler
	from torch import float16
	import torch as torch

	pipe = StableDiffusionPipeline.from_pretrained(
			"prompthero/openjourney",
    	use_auth_token=os.environ["HUGGINGFACE_TOKEN"],
    	# revision="fp16",
   	 	torch_dtype=float16,
  	  cache_dir=CACHE_PATH,
  	  device_map="auto",
			# scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
			# scheduler = DDIMScheduler(
		#	scheduler = DDIMScheduler(
		#		beta_start=0.0001,
		#		beta_end=0.02,
		#		beta_schedule='scaled_linear',
		#		clip_sample=False,
		#		set_alpha_to_one= False,
		#		steps_offset=0
		#	)
		scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False),
		safety_checker=None,
	)

	generator = torch.Generator("cuda").manual_seed(seed)

	# seed = seed
	image = pipe(prompt, num_inference_steps=steps, guidance_scale=scale,
	width=width, generator=generator, height=height).images[0]

	# Convert PIL Image to PNG byte array.
	buf = io.BytesIO()
	image.save(buf, format="PNG")
	img_bytes = buf.getvalue()

	# if channel_name:
	    # `post_to_slack` is implemented further below.
	    # post_image_to_slack(prompt, channel_name, img_bytes)

	return img_bytes



# Add code here that actually runs the model and returns the image in a form that can be saved to disk
@stub.function # this is a "decorator" and in this codebase is needed to run the function remotely
def run_serverless_inference_testing(x):
	# Any stdout happening inside a stub function are happening remotely and go into the modal logs automatically!
	print("[modal] running serverless inference remotely")
	return x + 100

if __name__ == "__main__":

	timeAtStartOfRun = time.monotonic()


	PROMPT = "mdjrny v4 style " + "Portrait of Nana Mizuki, league of legends character art"
	SEED  = 554641278
	WIDTH = 512
	HEIGHT = 768
	STEPS = 75
	SCALE = 7

	# Running remotely
	with stub.run():

		img_bytes = run_serverless_inference(PROMPT, SEED, WIDTH, HEIGHT, STEPS, SCALE)
		output_path = os.path.join(OUTPUT_DIR, "output.png")

		with open(output_path, "wb") as f:
			f.write(img_bytes)

		timeAtCompletion = time.monotonic() - timeAtStartOfRun
		print(f"wrote data to {output_path}")
		print(f"finished in: {timeAtCompletion:.2f} seconds")