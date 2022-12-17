#!/bin/python3
import modal, time, argparse, os

if __name__ == "__main__":
	OUT_DIR = "/home/valar/model/images/pngs/"
	run_mj_sd = modal.lookup("serverless-gpu-worker-1", "run_large_diffusion_model")
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
	prompt = "mdjrny v4 style " + args.prompt
	seed  = int(args.seed)
	width = int(args.width)
	height = int(args.height)
	steps = int(77)
	scale = int(args.scale)
	file_name = args.jobid + ".png"
	# Run serverless inference job
	img_bytes = run_mj_sd(prompt, seed, width, height, steps, scale)
	output_path = os.path.join(OUT_DIR, file_name)
	with open(output_path, "wb") as f:
		f.write(img_bytes)
	print(f"wrote data to {output_path}")
	timeAtCompletion = time.monotonic() - timeAtStartOfRun
	print(f"finished in: {timeAtCompletion:.2f} seconds")