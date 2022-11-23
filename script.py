#!/bin/python3

import argparse

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("prompt", help="the prompt used to guide the model")
	parser.add_argument("seed", help="the seed used by noise generator")
	parser.add_argument("width", help="the width of the image")
	parser.add_argument("height", help="the height of the image")
	parser.add_argument("steps", help="the number of denoising steps")
	parser.add_argument("scale", help="the guidance scale")

	args = parser.parse_args()

	PROMPT = args.prompt
	SEED  = args.seed
	WIDTH = args.width
	HEIGHT = args.height
	STEPS = args.steps
	SCALE = args.scale

	print(PROMPT)
	print(SEED)
	print(f"{WIDTH} by {HEIGHT}")
	print(f"steps: {STEPS}")
	print(f"guidance_scale: {SCALE}")