#!/bin/python3
import modal, argparse, os

OUT_DIR = "/home/valar/model/images/pngs/"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="select model pipeline, set 0 or 1")
    parser.add_argument("prompt", help="text prompt for the model")
    parser.add_argument("seed", help="seed used by noise generator")
    parser.add_argument("width", help="width of the final image in pixels")
    parser.add_argument("height", help="height of the final image pixels")
    parser.add_argument("steps", help="number of denoising steps")
    parser.add_argument("scale", help="guidance scale")
    parser.add_argument("jobid", help="the id for so output is saved as <jobid>.png")

    args = parser.parse_args()
    seed = int(args.seed)
    width = int(args.width)
    height = int(args.height)
    scale = int(args.scale)
    file_name = args.jobid + ".png"

    if args.model == "0":
        prompt = "mdjrny v4 style " + args.prompt
        steps = int(75)
        run_inference = modal.lookup("serverless-worker-1", "run_sd1_5")
    if args.model == "1":
        prompt = args.prompt
        steps = int(75)
        run_inference = modal.lookup("serverless-worker-4", "run_abyss_orange_mix")
    else:
        prompt = args.prompt
        steps = int(25)
        run_inference = modal.lookup("serverless-worker-2", "run_sd2_1")

    img_bytes = run_inference.call(prompt, seed, width, height, steps, scale)

    output_path = os.path.join(OUT_DIR, file_name)
    with open(output_path, "wb") as file:
        file.write(img_bytes)
