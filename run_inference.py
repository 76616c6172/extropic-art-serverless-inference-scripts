#!/bin/python3
import modal, time, argparse, os

OUT_DIR = "/home/valar/model/images/pngs/"

if __name__ == "__main__":
    timeAtStartOfRun = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="1 or 2 for sd 1.5mj or 2 for sd2.1")
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
        steps = int(77)
        run_inference = modal.lookup(
            "serverless-gpu-worker-1", "run_large_diffusion_model"
        )
    else:
        prompt = args.prompt
        steps = int(25)
        run_inference = modal.lookup("serverless-gpu-worker-2", "run_sd2")

    # Run serverless inference job
    img_bytes = run_inference(prompt, seed, width, height, steps, scale)

    output_path = os.path.join(OUT_DIR, file_name)
    with open(output_path, "wb") as file:
        file.write(img_bytes)

    print(f"wrote data to {output_path}")
    timeAtCompletion = time.monotonic() - timeAtStartOfRun
    print(f"finished in: {timeAtCompletion:.2f} seconds")
