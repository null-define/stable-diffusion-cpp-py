from stable_diffusion_cpp import StableDiffusion
from PIL import Image
import numpy as np
import argparse


good_words = [
    "best quality",
    "fantasy",
    "highly detailed 8k UHD",
    "masterpiece",
]

parser = argparse.ArgumentParser(description='python cli.py -m mode_path -p prompt')
parser.add_argument("-m", required= True)
parser.add_argument("-p", required= True)
args = parser.parse_args()

sd = StableDiffusion(
    model_path=args.m,
)
img = sd.txt_to_img(prompt=good_words + [args.p])
img = sd.upscale_img(img, 4)
im = Image.fromarray(np.array(img))
im.save("img.jpg")
