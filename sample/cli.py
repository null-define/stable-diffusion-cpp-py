from stable_diffusion_cpp import StableDiffusion
from PIL import Image
import numpy as np
import argparse


good_words = [
    "best quality",
    "fantasy",
    "highly detailed 8k UHD",
    "dramatic cinematic lighting",
    "insanely detailed face",
    "anatomically correct",
    "symmetrical proportions", 
    "masterpiece",
    "HDR",
    "trending on pixiv",
]

prompt_words = [
    "hatsune miku",
    "1 girl",
    "cute",
    "sunny day"
    "at beach",
    "happy",
    "beautiful clothes",
    "long hair",
    "look at viewer"
]

neg_words = [
    "poorly Rendered face",
    "poorly drawn face",
    "poor facial details",
    "poorly drawn hands",
    "poorly rendered hands",
    "low resolution",
    "Images cut out at the top, left, right, bottom.",
    "bad composition",
    "mutated body parts",
    "blurry",
    "disfigured",
    "over saturated",
    "bad anatomy",
    "deformed body features",
    "logo",
]


parser = argparse.ArgumentParser(description='python cli.py -m mode_path -p prompt')
parser.add_argument("-m", required= True)
parser.add_argument("-u", required= False)
parser.add_argument("-p", required= True)
args = parser.parse_args()

sd = StableDiffusion(
    model_path=args.m,
    esrgan_path=args.u if args.u else ""
)

import random

img = sd.txt_to_img(prompt=good_words + prompt_words, negative_prompt=neg_words, width= 1024, height=704, seed=random.randint(-32768, 65536))
img2 = sd.upscale_img(img, 4)
im = Image.fromarray(np.array(img2))
im.save("img.png", bitmap_format = 'png', optimize= False, subsampling=0, quality=100)
