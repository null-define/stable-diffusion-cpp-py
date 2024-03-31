from stable_diffusion_cpp import StableDiffusion
from PIL import Image
import numpy as np


good_words = [
    "best quality",
    "fantasy",
    "highly detailed 8k UHD",
    "masterpiece",
]

sd = StableDiffusion(
    model_path="",
    esrgan_path="",
)
img = sd.txt_to_img(prompt=good_words + [""])
img = sd.upscale_img(img, 4)
im = Image.fromarray(np.array(img))
im.save("img.jpg")
