#!/usr/bin/env python3
import sys,os,re,datetime,random,torch
import numpy as np
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline, DDIMScheduler
from torch import autocast
from libsixel.encoder import Encoder
from SD import SDConfig

def drawFile(filename):
  if (filename != ''):
    if (os.path.isfile(filename) == True):
      encoder = Encoder()
      encoder.encode(filename)
  pass

def readBasefile(basefile, w, h):
  image = Image.open(basefile)
  if '.png' in basefile:
    image = image.convert('RGB')
  image = image.resize((w, h))
  return(image)
  pass

sd = SDConfig('config.txt')
sd.readConfigPreload()

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
  sd.model,
  revision="fp16",
  torch_dtype=torch.float16,
  use_auth_token=sd.token,
  scheduler=DDIMScheduler(
      beta_start=0.00085,
      beta_end=0.012,
      beta_schedule="scaled_linear",
      clip_sample=False,
      set_alpha_to_one=False,
  ),
)
pipe.to(sd.device)

with autocast(sd.device):
  for i in range(sd.loops):
    sd.readConfigOndemand()

    if (sd.basefile == ''):
      print("Setup your basefile in " + sd.config)
      exit()
      pass

    basefile_image = readBasefile(sd.basefile, sd.width, sd.height)

    prompts = [sd.prompt] * sd.concurs

    seed = sd.seed
    if (seed < 0):
      seed = random.randrange(0, 2147483647, 1)
    generator = torch.Generator(sd.device).manual_seed(seed)

    images = pipe(
      prompt=prompts,
      num_inference_steps=sd.iterations,
      guidance_scale=sd.guidance,
      strength=sd.strength,
      init_image=basefile_image,
      generator=generator
    ).images

    prefix = sd.getPrefix()
    j = 0
    for image in images:
      filename = prefix + '_' + str(i) + '_' + str(seed) + '_' + str(j) + '.png'
      image.save(filename)

      if (sd.draw == True):
        drawFile(filename)
      buf = "Width:" + str(sd.width) + ",Height:" + str(sd.height) + ",Iterations:" + str(sd.iterations) + ",Guidance:" + str(sd.guidance) + ",Strength:" + str(sd.strength) + ",Seed:" + str(seed) + ",Filename:" + filename + ",Prompt:\'" + sd.prompt + "\'"
      print(buf)
      sd.writeLogfile(buf)
      j = j + 1
      pass
    i = i + 1
    pass
  pass
