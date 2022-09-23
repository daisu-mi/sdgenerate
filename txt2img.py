#!/usr/bin/env python3
import sys,os,re,datetime,random,torch
import numpy as np
from diffusers import StableDiffusionPipeline, DDIMScheduler
from torch import autocast
from libsixel.encoder import Encoder
from SD import SDConfig

def drawFile(filename):
  if (filename != ''):
    if (os.path.isfile(filename) == True):
      encoder = Encoder()
      encoder.encode(filename)
  pass


sd = SDConfig('config.txt')
sd.readConfigPreload()

pipe = StableDiffusionPipeline.from_pretrained(
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
    prompts = [sd.prompt] * sd.concurs

    seed = sd.seed
    if (seed < 0):
      seed = random.randrange(0, 2147483647, 1)
    generator = torch.Generator(sd.device).manual_seed(seed)

    images = pipe(
      prompt=prompts,
      num_inference_steps=sd.iterations,
      width=sd.width,
      height=sd.height,
      guidance_scale=sd.guidance,
      generator=generator
    ).images

    prefix = sd.getPrefix()
    j = 0
    for image in images:
      filename = prefix + '_' + str(i) + '_' + str(seed) + '_' + str(j) + '.png'
      image.save(filename)

      if (sd.draw == True):
        drawFile(filename)
      buf = "Width:" + str(sd.width) + ",Height:" + str(sd.height) + ",Iterations:" + str(sd.iterations) + ",Guidance:" + str(sd.guidance) + ",Seed:" + str(seed) + ",Filename:" + filename + ",Prompt:\'" + sd.prompt + "\'"
      print(buf)
      sd.writeLogfile(buf)
      j = j + 1
      pass
    i = i + 1
    pass
  pass

print(sd.prompt)
