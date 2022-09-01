#!/usr/bin/env python3
import sys,os,re,datetime,random,torch
from diffusers import StableDiffusionPipeline
from torch import autocast
from libsixel.encoder import Encoder

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
CONFIG_FILE = "config.txt"

def getPrefix():
  t_delta = datetime.timedelta(hours=9)
  JST = datetime.timezone(t_delta, 'JST')
  return(datetime.datetime.now(JST).strftime('%Y%m%d_%H%M%S'))
  pass

def getDate():
  t_delta = datetime.timedelta(hours=9)
  JST = datetime.timezone(t_delta, 'JST')
  return(datetime.datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S'))
  pass

def readConfig(file, item):
  ret = ''
  with open(file) as f:
    for line in f:
      line = line.rstrip('\r').rstrip('\n')
      match = '^' + str(item) + '\s*:\s*(.+)';
      result = re.search(match, line)
      if result:
        ret = result.group(1)
        break
  return(ret)
  pass

def readConfigToken(file):
  ret = readConfig(file, "token")
  if (ret == ''):
    print("Setup your token in " + file)
    exit()
  return(ret)
  pass

def readConfigLoops(file):
  ret = readConfig(file, "loops")
  if (ret == ''):
    ret = 1
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigWidth(file):
  ret = readConfig(file, "width")
  if (ret == ''):
    ret = 512
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigConcurs(file):
  ret = readConfig(file, "pics")
  if (ret == ''):
    ret = 1
  else:
    ret = int(ret)
  return(ret)
  pass
  if (ret == ''):
    exit()
  return(ret)
  pass

def readConfigPrompt(file):
  ret = readConfig(file, "prompt")
  if (ret == ''):
    print("Setup your prompt in " + file)
    exit()
  return(ret)
  pass

def readConfigHeight(file):
  ret = readConfig(file, "height")
  if (ret == ''):
    ret = 512
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigWidth(file):
  ret = readConfig(file, "width")
  if (ret == ''):
    ret = 512
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigConcurs(file):
  ret = readConfig(file, "concurs")
  if (ret == ''):
    ret = 1
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigDraw(file):
  ret = readConfig(file, "draw")
  if (ret == ''):
    ret = 0
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigSeed(file):
  ret = readConfig(file, "seed")
  if (ret == ''):
    ret = -1
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigIterations(file):
  ret = readConfig(file, "iterations")
  if (ret == ''):
    ret = 50
  else:
    ret = int(ret)
  return(ret)
  pass

def readConfigGuidance(file):
  ret = readConfig(file, "guidance")
  if (ret == ''):
    ret = 7.5
  else:
    ret = float(ret)
  return(ret)
  pass


def readConfigLog(file):
  ret = readConfig(file, "log")
  if (ret == ''):
      ret = 'log.txt'
  return(ret)
  pass

def writeLog(file, input):
  output = getDate() + ',' + str(input)
  with open(file, 'a') as f:
    print(output, file=f)
  pass

def drawFile(file):
  token = ""
  if (file != ''):
    if (os.path.isfile(file) == True):
        encoder = Encoder()
        encoder.encode(file)
  pass

def doStableDiffusion():
  token = readConfigToken(CONFIG_FILE)

  pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=token)
  pipe.to(DEVICE)

  with autocast(DEVICE):
    loops = readConfigLoops(CONFIG_FILE)
    for i in range(loops):
      print("Loop:" + str(i))

      # read parameters from config file,
      prompt     = readConfigPrompt(CONFIG_FILE)
      width      = readConfigWidth(CONFIG_FILE)
      height     = readConfigHeight(CONFIG_FILE)
      iterations = readConfigIterations(CONFIG_FILE)
      concurs    = readConfigConcurs(CONFIG_FILE)
      draw       = readConfigDraw(CONFIG_FILE)
      seed       = readConfigSeed(CONFIG_FILE)
      log        = readConfigLog(CONFIG_FILE)
      guidance   = readConfigGuidance(CONFIG_FILE)

      # applying concursrent tasks
      prompts = [prompt] * concurs

      # storing random seed for reproducibility
      if (seed < 0):
        seed = random.randrange(0, 2147483647, 1)
      generator = torch.Generator(DEVICE).manual_seed(seed)

      images = pipe(prompts, num_inference_steps=iterations, guidance=guidance, width=width, height=height, generator=generator)["sample"]

      prefix = getPrefix()

      j = 0
      for image in images:
        filename = prefix + '_' + str(i) + '_' + str(seed) + '_' + str(j) + '.png'
        image.save(filename)

        buf = "Width:" + str(width) + ",Height:" + str(height) + ",Iterations:" + str(iterations) + ",Guidance:" + str(guidance) + ",Seed:" + str(seed) + ",Filename:" + filename + ",Prompt:\'" + prompt + "\'"

        writeLog(log, buf)

        if (draw == 1):
            drawFile(filename)

        print(buf)

        j = j + 1
      i = i + 1
  pass

if __name__ == '__main__':
  doStableDiffusion()
