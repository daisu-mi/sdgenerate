import sys,os,datetime,random
import torch
from diffusers import StableDiffusionPipeline
from torch import autocast

MODEL_ID = "CompVis/stable-diffusion-v1-4"
DEVICE = "cuda"
TOKEN_FILE = "token.txt"
PROMPT_FILE = "prompt.txt"
HISTORY_FILE = "history.txt"

def getPrefix():
  t_delta = datetime.timedelta(hours=9)
  JST = datetime.timezone(t_delta, 'JST')
  return(datetime.datetime.now(JST).strftime('%Y%m%d_%H%M%S'))
  pass

def readToken(file):
  token = ""
  if (file != ''):
    if (os.path.isfile(file) == True):
      with open(file) as f:
        token = f.readline().rstrip('\r').rstrip('\n')

  return(token)
  pass

def readPrompt(file):
  prompt = ""
  if (file != ''):
    if (os.path.isfile(file) == True):
      with open(file) as f:
        prompt = f.readline().rstrip('\r').rstrip('\n')

  return(prompt)
  pass

def writeHistory(file, buf):
  with open(file, 'a') as f:
    print(buf, file=f)
  pass

prefix = getPrefix()
token = readToken(TOKEN_FILE)

if (token == ''):
        print("Setup your token in " + TOKEN_FILE)
        exit()

pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=token)
pipe.to(DEVICE)

# GeForce 3070
width = 512
height = 512

# A100
#width = 768
#height = 512

with autocast(DEVICE):
  nums = 100
  for i in range(nums):
    print("loop:" + str(i))
    # load (reload) config
    prompt = readPrompt(PROMPT_FILE)
    if (prompt == ''):
        exit()

    prompts = [prompt] * 1

    # store rando seed
    seed = random.randrange(0, 2147483647, 1)

    generator = torch.Generator(DEVICE).manual_seed(seed)

    images = pipe(prompts, num_inference_steps=200, width=width, height=height, generator=generator)["sample"]

    prefix = getPrefix()

    j = 0
    for image in images:
        filename = prefix + '_' + str(i) + '_' + str(seed) + '_' + str(j) + '.png'
        image.save(filename)
        writeHistory(HISTORY_FILE, filename + ',' + prompt)
        j = j + 1
    i = i + 1
