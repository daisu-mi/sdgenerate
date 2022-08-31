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

def readToken(file):
  ret = readConfig(file, "token")
  if (ret == ''):
    print("Setup your token in " + file)
    exit()
  return(ret)
  pass

def readPrompt(file):
  ret = readConfig(file, "prompt")
  if (ret == ''):
    print("Setup your prompt in " + file)
    exit()
  return(ret)
  pass

def readHeight(file):
  ret = readConfig(file, "height")
  if (ret == ''):
    ret = 512
  else:
    ret = int(ret)
  return(ret)
  pass

def readWidth(file):
  ret = readConfig(file, "width")
  if (ret == ''):
    ret = 512
  else:
    ret = int(ret)
  return(ret)
  pass

def readPics(file):
  ret = readConfig(file, "pics")
  if (ret == ''):
    ret = 1
  else:
    ret = int(ret)
  return(ret)
  pass

def readDraw(file):
  ret = readConfig(file, "draw")
  if (ret == ''):
    ret = 0
  else:
    ret = int(ret)
  return(ret)
  pass

def readHistory(file):
  ret = readConfig(file, "history")
  if (ret == ''):
      ret = 'history.txt'
  return(ret)
  pass

def writeHistory(file, buf):
  with open(file, 'a') as f:
    print(buf, file=f)
  pass

def drawFile(file):
  token = ""
  if (file != ''):
    if (os.path.isfile(file) == True):
        encoder = Encoder()
        encoder.encode(file)
  pass


if __name__ == '__main__':
  token = readToken(CONFIG_FILE)

  pipe = StableDiffusionPipeline.from_pretrained(MODEL_ID, revision="fp16", torch_dtype=torch.float16, use_auth_token=token)
  pipe.to(DEVICE)

  pics = 1
  width = 512
  height = 512
  steps = 150
  draw = 0

  with autocast(DEVICE):
    nums = 100
    for i in range(nums):
      print("loop:" + str(i))
      # read (re-read) config
      prompt = readPrompt(CONFIG_FILE)
      pics = readPics(CONFIG_FILE)
      draw = readDraw(CONFIG_FILE)
      width = readWidth(CONFIG_FILE)
      height = readHeight(CONFIG_FILE)
      history = readHistory(CONFIG_FILE)

      prompts = [prompt] * pics

      # storing random seed for reproducibility
      seed = random.randrange(0, 2147483647, 1)

      generator = torch.Generator(DEVICE).manual_seed(seed)

      images = pipe(prompts, num_inference_steps=steps, width=width, height=height, generator=generator)["sample"]

      prefix = getPrefix()

      j = 0
      for image in images:
        filename = prefix + '_' + str(i) + '_' + str(seed) + '_' + str(j) + '.png'
        image.save(filename)
        writeHistory(history, filename + ',' + prompt)
        if (draw == 1):
            drawFile(filename)
            print(filename)

        j = j + 1
      i = i + 1
