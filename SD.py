#!/usr/bin/env python3
import sys,os,re,datetime

class SDConfig(object):
  device = "cuda"
  model = ''
  token = ''
  config = 'config.txt'
  width = 512
  height = 512
  concurs = 1
  loops = 1
  iterations = 50
  prompt = ''
  seed = -1
  draw = False
  guidance = 7.5
  strength = 0.5
  basefile = ''
  logfile = 'log.txt'

  def __init__(self, config):
    self.config = config
    pass

  def getPrefix(self):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    return(datetime.datetime.now(JST).strftime('%Y%m%d_%H%M%S'))
    pass

  def getDate(self):
    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    return(datetime.datetime.now(JST).strftime('%Y-%m-%d %H:%M:%S'))
    pass

  def readConfig(self, item, recursive=False):
    ret = ''
    with open(self.config) as f:
      for line in f:
        line = line.rstrip('\r').rstrip('\n')
        match = '^' + str(item) + '\s*:\s*(.+)';
        result = re.search(match, line)
        if result:
          if (recursive == True):
             ret = ret + ' ' + result.group(1)
          else:
             ret = result.group(1)
             break
    return(ret)
    pass

  def readConfigModel(self):
    ret = self.readConfig("model")
    if (ret != ''):
      self.model = ret
    else:
      print("Setup your model in " + self.config)
      exit()
    return(ret)
    pass

  def readConfigToken(self):
    ret = self.readConfig("token")
    if (ret != ''):
      self.token = ret
    else:
      print("Setup your token in " + self.config)
      exit()
    return(ret)
    pass

  def readConfigPrompt(self):
    # Prompt field was read incrementary 
    ret = self.readConfig("prompt", True)
    if (ret != ''):
      self.prompt = ret
    else:
      print("Setup your prompt in " + self.config)
      exit()
    pass

  def readConfigDevice(self):
    ret = self.readConfig("device")
    if (ret != ''):
      self.prompt = ret
    pass

  def readConfigLoops(self):
    ret = self.readConfig("loops")
    if (ret != ''):
      self.loops = int(ret)
    pass

  def readConfigConcurs(self):
    ret = self.readConfig("concurs")
    if (ret != ''):
      self.concurs = int(ret)
    pass

  def readConfigIterations(self):
    ret = self.readConfig("iterations")
    if (ret != ''):
      self.iterations = int(ret)
    pass

  def readConfigHeight(self):
    ret = self.readConfig("height")
    if (ret != ''):
      self.height = int(ret)
    pass

  def readConfigWidth(self):
    ret = self.readConfig("width")
    if (ret != ''):
      self.width = int(ret)
    pass

  def readConfigDraw(self):
    ret = self.readConfig("draw")
    if (ret != ''):
      self.draw = True
    else:
      self.draw = False
    pass

  def readConfigSeed(self):
    ret = self.readConfig("seed")
    if (ret != ''):
      self.seed = int(ret)
    pass

  def readConfigGuidance(self):
    ret = self.readConfig("guidance")
    if (ret != ''):
      self.guidance = float(ret)
    pass

  def readConfigStrength(self):
    ret = self.readConfig("guidance")
    if (ret != ''):
      self.guidance = float(ret)
    pass

  def readConfigBasefile(self):
    ret = self.readConfig("basefile")
    if (ret != ''):
      self.basefile = ret
    pass

  def readConfigLogfile(self):
    ret = self.readConfig("logfile")
    if (ret != ''):
      self.logfile = ret
    pass

  def writeLogfile(self, input):
    output = self.getDate() + ',' + str(input)
    with open(self.logfile, 'a') as f:
      print(output, file=f)
    pass

  def readConfigPreload(self):
    self.readConfigModel()
    self.readConfigToken()
    self.readConfigDevice()
    self.readConfigLoops()

  def readConfigOndemand(self):
    self.readConfigPrompt()
    self.readConfigWidth()
    self.readConfigHeight()
    self.readConfigIterations()
    self.readConfigConcurs()
    self.readConfigSeed()
    self.readConfigDraw()
    self.readConfigGuidance()
    self.readConfigStrength()
    self.readConfigBasefile()
    self.readConfigLogfile()
