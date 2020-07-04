
import logging
import math

import torch
import torch.nn.functional as F
import io
import sys

class AverageMeter(object):
  """Computes and stores the average and current value"""
  def __init__(self, tmpfile=''):
    self.reset()

    if tmpfile == '':
      self.tmpwriter = None
    else:
      self.tmpwriter = open(tmpfile, 'w')

  def done(self):
    if tmpwriter != None:
      self.tmpwriter.close()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0
    self.history = []

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count
    for k in range(n):
      self.history.append(val)
      if self.tmpwriter != None:
        self.tmpwriter.write(str(val)+'\n')
        self.tmpwriter.flush()

