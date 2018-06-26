#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 01:21:12 2018

@author: dave
"""

import numpy as np
from scipy.signal import lfilter

def FindBlockBoundary(refBits,refCarBits,numBitsPerBlk):
  
  # Search for block boundaries in the ref subcarriers
  tmp = lfilter(refBits,1,refCarBits,axis=0)
  tmp = tmp.sum(axis=1)
  tmp = tmp[0:-1]
  tmp = tmp.reshape((int((numBitsPerBlk-1)/32),32))
  tmp = tmp.sum(axis=0)
  blkNdx = np.argmax(np.abs(tmp))
  if tmp[blkNdx] < 0:
    phsInv = True
  else:
    phsInv = False
  blkNdx = blkNdx + (refBits.size - 1) / 2 - 1
  blkNdx = int(blkNdx)
  
  return (blkNdx,phsInv)