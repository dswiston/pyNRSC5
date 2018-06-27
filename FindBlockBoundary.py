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
  # Filter looking for the expected bit pattern
  tmp = lfilter(refBits,1,refCarBits,axis=0)
  # Sum across the different reference carriers.Each sub should have the same 
  # sync pattern and summing increases SNR.
  tmp = tmp.sum(axis=1)
  # Reshape to 32bit block size
  tmp = tmp.reshape((int((numBitsPerBlk-1)/32),32))
  # Sum over blocks, increase SNR
  tmp = tmp.sum(axis=0)
  # The maximum index represents the block boundary location
  blkNdx = np.argmax(np.abs(tmp))
  # Use the phase of the correlation spike to indicate if locked in phase or 
  # out of phase w/the DBPSK signal
  if tmp[blkNdx] < 0:
    phsInv = True
  else:
    phsInv = False
  # Calculate the actual block boundary location after accounting for the 
  # filter group delay
  blkNdx = blkNdx + (refBits.size - 1) / 2 - 1
  blkNdx = int(blkNdx)
  
  return (blkNdx,phsInv)