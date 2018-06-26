#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 11:10:41 2018

@author: dave
"""

import numpy as np
import matplotlib.pyplot as plt


def CoarseTimeSync(decData,msgSize,frameSize,prefixFilt,fs):
  
  #import pdb; pdb.set_trace()

  prefixSize = prefixFilt.size
  
  # Execute the auto-correlation for syncronization. Only need to compute it for one lag (msgSize)
  corrSig = decData[0:-msgSize] * np.conj(decData[msgSize-1:-1])
  # Reshape so that multiple frames can be easily summed
  corrSig = np.reshape(corrSig,(-1,frameSize))
  # Sum across multiple frames to increase effective SNR
  corrSig = np.sum(corrSig,axis=0)
  # Filter the output. We know the length of the cyclic prefix. Specifically look for repeating 
  # correlation of that length, effectively performing a matched filter like operation.
  filtOut = np.fft.ifft( np.fft.fft(corrSig)*np.fft.fft(prefixFilt,n=len(corrSig)) )
  # Grab the highest value, indicating the offset
  timeOffset = np.argmax(np.abs(filtOut))
  # Calculate the frequency error
  freqErr = -np.angle(filtOut[timeOffset]) * fs / (2 * np.pi * msgSize)
  
  # Trim off the incomplete symbol
  sigOut = np.concatenate((np.zeros(frameSize),decData[timeOffset-prefixSize-115-108:-1]))

  return(timeOffset,freqErr,sigOut)