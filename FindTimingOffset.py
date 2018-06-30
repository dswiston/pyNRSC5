#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 15:25:43 2018

@author: dave
"""

import numpy as np

def FindTimingOffset(input,fftSize):
  
  #import pdb; pdb.set_trace()
  # Concept is to calculate the phase change across subcarriers (frequency)
  # The estimator being used is a delay discriminator. It isn't the most 
  # accurate estimator possible but it is computationally efficient and works
  # at low SNRs

  # Calculate the vector difference between adjacent samples (delay discrim)
  vectDiff = input[:,0:-1] * np.conj(input)[:,1:]
  # Remove the subcarrier jump midway through. Could also possibly make use of
  # this to help increase frequency estimate accuracy.
  vectDiff = np.delete(vectDiff,10,axis=1)
  # Reference signals are BPSK.  Squaring removes 180deg phase transitions
  vectDiff = vectDiff**2
  # Create frequency estimate
  freqErr = np.angle(np.sum(vectDiff))/2
  
  phsPred = freqErr * 1092 / 19
  phsPredWrap = ( phsPred + np.pi) % (2 * np.pi ) - np.pi

  phsAct = input[:,0] * np.conj(input[:,-1])
  phsAct = np.angle(np.sum(phsAct**2))/2
  phsErr = phsAct - phsPredWrap
  phsErr = ( phsErr + np.pi/2) % ( np.pi ) - np.pi/2

  freqEst = (phsPred + phsErr)/1092*19
  
  # Frequency offset in frequency domain translates to a timing offset in the
  # time domain.  Perform conversion to time domain.
  timingErr = freqErr / (2*np.pi) / 19 * fftSize
  timingErr2 = freqEst / (2*np.pi) / 19 * fftSize

  print('Timing Error:  ' + str(timingErr))
  print('Timing Error2: ' + str(timingErr2))

  return timingErr2

'''
    0.06rad/sample
    0.06rad/sample * 363samples/sec / 2PIrad/cycle = cycles/sec
    2 * pi * k * timeErr / NFFT
    0.40142572795869574 = 2 * pi * 1 * timeErr / 2048
    timeErr = freqEst / (2*pi) * 2048
    1.04719 = 2 * pi * 19 * timeErr / 2048
'''