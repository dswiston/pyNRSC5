#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:25:09 2018

@author: dave
"""
import numpy as np
from scipy.signal import lfilter
from scipy.signal import hilbert


class CostasLoop:

  filtErr = []      # 
                    # Size 
  phsVal = []       # 
                    # Size 
  errSig = []       # 
                    # Size 
  loopFilterA = []  # 
                    # Size 
  loopFilterB = []  # 
                    # Size 

  def __init__(self,loopFilter,filterSize,bufferSize):
    self.filtErr = np.array([])
    self.phsVal = 0
    self.bufferSize = bufferSize
    self.filterSize = filterSize
    self.errSig = np.zeros(filterSize)
    self.output = np.zeros(bufferSize,dtype=np.complex64)
    self.phsVal = np.zeros(bufferSize,dtype=np.complex64)
    if len(loopFilter) == 0:
      self.loopFilterA = 1
      self.loopFilterB = np.ones(filterSize) / filterSize
    elif np.ndim(loopFilter) == 1:
      self.loopFilterA = 1
      self.loopFilterB = np.array(loopFilter)
    elif np.ndim(loopFilter) == 2:
      self.loopFilterA = np.array(loopFilter[0])
      self.loopFilterB = np.array(loopFilter[1])
    else:
      print('Unexpected loop filter size, defaulting filter')
      self.loopFilterA = 1
      self.loopFilterB = np.ones(filterSize) / filterSize


  def run(self,loopInput):

    # Execute the costas loop
    # Normalize the input, may not be necessary
    loopInput = loopInput / np.abs(loopInput)
    
    # Is the input is not complex, create a complex representation
    if ~np.any(np.iscomplex(loopInput[0])):
      loopInput = hilbert(loopInput)
    
    # Grab the real and imaginary values in preperation for the error signal 
    # calculation
    realVal = np.real(loopInput[0])
    imagVal = np.imag(loopInput[0])
    
    # Loop over the input
    for ndx in range(len(loopInput)):
      
      # Calculate the latest error signal and update the error buffer
      self.errSig = np.concatenate((np.array([realVal * imagVal]), self.errSig[0:-1]))
      
      if self.loopFilterA == 1:
        # If the denominator is 1, this is an FIR filter, don't need to keep a 
        # memory of previous filter outputs
        filtErr = np.dot(self.loopFilterB,np.flip(self.errSig,0))
      else:
        # Else this is an IIR filter, must evaluate the entire filter sturcture
        newVal = np.dot(self.loopFilterB,np.flip(self.errSig,0)) - \
          np.dot(self.loopFilterA[1:],self.filtErr[0:-1])
        newVal = newVal / self.loopFilterA[0]
        # Update the filter outputs
        self.filtErr = np.roll(self.filtErr,1)
        self.filtErr[0] = newVal
        # Save off the latest output for code convenience later
        filtErr = self.filtErr[0]

      if ndx == 0:
        # If this is the first time through, use the last output of the 
        # previous iteration to maintain coherency across processing blocks
        self.phsVal[ndx] = self.phsVal[-1] + filtErr
      else:
        # This isn't the first time through, use the previous output
        self.phsVal[ndx] = self.phsVal[ndx-1] + filtErr
        
      # Calculate the output
      self.output[ndx] = loopInput[ndx] * np.exp(1j*self.phsVal[ndx])
      
      # Feed back the latest output as input to the next iteration
      realVal = np.real(self.output[ndx])
      imagVal = np.imag(self.output[ndx])
      
    self.output = self.output * np.exp(1j*np.pi/2);
#    bits = np.append(bits,np.real(rawBits) > 0);