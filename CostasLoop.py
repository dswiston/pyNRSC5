#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 22:25:09 2018

@author: dave
"""
import numpy as np
from scipy.signal import hilbert


class CostasLoop:

  phsVal = []       # Phase modulation vector applied to each input that 
                    # basebands (freq locks) and phase locks to the input
                    # Size n
  output = []       # Basebanded output signal
                    # Size n
  freqInc = []      # Frequency estimate of the pll
                    # Size 1
  bufferSize = []   # Size of input vector to be basebanded
                    # Size 1
  loopFiltFreq = [] # The loop filter frequency coefficient
                    # Size 1
  loopFiltPhs = []  # The loop filter phase coefficient
                    # Size 1


  def __init__(self,loopFiltFreq,loopFiltPhs,bufferSize):
    self.freqInc = 0
    self.bufferSize = bufferSize
    self.loopFiltFreq = loopFiltFreq
    self.loopFiltPhs = loopFiltPhs
    self.output = np.zeros(bufferSize,dtype=np.complex64)
    self.phsVal = np.zeros(bufferSize,dtype=np.complex64)


  def run(self,loopInput):
    phsInc = 0
    #import pdb; pdb.set_trace()
    # Execute the costas loop
    # Normalize the input.  Allows the loop filter gain to not depend on the
    # input signal amplitude
    loopInput = loopInput / np.max(np.abs(loopInput))
    
    # Is the input is not complex (analytic), create a complex (analytic) 
    # representation
    if ~np.any(np.iscomplex(loopInput[0])):
      loopInput = hilbert(loopInput)
    
    # Loop over the input
    for ndx in range(len(loopInput)):
      
      # Create the basebanded signal
      modSig = loopInput[ndx] * np.exp(1j*(self.phsVal[ndx]))
      
      # Store in an output array
      self.output[ndx] = modSig
      
      # Calculate the individual I and Q values, to be used to compute the 
      # error signal
      realVal = np.real(modSig)
      imagVal = np.imag(modSig)
      
      # Calculate the latest error signal.  Note, I don't believe I need to 
      # worry about filtering the I/Q to remove the 2Wc component because the 
      # NCO is being  applied as an analytic signal
      errSig = realVal * imagVal
      # Apply the loop filter and calculate the most recent error to be fed 
      # into the NCO.  Track frequency and phase necessary to lock.
      self.freqInc += self.loopFiltFreq * errSig
      phsInc = self.loopFiltPhs * errSig + self.freqInc
      
      # Update the phase of the next NCO output sample
      if ndx == self.bufferSize-1:
        # This is the last time through.  Act as a circular buffer and wrap to
        # the first value so that the next time through the first sample will 
        # be modulated by the last loop output of the previous buffer
        self.phsVal[0] = self.phsVal[ndx] + phsInc
      else:
        # This isn't the last block iteration, output goes into next index
        self.phsVal[ndx+1] = self.phsVal[ndx] + phsInc
      
    self.output = self.output * np.exp(1j*np.pi/2);
#    bits = np.append(bits,np.real(rawBits) > 0);