#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:00:41 2018

@author: dave
"""

import numpy as np


def RemoveCyclicPrefix(sigIn,frameSize,prefixSize):
  
  # Reshape into a properly framed matrix of OFDM symbols
  sigOut = np.reshape(sigIn,(-1,frameSize))
  
  # Trim off the cyclic prefix from each symbol
  sigOut = sigOut[:,prefixSize-1:-1]

  return sigOut
