#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 14:57:01 2018

@author: dave
"""

def TrimBuffer(sigIn,frameSize):
  # Calculate the index of the last (incomplete) symbol
  symbolBorder = sigIn.size % frameSize
  
  # First, save off the incomplete symbol at the end of the buffer
  leftOvrBuf = sigIn[-symbolBorder:-1]
  
  # Trim the incomplete symbol off of the buffer
  sigOut = sigIn[0:-symbolBorder]
  
  return (sigOut,leftOvrBuf)