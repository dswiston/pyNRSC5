#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:39:04 2018

@author: dave
"""

import numpy as np


def DBPSKDemod(input):

  bits = np.real(input[1:,:] * np.conj(input[0:-1,:]))
  bits[bits < 0] = -1
  bits[bits > 0] = 1
  return bits