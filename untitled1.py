#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 01:35:14 2018

@author: dave
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from CostasLoop import CostasLoop
from DBPSKDemod import DBPSKDemod
import TicTocGenerator
from CoarseTimeSync import CoarseTimeSync
from RemoveCyclicPrefix import RemoveCyclicPrefix
from TrimBuffer import TrimBuffer
from FindBlockBoundary import FindBlockBoundary
from FindTimingOffset import FindTimingOffset

plotData = 0
fs = 1488375
sampRateRatio = fs / 744187.5
subcarrierFs = fs/2048/sampRateRatio
prefixSize = int(112 * sampRateRatio)
msgSize = int(round(2048 * sampRateRatio))
frameSize = int(2160 * sampRateRatio)
refSubCar = np.concatenate((np.arange(-546,-355,19),np.arange(356,547,19))) + 2048
p1SubCar = np.concatenate((np.arange(-546,-355),np.arange(356,547))) + 2048
p1SubCar = [e for e in p1SubCar if e not in (refSubCar)]
refSubCarDiff = refSubCar - refSubCar[0]
p1SubCarDiff = p1SubCar - refSubCar[0]
# Ref frame expected data to help sync to bitstream.  0 and 1s are expected values, -1s are unknown/do-not-cares
refBits = np.array([-1, 1, 1,-1,-1, 1,-1, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 1])
                   #31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,09,08,07,06,05,04,03,02,01,00
firstBlkRefBits = np.array([-1, 1, 1,-1,-1, 1,-1, 0, 0, 1, 1,-1,-1, 1,-1, 0,-1,-1,-1,-1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
                            #31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,09,08,07,06,05,04,03,02,01,00

refBits = np.flip(refBits,axis=0)

numBlks = 13
numBitsPerBlk = 32*numBlks+1
blkSize = (frameSize * numBitsPerBlk + msgSize)

# Define the FIR filter taps used to remove the analog signal
digiFilt = np.array([0.00005639838100182599,-0.00000438567889237624,-0.000014918315587035234,-0.00001831364241000401,-0.000008900086627316432,0.000003215361699106406,-0.0000026485108464951385,-0.00003876970674719022,-0.00009065918953048294,-0.00011762411129216404,-0.0000784559099163356,0.00003260313062517485,0.00016794108864133154,0.0002470818802555891,0.00020859944944434967,0.000058703047270330415,-0.00012277236439812013,-0.00023113326709946119,-0.00021041720541598865,-0.00009704090400461583,0.00000528091336725085,0.000007757767376210417,-0.00008445799723955361,-0.00016324354278411565,-0.00009738363511632341,0.00015098155065787074,0.0004649935377557557,0.0006286653202463515,0.0004755717123625656,0.000027439390844152245,-0.0004889351883709301,-0.0007774138831523228,-0.0006757572520753092,-0.00027744116176248396,0.0001299679787087992,0.00027846075017039516,0.00012986994515194435,-0.0000859760813603447,-0.000053514635490119006,0.00035123110561000926,0.0009065120299103647,0.0011596765018337069,0.0007572454204446319,-0.0002448728140863471,-0.0013418744606272276,-0.0018799022353390787,-0.0015057283571949128,-0.0004387256709077766,0.0006582761160559692,0.0011430878055734093,0.0008614655554018075,0.00025594847729658933,0.000006829663597120252,0.00045769884943017977,0.0012729468559736449,0.0016314174683085137,0.0008601899339280276,-0.0009522831117159397,-0.002861087582527683,-0.003633209594942286,-0.0026147935296924143,-0.0002660208973479808,0.002077316112254722,0.0030983235440291104,0.002420736970344698,0.0008610582050946798,-0.0002171874365431951,0.000007137473683039207,0.0011138363774014638,0.0017212342659091641,0.0006025873227119503,-0.0021848862381834734,-0.005073618896306015,-0.005953531441200416,-0.0037125885769001823,0.0007842215522739398,0.005087658686833498,0.006757127708532511,0.005049479420816566,0.0014433627803250122,-0.0014582327971298947,-0.001890304200155795,-0.00028116009001238836,0.0010933675642372713,-0.000009213995937522116,-0.003775195320874346,-0.007795707359442544,-0.008609851640816162,-0.004338311560812399,0.003544759379478087,0.010779909515157567,0.013044086042631634,0.008970017721033624,0.001223022575483549,-0.005302272677571572,-0.006979400178271112,-0.0040821930823673885,-0.0004859310279722767,-0.0004996845570223097,-0.005139294735089198,-0.010784801855384787,-0.011555853103326515,-0.00393308823814007,0.009780637167745293,0.022046047361067217,0.024795947494070235,0.015177172903433971,-0.001889179729842196,-0.016569058031234703,-0.020762092096596397,-0.013881441305938963,-0.003333399930858554,0.001127029153171003,-0.004623158488293891,-0.01477363019332902,-0.01687522308160704,-0.0013103437575226124,0.02943606037982136,0.058930249564358904,0.06520896843980004,0.03524524448960296,-0.023749084692177787,-0.08454721313031346,-0.11329506765857314,-0.08911637414600379,-0.01817258509395911,0.0671257949319007,0.12443459083204704,0.12443459083204704,0.0671257949319007,-0.01817258509395911,-0.08911637414600379,-0.11329506765857314,-0.08454721313031346,-0.023749084692177787,0.03524524448960296,0.06520896843980004,0.058930249564358904,0.02943606037982136,-0.0013103437575226124,-0.01687522308160704,-0.01477363019332902,-0.004623158488293891,0.001127029153171003,-0.003333399930858554,-0.013881441305938963,-0.020762092096596397,-0.016569058031234703,-0.001889179729842196,0.015177172903433971,0.024795947494070235,0.022046047361067217,0.009780637167745293,-0.00393308823814007,-0.011555853103326515,-0.010784801855384787,-0.005139294735089198,-0.0004996845570223097,-0.0004859310279722767,-0.0040821930823673885,-0.006979400178271112,-0.005302272677571572,0.001223022575483549,0.008970017721033624,0.013044086042631634,0.010779909515157567,0.003544759379478087,-0.004338311560812399,-0.008609851640816162,-0.007795707359442544,-0.003775195320874346,-0.000009213995937522116,0.0010933675642372713,-0.00028116009001238836,-0.001890304200155795,-0.0014582327971298947,0.0014433627803250122,0.005049479420816566,0.006757127708532511,0.005087658686833498,0.0007842215522739398,-0.0037125885769001823,-0.005953531441200416,-0.005073618896306015,-0.0021848862381834734,0.0006025873227119503,0.0017212342659091641,0.0011138363774014638,0.000007137473683039207,-0.0002171874365431951,0.0008610582050946798,0.002420736970344698,0.0030983235440291104,0.002077316112254722,-0.0002660208973479808,-0.0026147935296924143,-0.003633209594942286,-0.002861087582527683,-0.0009522831117159397,0.0008601899339280276,0.0016314174683085137,0.0012729468559736449,0.00045769884943017977,0.000006829663597120252,0.00025594847729658933,0.0008614655554018075,0.0011430878055734093,0.0006582761160559692,-0.0004387256709077766,-0.0015057283571949128,-0.0018799022353390787,-0.0013418744606272276,-0.0002448728140863471,0.0007572454204446319,0.0011596765018337069,0.0009065120299103647,0.00035123110561000926,-0.000053514635490119006,-0.0000859760813603447,0.00012986994515194435,0.00027846075017039516,0.0001299679787087992,-0.00027744116176248396,-0.0006757572520753092,-0.0007774138831523228,-0.0004889351883709301,0.000027439390844152245,0.0004755717123625656,0.0006286653202463515,0.0004649935377557557,0.00015098155065787074,-0.00009738363511632341,-0.00016324354278411565,-0.00008445799723955361,0.000007757767376210417,0.00000528091336725085,-0.00009704090400461583,-0.00021041720541598865,-0.00023113326709946119,-0.00012277236439812013,0.000058703047270330415,0.00020859944944434967,0.0002470818802555891,0.00016794108864133154,0.00003260313062517485,-0.0000784559099163356,-0.00011762411129216404,-0.00009065918953048294,-0.00003876970674719022,-0.0000026485108464951385,0.000003215361699106406,-0.000008900086627316432,-0.00001831364241000401,-0.000014918315587035234,-0.00000438567889237624,0.00005639838100182599])
digiFiltState = np.zeros(digiFilt.size-1)
# Define the decimation rate to convert from the sampling rate to the digital rate
digiDec = 1
# Define the modulation frequency to approximately baseband the digital sidebands
demodFreq = 0
demodPhs= 0
# Create the demodulation signal to baseband digital sidebands
sigTime = np.linspace(0,int(blkSize)-1,int(blkSize))   # <------------------------- reminder to add time reference in the future to preserve phase
sigTime = sigTime / fs
# Create a filter for the cyclic prefix
prefixFilt = np.blackman(224)

# Create the costas loop object
costasLoop = CostasLoop(0.25**2,0.25,numBitsPerBlk)

leftOvrBuf = np.array([],dtype=np.complex64)

isSync = False
intTimingOffset = 0
pltNdx = np.zeros(1)

readSize = blkSize * 2
#readSize = blkSize * 4 * 2
filename = '/home/dave/Projects/PythonWork/NRSC5/nrsc5-sample.wav'
#filename = '/home/dave/Projects/PythonWork/NRSC5/gqrx_20180422_150802_101099000_1488375_fc.raw'
readFormat = str(readSize) + 'B'
#readFormat = str(int(readSize/4)) + 'f'    # DIV BY FOUR DUE TO GQRX FLOAT FORMAT

#fid = open(filename,'rb')
#if True:
with open(filename,'rb') as fid:
  while True:
    tmp = fid.read(readSize)

    if not tmp: break

    tmp = struct.unpack(readFormat,tmp)
    # Convert to a numpy array of floats
    tmp = np.asarray(tmp,dtype=np.float32)
    # Subtract 127 from the data (to convert to signed)
    tmp = tmp - 127
    
    # Complex data is interleaved, create complex representation
    data = np.zeros(int(len(tmp)/2), dtype=np.complex64)
    data.real = tmp[::2]
    data.imag = tmp[1::2]
    
    # Perform filtering
    data,digiFiltState = lfilter(digiFilt, 1, data, zi=digiFiltState)
    
    if intTimingOffset > 0:
      leftOvrBuf = leftOvrBuf[intTimingOffset:]
    elif intTimingOffset < 0:
      leftOvrBuf = np.concatenate((np.zeros(-intTimingOffset),leftOvrBuf))
    
    # Append the unused data from the previous buffer
    data = np.concatenate((leftOvrBuf,data))
    
    if not(isSync):
      timeOffset, demodFreq, data = CoarseTimeSync(data,msgSize,frameSize,prefixFilt,fs)
      isSync = True
    
    # Trim the buffer to an integer number of frames based on the calculated
    # cyclic prefix location
    data, leftOvrBuf = TrimBuffer(data,frameSize)

    # Create the demodulation signal to baseband the signal
    sigTime = np.linspace(0,len(data)-1,len(data))
    sigTime = sigTime / fs
    demodSig = np.exp(-1j*(2*np.pi*demodFreq*sigTime+demodPhs))
    
    # Perform the basebanding
    ofdmSig = data * demodSig
    
    # Remove the cyclic prefix from the buffer and format the data by OFDM symbol
    ofdmSig = RemoveCyclicPrefix(ofdmSig,frameSize,prefixSize)
    
    # Remember the phase of the demodulator for the next iteration, maintains 
    # coherency buffer to buffer
    demodPhs = np.angle(demodSig[-1]) + 1/fs*2*np.pi*demodFreq

    # Perform FFT, including FFT shift
    fftOut = np.fft.fftshift(np.fft.fft(ofdmSig,msgSize),axes=1)
    
    # Only keep the necessary subcarriers
    refSubCars = fftOut[:,refSubCar]
    p1SubCars = fftOut[:,p1SubCar]
    
    # Run the costas loop
    costasLoop.run(refSubCars[:,0])
    freqErr = costasLoop.freqInc * subcarrierFs / (2*np.pi)
    
    # Update the receiver demod frequency
    #demodFreq -= freqErr
    
    # Apply the resulting NCO output to all of the subcarrier channels
    phsModMat = np.expand_dims(np.exp(1j*costasLoop.phsVal[0:-1]),1)
    output1 = refSubCars * np.tile(phsModMat,[1, refSubCars.shape[1]])
    output2 = p1SubCars * np.tile(phsModMat,[1, p1SubCars.shape[1]])
    
    # Find the timing offset of the buffer
    timingOffset = FindTimingOffset(output1,msgSize)
    intTimingOffset = int(np.round(timingOffset))
    fracTimingOffset = timingOffset - intTimingOffset
    
    # Account for fractional offsets
    timeOffPhsVal = timingOffset * (2*np.pi) / msgSize
    refPhsErr = np.concatenate((np.zeros(1),np.cumsum(np.diff(refSubCar)))) * timeOffPhsVal
    refPhsErr = np.exp(1j*refPhsErr)
    output1 = output1 * np.tile(refPhsErr,[refSubCars.shape[0],1])
    p1PhsErr = (np.concatenate((np.zeros(1),np.cumsum(np.diff(p1SubCar)))) + 1) * timeOffPhsVal
    p1PhsErr = np.exp(1j*p1PhsErr)
    output2 = output2 * np.tile(p1PhsErr,[p1SubCars.shape[0],1])

    # Perform DBPSK demodulation on the reference subcarriers
    refCarBits = DBPSKDemod(output1)
    
    # Find the block boundaries
    blkNdx, phsInv = FindBlockBoundary(refBits,refCarBits,numBitsPerBlk)
    
    # If locked 180deg out of phase, flip the bits
    if phsInv:
      refCarBits = refCarBits * -1
    
    plt.figure(1)
    plt.subplot(3,2,1); plt.cla()
    plt.plot(costasLoop.phsVal)
    plt.title('PLL Output')
    plt.subplot(3,2,2); plt.cla()
    for ndx in range(0,359):
      phsErr = 0.0
      plt.plot(np.real(output2[:,ndx]*np.exp(-1j*ndx*phsErr)),np.imag(output2[:,ndx]*np.exp(-1j*ndx*phsErr)),'o')
    plt.title('First Data Subcarriers')
    plt.subplot(3,2,3); plt.cla()
    for ndx in range(0,22):
      phsErr = 0.0
      plt.plot(np.real(output1[:,ndx]*np.exp(-1j*ndx*phsErr)),np.imag(output1[:,ndx]*np.exp(-1j*ndx*phsErr)),'o')
    plt.title('First Reference Subcarriers')
    plt.pause(0.5)
    
    '''
    plt.plot(np.real(output2[:,0]),np.imag(output2[:,0]),'.',markersize=12)
    #plt.plot(np.real(output2[:,1]),np.imag(output2[:,1]),'.',markersize=12)
    output3 = output2 * np.exp(1j*(-1*23/180*np.pi))
    plt.plot(np.real(output3[:,1]),np.imag(output3[:,1]),'.',markersize=12)

    plt.plot(np.real(output1[:,0]),np.imag(output1[:,0]),'.',markersize=12)
    #plt.plot(np.real(output1[:,2]),np.imag(output1[:,2]),'.',markersize=12)
    output3 = output1 * np.exp(1j*(-30/180*np.pi))
    plt.plot(np.real(output3[:,1]),np.imag(output3[:,1]),'.',markersize=12)
    
    
    plt.plot(np.real(refPhsMod[:,1]),np.imag(refPhsMod[:,1]),'.',markersize=12)

    0.06rad/sample
    0.06rad/sample * 363samples/sec / 2PIrad/cycle = cycles/sec
    2 * pi * k * timeErr / NFFT
    0.40142572795869574 = 2 * pi * 1 * timeErr / 2048
    1.04719 = 2 * pi * 19 * timeErr / 2048

'''
