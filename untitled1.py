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
import scipy as sp
import time
from PolyphaseDecimate import PolyphaseDecimate
from FreqEstKalman import FreqEstKalman
from CostasLoop import CostasLoop

plotData = 0
fs = 1488375
sampRateRatio = fs / 744187.5
prefixSize = int(112 * sampRateRatio)
msgSize = int(round(2048 * sampRateRatio))
frameSize = int(2160 * sampRateRatio)
refSubCar = np.concatenate((np.arange(-546,-355,19),np.arange(356,547,19))) + 2048
# Ref frame expected data to help sync to bitstream.  0 and 1s are expected values, -1s are unknown/do-not-cares
refBits = np.array([0, 1, 1, 0, 0, 1, 0, -1, -1, 1, 1, 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 1, 1, 1])
refBits = np.array([-1, 1, 1,-1,-1, 1,-1, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0, 0, 1, 1, 0])
                   #31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,09,08,07,06,05,04,03,02,01,00
refBits = np.flip(refBits,axis=0)
# Create the Kalman filter frequency tracker
A = np.eye(1)            # State difference equation
R = np.eye(1)*5**2       # Measurement covariance
H = np.eye(1)            # Measurement to state equation
P = 1500**2              # Initial state covariance
Q = np.eye(1)*0.1**2     # Process noise
# Create the filter object
kalman = FreqEstKalman(A,R,H,P,Q)
# Initialize the filter object
kalman.initialize(0)




blkSize = (frameSize * 400 + msgSize)


readSize = blkSize * 2
#readSize = blkSize * 4 * 2
filename = '/home/dave/Projects/PythonWork/NRSC5/nrsc5-sample.wav'
#filename = '/home/dave/Projects/PythonWork/NRSC5/gqrx_20180422_150802_101099000_1488375_fc.raw'
readFormat = str(readSize) + 'B'
#readFormat = str(int(readSize/4)) + 'f'    # DIV BY FOUR DUE TO GQRX FLOAT FORMAT

with open(filename,'rb') as fid:
  while True:
    tmp = fid.read(readSize)
    if not tmp: break

    tmp = struct.unpack(readFormat,tmp)
    # Convert to a numpy array of floats
    tmp = np.asarray(tmp,dtype=np.float32)
    # Subtract 127 from the data (to convert to signed)
    tmp = tmp - 127
    
    data = np.zeros(int(len(tmp)/2), dtype=np.complex64);
    data.real = tmp[::2];
    data.imag = tmp[1::2];
    
    if plotData:
      fftData = np.fft.fft(data)
      freq = np.fft.fftfreq(data.shape[-1])
      plt.plot(freq, 20*np.log10(np.abs(fftData)))
      plt.show()
      #time.sleep(0.1)
      #plt.gcf().clear()
    
    
    
    
    
    # Define the FIR filter taps used to remove the analog signal
    digiFilt = np.array([0.00005639838100182599,-0.00000438567889237624,-0.000014918315587035234,-0.00001831364241000401,-0.000008900086627316432,0.000003215361699106406,-0.0000026485108464951385,-0.00003876970674719022,-0.00009065918953048294,-0.00011762411129216404,-0.0000784559099163356,0.00003260313062517485,0.00016794108864133154,0.0002470818802555891,0.00020859944944434967,0.000058703047270330415,-0.00012277236439812013,-0.00023113326709946119,-0.00021041720541598865,-0.00009704090400461583,0.00000528091336725085,0.000007757767376210417,-0.00008445799723955361,-0.00016324354278411565,-0.00009738363511632341,0.00015098155065787074,0.0004649935377557557,0.0006286653202463515,0.0004755717123625656,0.000027439390844152245,-0.0004889351883709301,-0.0007774138831523228,-0.0006757572520753092,-0.00027744116176248396,0.0001299679787087992,0.00027846075017039516,0.00012986994515194435,-0.0000859760813603447,-0.000053514635490119006,0.00035123110561000926,0.0009065120299103647,0.0011596765018337069,0.0007572454204446319,-0.0002448728140863471,-0.0013418744606272276,-0.0018799022353390787,-0.0015057283571949128,-0.0004387256709077766,0.0006582761160559692,0.0011430878055734093,0.0008614655554018075,0.00025594847729658933,0.000006829663597120252,0.00045769884943017977,0.0012729468559736449,0.0016314174683085137,0.0008601899339280276,-0.0009522831117159397,-0.002861087582527683,-0.003633209594942286,-0.0026147935296924143,-0.0002660208973479808,0.002077316112254722,0.0030983235440291104,0.002420736970344698,0.0008610582050946798,-0.0002171874365431951,0.000007137473683039207,0.0011138363774014638,0.0017212342659091641,0.0006025873227119503,-0.0021848862381834734,-0.005073618896306015,-0.005953531441200416,-0.0037125885769001823,0.0007842215522739398,0.005087658686833498,0.006757127708532511,0.005049479420816566,0.0014433627803250122,-0.0014582327971298947,-0.001890304200155795,-0.00028116009001238836,0.0010933675642372713,-0.000009213995937522116,-0.003775195320874346,-0.007795707359442544,-0.008609851640816162,-0.004338311560812399,0.003544759379478087,0.010779909515157567,0.013044086042631634,0.008970017721033624,0.001223022575483549,-0.005302272677571572,-0.006979400178271112,-0.0040821930823673885,-0.0004859310279722767,-0.0004996845570223097,-0.005139294735089198,-0.010784801855384787,-0.011555853103326515,-0.00393308823814007,0.009780637167745293,0.022046047361067217,0.024795947494070235,0.015177172903433971,-0.001889179729842196,-0.016569058031234703,-0.020762092096596397,-0.013881441305938963,-0.003333399930858554,0.001127029153171003,-0.004623158488293891,-0.01477363019332902,-0.01687522308160704,-0.0013103437575226124,0.02943606037982136,0.058930249564358904,0.06520896843980004,0.03524524448960296,-0.023749084692177787,-0.08454721313031346,-0.11329506765857314,-0.08911637414600379,-0.01817258509395911,0.0671257949319007,0.12443459083204704,0.12443459083204704,0.0671257949319007,-0.01817258509395911,-0.08911637414600379,-0.11329506765857314,-0.08454721313031346,-0.023749084692177787,0.03524524448960296,0.06520896843980004,0.058930249564358904,0.02943606037982136,-0.0013103437575226124,-0.01687522308160704,-0.01477363019332902,-0.004623158488293891,0.001127029153171003,-0.003333399930858554,-0.013881441305938963,-0.020762092096596397,-0.016569058031234703,-0.001889179729842196,0.015177172903433971,0.024795947494070235,0.022046047361067217,0.009780637167745293,-0.00393308823814007,-0.011555853103326515,-0.010784801855384787,-0.005139294735089198,-0.0004996845570223097,-0.0004859310279722767,-0.0040821930823673885,-0.006979400178271112,-0.005302272677571572,0.001223022575483549,0.008970017721033624,0.013044086042631634,0.010779909515157567,0.003544759379478087,-0.004338311560812399,-0.008609851640816162,-0.007795707359442544,-0.003775195320874346,-0.000009213995937522116,0.0010933675642372713,-0.00028116009001238836,-0.001890304200155795,-0.0014582327971298947,0.0014433627803250122,0.005049479420816566,0.006757127708532511,0.005087658686833498,0.0007842215522739398,-0.0037125885769001823,-0.005953531441200416,-0.005073618896306015,-0.0021848862381834734,0.0006025873227119503,0.0017212342659091641,0.0011138363774014638,0.000007137473683039207,-0.0002171874365431951,0.0008610582050946798,0.002420736970344698,0.0030983235440291104,0.002077316112254722,-0.0002660208973479808,-0.0026147935296924143,-0.003633209594942286,-0.002861087582527683,-0.0009522831117159397,0.0008601899339280276,0.0016314174683085137,0.0012729468559736449,0.00045769884943017977,0.000006829663597120252,0.00025594847729658933,0.0008614655554018075,0.0011430878055734093,0.0006582761160559692,-0.0004387256709077766,-0.0015057283571949128,-0.0018799022353390787,-0.0013418744606272276,-0.0002448728140863471,0.0007572454204446319,0.0011596765018337069,0.0009065120299103647,0.00035123110561000926,-0.000053514635490119006,-0.0000859760813603447,0.00012986994515194435,0.00027846075017039516,0.0001299679787087992,-0.00027744116176248396,-0.0006757572520753092,-0.0007774138831523228,-0.0004889351883709301,0.000027439390844152245,0.0004755717123625656,0.0006286653202463515,0.0004649935377557557,0.00015098155065787074,-0.00009738363511632341,-0.00016324354278411565,-0.00008445799723955361,0.000007757767376210417,0.00000528091336725085,-0.00009704090400461583,-0.00021041720541598865,-0.00023113326709946119,-0.00012277236439812013,0.000058703047270330415,0.00020859944944434967,0.0002470818802555891,0.00016794108864133154,0.00003260313062517485,-0.0000784559099163356,-0.00011762411129216404,-0.00009065918953048294,-0.00003876970674719022,-0.0000026485108464951385,0.000003215361699106406,-0.000008900086627316432,-0.00001831364241000401,-0.000014918315587035234,-0.00000438567889237624,0.00005639838100182599])
    
    # Define the decimation rate to convert from the sampling rate to the digital rate
    digiDec = 1
    # Define the modulation frequency to approximately baseband the digital sidebands
    demodFreq = kalman.x
    # Create the demodulation signal to baseband digital sidebands
    sigTime = np.linspace(0,int(blkSize)-1,int(blkSize))   # <------------------------- reminder to add time reference in the future to preserve phase
    sigTime = sigTime / fs
    demodSig = np.exp(-1j*2*np.pi*demodFreq*sigTime)
    
    # Keep only 
    decDataUpper = PolyphaseDecimate(digiFilt,data,demodSig,[],digiDec)
    decDataLower = PolyphaseDecimate(digiFilt,data,np.conj(demodSig),[],digiDec)
    
    if plotData:
      
      fftData = np.fft.fft(decDataUpper)
      freq = np.fft.fftfreq(decDataUpper.shape[-1])
      plt.plot(freq, 20*np.log10(np.abs(fftData)))
      plt.show()
      
      fftData = np.fft.fft(decDataLower)
      freq = np.fft.fftfreq(decDataLower.shape[-1])
      plt.plot(freq, 20*np.log10(np.abs(fftData)))
      plt.show()
    
    # Execute the auto-correlation for syncronization. Only need to compute it for one lag (msgSize)
    corrSig = decDataUpper[0:-msgSize] * np.conj(decDataUpper[msgSize-1:-1])
    # Reshape so that multiple frames can be easily summed
    corrSig = np.reshape(corrSig,(-1,frameSize))
    # Sum across multiple frames to increase effective SNR
    corrSig = np.sum(corrSig,axis=0)
    # Create a filter for the cyclic prefix
    prefixFilt = np.blackman(224)
    # Filter the output. We know the length of the cyclic prefix. Specifically look for repeating 
    # correlation of that length, effectively performing a matched filter like operation.
    filtOut = lfilter(prefixFilt,1,corrSig)
    filtOut = np.fft.ifft( np.fft.fft(corrSig)*np.fft.fft(prefixFilt,n=len(corrSig)) )
    # Grab the highest value, indicating the offset
    timeOffset = np.argmax(np.abs(filtOut))
    # Calculate the frequency error
    freqErr = -np.angle(filtOut[timeOffset]) * fs / (2 * np.pi * msgSize)
    # Create the demodulation signal to baseband digital sidebands
    sigTime = np.linspace(0,len(decDataUpper)-1,len(decDataUpper))   # <------------------------- reminder to add time reference in the future to preserve phase
    sigTime = sigTime / fs
    demodSig = np.exp(-1j*2*np.pi*freqErr*sigTime)
    ofdmSig = decDataUpper * demodSig
    # Remove the cyclic prefix
    ofdmSig = ofdmSig[timeOffset:-1]
    ofdmSig = np.concatenate((np.zeros(prefixSize),ofdmSig))
    ofdmSig = ofdmSig[0:int(len(ofdmSig)/frameSize)*frameSize]
    ofdmSig = np.reshape(ofdmSig,(-1,frameSize))
    ofdmSig = ofdmSig[:,prefixSize-1:-1]
    #ofdmSig = np.reshape(ofdmSig,(-1))
    # Perform FFT
    fftOut = np.fft.fft(ofdmSig,msgSize)
    
    fftOut2 = np.fft.fftshift(fftOut,axes=1)
    
    freqEst = np.zeros((fftOut2.shape[0]-1,refSubCar.shape[0]))
    for i in range(0,refSubCar.shape[0]):
      fftNdx = refSubCar[i]
      # Attempt to fine-tune the frequency offset
      phsDiff = fftOut2[1:,fftNdx] * np.conj(fftOut2[0:-1,fftNdx])
      freqEst[:,i] = np.angle(phsDiff**2) / 2
      #phsEst[:,i] = 
    
    # Calculate frequency offset in Hz
    freqEst2 = np.mean(freqEst) * fs / msgSize / (2*np.pi)
    filtInput = demodFreq + freqErr + freqEst2
    kalman.run(filtInput)
    #print(kalman.x[0][0])
    print("\r KlmnEst: {0:.2f}, KlmnInput: {1:.2f}, FFT Err: {2:.2f}, RefCar Err: {3:.2f}".format(kalman.x,filtInput,freqErr,freqEst2), end=" ", flush=True)
    #radians/sample * samples/sec = radians/sec * 1cycle/(2*pi radians)
    
    
    # Create demod signal
    print(kalman.x-demodFreq)
    timeSamps = np.linspace(0,399,400)
    timeSamps = timeSamps / fs * 4096
    demodSig2 = np.exp(-1j*2*np.pi*(freqEst2)*timeSamps)
    demodSig2 = np.tile(demodSig2,(msgSize,1)).T   # <------- horribly inefficient, wasting time on non-existent subcarriers
    output = demodSig2 * fftOut2
    
    

    # Create the costas loop object
    costasLoop = CostasLoop(np.ones(5)/5,5,400)
    costasLoop = CostasLoop([],1,400)

    # Initialize the object
    costasLoop.run(fftOut2[:,refSubCar[0]])


    

    # DPSK demodulation
    bits = np.real(output[1:,refSubCar] * np.conj(output[0:-1,refSubCar]))
    bits[bits < 0] = -1
    bits[bits > 0] = 1
    tmp = lfilter(refBits*-1,1,bits,axis=0)
    
    
    
    

    plt.plot(np.angle(output[:,refSubCar[16]]),'.')
    plt.ylim([-3.15,3.15])
    plt.draw()
    plt.pause(0.5)
    time.sleep(0.5)
    plt.cla()
#    
#    plt.plot(abs(fftOut[1,:]))
#    plt.xlim(300,600)
#    
#    timeSamps = np.linspace(0,399,400)
#    timeSamps = timeSamps / fs * 4096
#    for i in refSubCar:
#      testSig = fftOut2[:,i]
#      for x in range(-470,-450):
#        testDemodSig = np.exp(-1j*2*np.pi*x/100*timeSamps)
#        output = testSig * testDemodSig
#        plt.plot(np.real(output),np.imag(output),'.')
#        plt.xlim(-600,600)
#        plt.ylim(-600,600)
#        plt.pause(0.1)
#        print(x)
#        plt.cla()
#    
#    if plotData:
#      plt.plot(filtOut)
