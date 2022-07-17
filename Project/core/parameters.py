NUMBER_OF_CHANNELS= 10
from scipy.constants import c
channels = 10

import scipy.constants as const
import numpy as np

c = const.speed_of_light
NUMBER_OF_CHANNELS = 10
side_channel_occupancy = True
BERt = 1e-3  # 1e-3
Rs = 32e9
Bn = 12.5e9

G = 10 ** (16/10) # linear
NF = 10 ** (3/10) # linear
h = const.h
f = 193.414 * 1e12

alfa_dB = 0.2 # dB
alfa_linear = (alfa_dB/1e3) / ( 20 * np.log10(np.exp(1)))
beta2 = 2.13e-26
gamma = 1.27e-3

df = 50e9
span_length = 80e3

Rb_min = 100e9



