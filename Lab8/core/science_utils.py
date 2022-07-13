import numpy as np
from core import parameters as param
from scipy.special import erfcinv as erfcinv


# function that returns the SNR
def to_snr(signal_power, noise_power):
    return 10 * np.log10(signal_power / noise_power)


# function that returns the distance between two points
def line_len(node_pos, second_node_pos):
    return np.sqrt((node_pos[0] - second_node_pos[0]) ** 2 + (node_pos[1] - second_node_pos[1]) ** 2)


# function to calculate the bit rate for fixed rate strategy
def bit_rate_fixed(gsnr):
    if gsnr >= 2 * ((erfcinv(2 * param.BERt)) ** 2) * param.Rs / param.Bn:
        return 100e9
    else:
        return 0


# function to calculate the bit rate for flex rate strategy
def bit_rate_flex(gsnr):
    test1 = 2 * ((erfcinv(2 * param.BERt)) ** 2) * param.Rs / param.Bn
    test2 = 14 / 3 * ((erfcinv(3 / 2 * param.BERt)) ** 2) * param.Rs / param.Bn
    test3 = 10 * ((erfcinv(8 / 3 * param.BERt)) ** 2) * param.Rs / param.Bn
    if gsnr < 2 * ((erfcinv(2 * param.BERt)) ** 2) * param.Rs / param.Bn:
        return 0
    elif gsnr < 14 / 3 * ((erfcinv(3 / 2 * param.BERt)) ** 2) * param.Rs / param.Bn:
        return 100e9
    elif gsnr < 10 * ((erfcinv(8 / 3 * param.BERt)) ** 2) * param.Rs / param.Bn:
        return 200e9
    else:
        return 400e9


# function to calculate the bit rate for shannon strategy
def bit_rate_shannon(gsnr):
    # theoretical Shannon rate with an ideal Gaussian modulation
    bit_rate = 2 * param.Rs * np.log2(1 + (gsnr * param.Bn / param.Rs))
    return bit_rate