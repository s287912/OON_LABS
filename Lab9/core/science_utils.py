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
def bit_rate_fixed(gsnr, Rs = None):
    if Rs is None:
        Rs = param.Rs
    if gsnr >= 2 * ((erfcinv(2 * param.BERt)) ** 2) * Rs / param.Bn:
        return 100e9
    else:
        return 0


# function to calculate the bit rate for flex rate strategy
def bit_rate_flex(gsnr, Rs = None):
    if Rs is None:
        Rs = param.Rs
 #   test1 = 2 * ((erfcinv(2 * param.BERt)) ** 2) * param.Rs / param.Bn
  #  test2 = 14 / 3 * ((erfcinv(3 / 2 * param.BERt)) ** 2) * param.Rs / param.Bn
   # test3 = 10 * ((erfcinv(8 / 3 * param.BERt)) ** 2) * param.Rs / param.Bn
    if gsnr < 2 * ((erfcinv(2 * param.BERt)) ** 2) * Rs / param.Bn:
        return 0
    elif gsnr < 14 / 3 * ((erfcinv(3 / 2 * param.BERt)) ** 2) * Rs / param.Bn:
        return 100e9
    elif gsnr < 10 * ((erfcinv(8 / 3 * param.BERt)) ** 2) * Rs / param.Bn:
        return 200e9
    else:
        return 400e9


# function to calculate the bit rate for shannon strategy
def bit_rate_shannon(gsnr, Rs = None):
    if Rs is None:
        Rs = param.Rs
    # theoretical Shannon rate with an ideal Gaussian modulation
    bit_rate = 2 * Rs * np.log2(1 + (gsnr * param.Bn / Rs))
    return bit_rate


###########
import numpy as np
import scipy.special as sci_spe
from core import parameters as param


# function that returns the SNR
def to_snr(signal_power, noise_power):
    return 10 * np.log10(signal_power / noise_power)


# function that returns the distance between two points
def line_len(node_pos, second_node_pos):
    return np.sqrt((node_pos[0] - second_node_pos[0]) ** 2 + (node_pos[1] - second_node_pos[1]) ** 2)


# linear to db
def linear_to_db(val):
    return 10 * np.log10(val)


# db to linear
def db_to_linear(val):
    return 10 ** (val / 10)


# function to calculate the bit rate for fixed rate strategy
def bit_rate_fixed(gsnr, Rs=None):
    if not Rs:
        Rs = param.Rs
    if gsnr >= (2 * ((sci_spe.erfcinv(2 * param.BERt)) ** 2) * Rs / param.Bn):
        bit_rate = 1e11  # 100Gbps, PM-QPSK
    else:
        bit_rate = 0  # 0Gbps
    return bit_rate


# function to calculate the bit rate for flex rate strategy
def bit_rate_flex(gsnr, Rs=None):
    if not Rs:
        Rs = param.Rs
    if gsnr >= (10 * ((sci_spe.erfcinv((8 / 3) * param.BERt)) ** 2) * Rs / param.Bn):
        bit_rate = 4e11  # 400Gbps, PM-16QAM
    elif gsnr >= ((14 / 3) * ((sci_spe.erfcinv((3 / 2) * param.BERt)) ** 2) * Rs / param.Bn):
        bit_rate = 2e11  # 200Gbps, PM-8QAM
    elif gsnr >= (2 * ((sci_spe.erfcinv(2 * param.BERt)) ** 2) * Rs / param.Bn):
        bit_rate = 1e11  # 100Gbps, PM-QPSK
    else:
        bit_rate = 0  # 0Gbps
    return bit_rate


# function to calculate the bit rate for shannon strategy
def bit_rate_shannon(gsnr, Rs=None):
    if not Rs:
        Rs = param.Rs
    # theoretical Shannon rate with an ideal Gaussian modulation
    bit_rate = 2 * Rs * np.log2(1 + (gsnr * param.Bn / Rs))
    return bit_rate


# function to calculate the ASE
#      [adimensional]      [Hz]          [Hz]     [linear] [linear]
def ase(n_amplifiers, freq_band_center, noise_bw, noise_fig, gain):
    return n_amplifiers * param.h_plank * freq_band_center * noise_bw * noise_fig * (gain - 1)


# function to calculate the nonlinear interference.
def nli(Pch, eta_nli, N_span, Bn):
    return (Pch ** 3) * eta_nli * N_span * Bn


# function to calculate eta_nli for nonlinear interference
def nli_eta_nli(beta_2, Rs, Nch, delta_f, gamma, alpha, L_eff):
    return (16 / (27 * param.pi)) * np.log(((param.pi ** 2) / 2) * (beta_2 * (Rs ** 2) / alpha) *
                                           (Nch ** (2 * Rs / delta_f))) * (alpha / beta_2) * ((gamma ** 2) *
                                                                                              (L_eff ** 2) / (Rs ** 3))


# function to calculate the optimal launch power in input to a line
# def opt_launch_pwr(ampl_noise_fig, fiber_span_loss, freq_band_center, noise_bandwidth, eta_nli):
#    p_base = param.h_plank * freq_band_center * noise_bandwidth
#    return np.cbrt(ampl_noise_fig * fiber_span_loss * p_base / (2 * noise_bandwidth * eta_nli))
def opt_launch_pwr(P_ase, eta_nli, N_span, noise_bandwidth):
    return np.cbrt(P_ase / (2 * noise_bandwidth * eta_nli * N_span))