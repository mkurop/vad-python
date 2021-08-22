"""VAD is the Voice Activity Detection module"""

__author__ = 'Marcin Kuropatwiński'


import copy

import numpy as np
from scipy.special import logsumexp
import fbe_vad_sohn
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath("__file__")),"./noise-tracking-hendriks/"))
import noise_tracking_hendriks

class vad:

    """ Implements the VAD algorithm from the paper:

        Jongseo Sohn, Nam Soo Kim, Wonyong Sung,
        A Statistical Model-Based Voice Activity Detection,
        IEEE Signal Processing Letters, vol. 6., no. 1, January 1999

        :param frame: frame length
        :param a01: log natural base probability of noise -> speech transition
        :param a10: log natural base probability of speech -> noise transition

        :return: nothing

        .. note::
            The speech -> noise transition probability should be extremly small to take effect

        .. todo::
            Change transition probabilities to log domain - done on 25.10.2016
    """


    def __init__(self, frame = 512, a01 = -.00001, a10 = -150):

        self.frame = frame

        self.mag_len = int(frame/2+1) # the length of the left half of the spectrum

        self.G_prev = 0 # previous speech detection statistics

        self.start = True # flag indicating beginning of the algorithm operation

        self.X_prev = np.zeros((self.mag_len,)) # initial previous frame speech power spectrum

        # algorithmic constants

        self.ksi_min=10 ** (-250./10)

        self.a01 = a01 # log a priori probability of no speech -> speech transition

        self.a10 = a10 # log a priori probability of speech -> no speech transition

        self.a00 = logsumexp([0,a01],b=[1,-1]) # probability of no speech -> no speech transition

        self.a11 = logsumexp([0,a10],b=[1,-1]) # probability of speech -> speech transition

        self.aa = 0.9 # running average smoothing factor

    def vad(self,X,lambdaN):

        """
        Method for speech presence probability

        :param X: the noisy speech PSD
        :param lambdaN: the estimated noise PSD

        :returns: **spprb** speech presence probability
        :rtype: float

        """

        lambdaN = np.maximum(lambdaN, 1.e-3)

        gammak = np.minimum(X / lambdaN, 1000)  # limit post SNR to avoid overflows


        if self.start:

            ksi = self.aa + (1 - self.aa) * np.maximum(gammak - 1, 0)

        else:

            ksi = self.aa * np.minimum(self.X_prev / lambdaN, 100) + (1 - self.aa) * np.maximum(gammak - 1, 0)

            ksi = np.maximum(self.ksi_min, ksi)

        print(f"sum snr {10*np.log10(np.sum(ksi)/self.frame)}")

        self.l = gammak * ksi / (1 + ksi) - np.log(1 + ksi)  # eq 3

        self.X_prev = X

        lambdaN[0] *= .5
        lambdaN[-1] *= .5
        gammak[0] *= .5
        gammak[-1] *= .5
        X[0] *= .5
        X[-1] *= .5

        l = copy.copy(self.l)

        l[0] *= .5
        l[-1] *= .5

        lpXH0 = 2.*np.sum(-np.log(np.pi*lambdaN) - gammak)

        lpXH1dpXH0 = 2.*np.sum(l)

        lpXH1 = lpXH1dpXH0+lpXH0

        if self.start:

            self.logsp = lpXH1
            self.lognp = lpXH0

            self.start = False

        else:

            self.logsp = logsumexp([self.logsp+self.a11,self.lognp+self.a01])+lpXH1

            self.lognp = logsumexp([self.logsp+self.a10,self.lognp+self.a00])+lpXH0

        spprb = np.exp(self.logsp-logsumexp([self.logsp,self.lognp]))


        return spprb

FRAME = 512

if __name__ == "__main__":

    speech, sampling_rate = fbe_vad_sohn.load_wav("./test-data/SI2290.wav")

    nst = noise_tracking_hendriks.NoiseTracking(frame = FRAME)

    v = vad(frame = FRAME)

    speech_psd = fbe_vad_sohn.fbe()

    start = 0

    while start + FRAME <= speech.shape[0]:

        fs = speech[start:start+FRAME]

        speech_psd.set_frm(fs)

        psd = speech_psd.psd()
        
        nst.noisePowRunning(psd)

        speech_presence_probability = v.vad(psd,nst.get_noise_psd())

        print(f"Speech presence probability: {speech_presence_probability}, frame mean power: {np.sum(fs*fs)/FRAME}")

        start += FRAME//2




