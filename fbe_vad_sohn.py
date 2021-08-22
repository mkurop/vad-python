
# -*- coding: utf-8 -*_
"""
fbe.py
=========================================================================
FBE module provides several utilities and signal parametrization methods. 
"""


__author__ = 'Marcin Kuropatwiński'

import spectrum

import numpy as np

from typing import Tuple

from scipy.io import wavfile

import scipy.signal

import os

import soundfile as sf

class sin2cos2:

    """
    Class for computing signal windowing function with sin(x)^2 and cos(x)^2 tails.

    :param frame: the frame length in samples
    :type frame: int
    :param overlap: the size of the overlaping part of the window (the length of the tails on both sides)
    :type overlap: int

    :return: nothing

    """

    def __init__(self, frame : int = 512, overlap : int = 50):

        self._win = np.zeros((frame,))

        self._frame = frame

        self._overlap = overlap

        self._compute_window()

    def _compute_window(self):

        for i in range(self._overlap):

            self._win[i] = np.sin(2*np.pi/(4*(self._overlap+2))*(i+1))**2

        for i in range(self._overlap,self._frame-self._overlap):

            self._win[i] = 1

        for i in range(self._frame-self._overlap,self._frame):

            self._win[i] = np.cos(2*np.pi/(4*(self._overlap+2))*(i-self._frame+self._overlap+1))**2

    def window(self):

        """
        Method returning the vector of window's values.

        :return: the window

        :rtype: numpy array of length frame

        """

        return self._win

class fbe:

    """
    Versatile class computing various speech signal representations, mostly based on AR modelling and Mel Frequency
    Filterbanks. 

    :param frame_zero_adding: required length of the sequence after zero adding operation, defaults to None, which indicates no zero adding
    :type frame_zero_adding: int

    :param frame: frame length in samples
    :type frame: int

    :param sr: sampling frequency in Hz
    :type sr: float

    :param preem_alfa: the preemphasis coefficient
    :type preem_alfa: float

    :param freq_range: frequency range in which the mel frequency filterbanks should be computed
    :type freq_range: np.ndarray two elemements vector of floats

    :param filts_num: number of mel frequency triangular filters in the filterbank
    :type filts_num: int

    :param window: the windowing function
    :type window: np.ndarray, numpy vector of floats, defaults to None, which causes using of rectangular window

    :param ar_order: the AR model order
    :type ar_order: int

    :param cepstral_lifter: the cepstral lifter in MFCC computation
    :type cepstral_lifter: int

    :param num_ceps: number of cepstra
    :type num_ceps: int

    :returns: nothing

    .. note:: PSD is abbreviation for power spectral density in the whole documentation. AR is abbreviation for
                autoregressive in the whole documentation.

    """

    def __init__(self, frame_zero_adding=None, frame=512, sr=16000, preem_alfa=0.95, overlap=0,
                 freq_range=[20., 8000.], filts_num=23, num_gfs=70, spl_of_max_amplitude=88,
                 window=None, ar_order=16, cepstral_lifter=22, num_ceps=13):

        if overlap==0 or overlap > frame/2:
            overlap = frame/2

        if window is None:
            window = np.ones((frame,))

        if frame != len(window):
            print("ERROR in fbe, frame and window lengths do not match, program exits ...")
            sys.exit(1)

        self.sr = sr  # sampling frequency in Hz

        self.frame = frame  # number of samples in the frame

        self.num_ceps = num_ceps

        if not frame_zero_adding is None:
            self._nfft = frame_zero_adding  # fft length, sets the self._nfft atribute
        else:
            self._nfft = frame

        self.preem_alfa = preem_alfa  # preemphasis coefficient

        self.freq_range = freq_range  # frequency range in Hz

        self.filts_num = filts_num  # number of triangular filterbank channels

        self.K = int(self._nfft / 2.) + 1  # length of the unique part of the FFT

        self.f_min = 0

        self.f_max = float(sr) / 2.

        self.f_low = self.freq_range[0]

        self.f_high = self.freq_range[1]

        # matrices

        self._tfb = self._tfb()  # compute the mel-frequency triangular filterbank, sets the H atribute

        self._pinv_tfb = self._pinv_tfb()

        self._wgh_mat = self._wgh_mat()

        self._inv_wgh_mat = self._inv_wgh_mat()

        # window

        self._window = window

        self._ar_order = ar_order

        # compute cepstral lifter

        L = cepstral_lifter
        N = num_ceps

        self.cepstral_lifter =  1+0.5*L*np.sin(np.pi*np.asarray(range(N))/float(L))

        # dct matrix

        self.dctmat = np.zeros((self.num_ceps,self.filts_num))

        for i in range(self.num_ceps):

            for j in range(self.filts_num):

                self.dctmat[i,j] = np.sqrt(2./self.filts_num) * np.cos(np.pi*i/self.filts_num*(j+.5))

        self.lst_elm = ['fr','frwin','fft','mag','ang','psd','senmatpsd','senpsd','lpc','var_lpc','armag','arpsd','fbe',\
                        'fbekaldi','arfbe','wgh','arwgh','sfbe','sarfbe','smag','spsd','sarmag','sarpsd','sfbewgh',\
                        'smagwgh','spsdwgh','senmatspsdwgh','senspsdwgh','sarfbewgh','sarpsdwgh','psdspl']

        self.results = {}

        self._reset()

    def _reset(self): 

        """ 
        Resets the cache. 

        """

        for e in self.lst_elm:

            self.results[e] = None


    def get_frame_len(self) -> int:

        """Returns the frame length in samples

        :return: the frame length

        :rtype: int

        """

        return self.frame

    def get_tfb(self) -> np.ndarray:

        """Gets the triangular mel frequency filterbank.

        :return: the filter matrix containing in each row a single filter

        :rtype: np.ndarray, numpy array with filts_num rows

        """

        return self._tfb

    def get_wgh(self) -> np.ndarray:

        """Gets the weighting matrix, which is a square of the product of pseudo inverses of the Jacobian of the linear
        magnitude spectrum filter banks transform.

        :return: the weighting matrix

        :rtype: numpy array with dimension filts_num x filts_num

        """
        return self._wgh_mat

    def get_inv_wgh(self) -> np.ndarray:

        """
        Gets pseudo inverse of the weighting matrix.

        :returns: the pseudo inverse of the weighting matrix

        :rtype: np.ndarray, numpy array with dimension filts_num x filts_num


        """

        return self._inv_wgh_mat

    def get_pinv_tfb(self) -> np.ndarray:

        """
        Gets the pseudoinverse of the filterbanks matrix.

        :returns: the pseudo inverse of the weighting matrix

        :rtype: np.ndarray, numpy array with dimension filts_num x filts_num


        """

        return self._pinv_tfb

    def window(self) -> np.ndarray:

        """
        Gets the signal windowing function.

        :returns: the windowing function

        :rtype: np.ndarray, numpy array with dimension 1 x frame

        """

        return self._window

    def _tfb(self):
        """
        Computes the mel frequency triangular filterbank.

        """

        # filter cutoff frequencies (Hz) for all filters, size 1x(M+2)

        aux = np.linspace(0, self.filts_num + 1, self.filts_num + 2)

        c = self._mel2hz(
            (self._hz2mel(self.f_low) + aux * (self._hz2mel(self.f_high) - self._hz2mel(self.f_low)) / float(self.filts_num + 1)))

        f = np.linspace(self.f_min, self.f_max, self.K)

        H = np.zeros((self.filts_num, self.K))

        for m in range(self.filts_num):

            a = list(f >= c[m])

            b = list(f <= c[m + 1])

            k = np.array([a[i] and b[i] for i in range(len(f))])

            H[m, k] = (f[k] - c[m]) / (c[m + 1] - c[m])

            a = list(f >= c[m + 1])

            b = list(f <= c[m + 2])

            k = np.array([a[i] and b[i] for i in range(len(f))])

            H[m, k] = (c[m + 2] - f[k]) / (c[m + 2] - c[m + 1])

        return H

    def _proj_symmat_pd(self,A : np.ndarray,rcond : float) -> np.ndarray:
        """Projecting matrix A onto space of positive definite matrices.

        :param A: matrix to be projected
        :type A: np.ndarray
        :param rcond: reciprocal condition number of the resulting matrix
        :type rcond: float

        :return: the projected matrix
        :rtype: np.ndarray
        """

        A = .5*(A.T+A)

        w, v = np.linalg.eigh(A)

        w[w<rcond*np.max(w)] = rcond*np.max(w)

        f = (np.sqrt(w) * v).T

        f = f.T.dot(f)

        return f

    def _wgh_mat(self):

        """
        The weighting matrix.

        """

        W = np.dot(self._pinv_tfb.T, self._pinv_tfb)

        W = self._proj_symmat_pd(W,10.e-3)

        f = np.linalg.cholesky(W).T

        return f

    def _inv_wgh_mat(self):

        """
        The inverse of the weighting matrix

        """

        return np.linalg.pinv(self._wgh_mat)

    def _pinv_tfb(self):

        """
        The pseudoinverse of the mel frequency triangular filter banks matrix

        """

        return np.linalg.pinv(self._tfb)

    def _nextpow2(self, i):

        """
        The next nearest power of 2.

        """

        n = 1

        while n < i: n *= 2

        self._nfft = n

    def _hz2mel(self, hz):

        """
        Hertz to mel frequency scale

        """

        mel = 1127. * np.log(1 + hz / 700.)

        return mel

    def _mel2hz(self, mel):

        """
        Mel to frequency scale

        """

        hz = 700. * np.exp(mel / 1127.) - 700.

        return hz


    def idx2freq(self, i : int) -> float:

        """Converts frequency index to frequency in Hz

        :param i: frequency index
        :type i: int

        :return: frequency in Hz
        :rtype: float

        """

        f = float(i)/self._nfft*self.sr

        return f

    def freq2idx(self,f : float) -> int:

        """Converts frequency in Hz to the frequency index

        :param f: frequency in Herz
        :type f: float

        :return: frequency index
        :rtype: int


        """

        idx = int(np.round(f*float(self._nfft)/self.sr))

        return idx


    def set_frm(self,fr : np.ndarray):

        """
        Sets the frame of the signal - this is then used to compute the all signal representations

        :param fr: signal frame
        :type fr: np.ndarray, numpy vector of floats

        :return: nothing

        """

        self._reset()

        self.results['fr'] = fr

    def set_wgh(self,wgh : np.ndarray):

        """
        Set compact spectrum

        :param wgh: the compact specturm with filt_num elements
        :type wgh: numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['wgh'] = wgh

    def set_arwgh(self,arwgh : np.ndarray):

        """
        Set AR compact spectrum

        :param arwgh: the compact autoregresive specturm with filt_num elements
        :type arwgh: np.ndarray, numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['arwgh'] = arwgh

    def set_fbe(self,fbe : np.ndarray):

        """
        Set filterbank energies

        :param fbe: the filter bank energies (vector with filt_num elements)
        :type fbe: np.ndarray, numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['fbe'] = fbe


    def set_mag(self,mag : np.ndarray):

        """Set magnitude spectrum

        :param mag: the magnitude spectrum
        :type mag: np.ndarray, numpy vector of floats

        :returns: nothing


        """

        self._reset()

        self.results['mag'] = mag

    def set_psd(self,psd : np.ndarray):

        """
        Set power density spectrum

        :param psd: the power density spectrum
        :type psd: np.ndarray, numpy vector of floats

        :returns: nothing

        """

        self._reset()

        self.results['psd'] = psd

    def fr(self) -> np.ndarray:

        """
        Gets frame

        :returns: the frame
        :rtype: np.ndarray, numpy vector of floats

        """

        if self.results['fr'] is None:

            print("Frame not given (emtpy vector), program exits ...")

            sys.exit(1)

        else:

            return self.results['fr']

    def fr_win(self) -> np.ndarray:

        """
        Gets windowed frame

        :returns: the windowed frame
        :rtype: np.ndarray, numpy vector of floats

        """

        if self.results['frwin'] is None:

            self.results['frwin'] = np.zeros((self._nfft,))

            self.results['frwin'][:self.frame] = self.fr() * self.window()

        else:

            pass

        return self.results['frwin']

    def fft(self) -> np.ndarray:

        """
        Gets FFT

        :returns: the fft of the, possibly zero added, signal frame
        :rtype: np.ndarray, numpy vector of complex floats

        """

        if self.results['fft'] is None:

            self.results['fft'] = np.fft.fft(self.fr_win())

        else:

            pass

        return self.results['fft']

    def mag(self) -> np.ndarray:

        """
        Gets magnitude spectrum

        :returns: the magnitude of the, possibly zero added, signal frame
        :rtype: np.ndarray, numpy vector of floats

        """

        if self.results['mag'] is None:

            self.results['mag'] = np.abs(self.fft())[ : self.K]

        else:

            pass

        return self.results['mag']

    def ang(self) -> np.ndarray:

        """
        Gets angular spectrum.

        :returns: the angular spectrum of the, possibly zero added, signal frame
        :rtype: np.ndarray, numpy vector of floats

        """

        if self.results['ang'] is None:

            self.results['ang'] = np.angle( self.fft() )

        else:

            pass

        return self.results['ang']

    def psd(self) -> np.ndarray:

        """
        Gets power density spectrum

        :returns: the PSD of the, possibly zero added, signal frame
        :rtype: np.ndarray, numpy vector of floats


        """

        if self.results['psd'] is None:

            self.results['psd'] = self.mag()**2.

        else:

            pass

        return self.results['psd']

    def lpc(self) -> np.ndarray:

        """
        Gets LPC coefficients aka STP coefficients or AR coefficients

        :return: LPC with the leading 1
        :rtype: np.ndarray

        """

        if self.results['lpc'] is None:

            _lpc, self.results['var_lpc'], k = spectrum.aryule(self.fr_win(), self._ar_order)

            self.results['lpc'] = np.concatenate((np.array([1]),_lpc))

        else:

            pass

        return self.results['lpc']

    def var_lpc(self) -> np.ndarray:

        """
        Gets variance of the short term residual spectrum

        :return: short term residual variance
        :rtype: np.ndarray

        """

        if self.results['var_lpc'] is None:

            self.results['lpc'], self.results['var_lpc'], k = spectrum.aryule(self.fr_win(), self._ar_order)

        else:

            pass

        return self.results['var_lpc']

    def set_ar(self,a,var):
        """Setting AR coefficients and STP residual variance

        :param a: AR coefficients with leading one
        :param var: variance of the short term residual
        """

        """Sets the AR coefficients"""

        self._reset()

        self.results['var_lpc'] = var

        self.results['lpc'] = a

    def armag(self) -> np.ndarray:

        """
        Gets AR magnitude spectrum

        :return: AR magnitude spectrum
        :rtype: np.ndarray, numpy vector of floats of length _nfft/2+1

        """

        if self.results['armag'] is None:

            p = len(self.lpc())-1

            aux = np.concatenate([self.lpc(),np.zeros((self._nfft-p-1,))],axis=0)

            fftaux = np.abs(np.fft.fft(aux))

            std = np.sqrt(self.var_lpc()*self._nfft)

            self.results['armag'] = np.real(std/fftaux[ : self.K])

        else:

            pass

        return self.results['armag']

    def arpsd(self) -> np.ndarray:

        """
        Gets AR power density spectrum

        :return: the AR PSD
        :rtype: np.ndarray

        """


        if self.results['arpsd'] is None:

            self.results['arpsd'] = self.armag() ** 2.

        else:

            pass

        return self.results['arpsd']

    def fbe(self) -> np.ndarray:

        """
        Gets filter banks outputs based on magnitude spectrum

        :return: filter bank filtered magnitude spectrum
        :rtype: np.ndarray, numpy vector of floats of length filt_num

        """

        if self.results['fbe'] is None:

            self.results['fbe'] = np.dot(self.get_tfb(),self.mag())

        else:

            pass

        return self.results['fbe']


    def sfbe2mfcc(self) -> np.ndarray:

        """
        Converts smoothed filter banks energies to MFCC coefficients

        :return: MFCC coefficients
        :rtype: np.ndarray, numpy vector of floats (size num_cep)

        """

        fbe = self.sfbe()

        logfbe = np.log(fbe)

        mfcc = np.dot(self.dctmat,logfbe) #scipy.fftpack.dct(logfbe,n=self.num_ceps,norm='ortho')

        # liftering

        cmfcc = self.cepstral_lifter*mfcc

        return cmfcc

    def fbe2mfcc(self) -> np.ndarray:

        """
        Converts filter banks energies to MFCC coefficients

        :return: MFCC coefficients
        :rtype: np.ndarray, numpy vector of floats (size num_cep)

        """

        fbe = self.fbe()

        logfbe = np.log(fbe)

        mfcc = np.dot(self.dctmat,logfbe) #scipy.fftpack.dct(logfbe,n=self.num_ceps,norm='ortho')

        # here comes liftering

        cmfcc = self.cepstral_lifter*mfcc

        return cmfcc


    def arfbe(self) -> np.ndarray:

        """
        AR magnitude spectrum to filter banks energies

        :return: filter bank filtered AR magnitude spectrum
        :rtype: np.ndarray, numpy vector of floats (size num_filt)

        """

        if self.results['arfbe'] is None:

            self.results['arfbe'] = np.dot(self.get_tfb(),self.armag())

        else:

            pass

        return self.results['arfbe']

    def wgh(self) -> np.ndarray:
        """
        Weighted filter bank energies

        :return: the magnitude compact spectrum
        :rtype: np.ndarray, numpy vector of floats (size num_filt)

        """

        if self.results['wgh'] is None:

            self.results['wgh'] = np.dot(self.get_wgh(),self.fbe())

        else:

            pass

        return self.results['wgh']

    def arwgh(self) -> np.ndarray:
        """
        AR weighted filter bank energies

        :return: the AR magnitude compact spectrum
        :rtype: np.ndarray, numpy vector of floats (size num_filt)

        """

        if self.results['arwgh'] is None:

            self.results['arwgh']  = np.real(np.dot(self.get_wgh(),self.arfbe()))

        else:

            pass

        return self.results['arwgh']

    def smag(self) -> np.ndarray:
        """
        Smoothed magnitude spectrum

        :return: magnitude spectrum computed from filter bank energies
        :rtype: np.ndarray, numpy vector of floats (size _nfft/2+1)

        """

        if self.results['smag'] is None:

            self.results['smag'] = np.dot(self.get_pinv_tfb(), self.fbe())

        else:

            pass

        return self.results['smag']

    def spsd(self) -> np.ndarray:

        """
        Smoothed power density spectrum

        :return: PSD computed from filter bank energies
        :rtype: np.ndarray, numpy vector of floats(size _nfft/2+1)

        """

        if self.results['spsd'] is None:

            self.results['spsd'] = self.smag()**2.

        else:

            pass

        return self.results['spsd']


    def sarmag(self)->np.ndarray:

        """
        Smoothed AR magnitude spectrum

        :return: smoothed (from arfbe) AR magnitude spectrum (size _nfft/2+1)
        :rtype: np.ndarray, numpy vector of floats

        """

        if self.results['sarmag'] is None:

            self.results['sarmag'] = np.dot(self.get_pinv_tfb(), self.arfbe())

        else:

            pass

        return self.results['sarmag']

    def sarpsd(self) -> np.ndarray:

        """
        Smoothed AR PSD

        :return: smoothed (from arfbe) AR PSD (size _nfft/2+1)
        :rtype: np.ndarray, numpy vector of floats

        """

        if self.results['sarpsd'] is None:

            self.results['sarpsd']  = self.sarmag() ** 2.

        else:

            pass

        return self.results['sarpsd']

    def preemphasis(self, signal : np.ndarray) -> np.ndarray:

        """Perform preemphasis on the input signal.

        :param signal: The signal to filter.
        :type signal: np.ndarray, numpy vector of floats
        :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
        :type coeff: float
        :returns: the filtered signal.
        :rtype: numpy vector of floats
        """

        return np.asarray(np.append(signal[0], signal[1:] - self.preem_alfa * signal[:-1]))

    def sfbe(self) -> np.ndarray:

        """
        Smoothed filter bank energies

        :return: smoothed, from compact spectrum, filter bank energies
        :rtype: np.ndarray, numpy vector of floats (size num_filt)

        """

        if self.results['sfbe'] is None:

            self.results['sfbe'] = np.dot(self.get_inv_wgh(), self.wgh())

        else:

            pass

        return self.results['sfbe']

    def sarfbe(self) -> np.ndarray:

        """
        Smoothed AR filter bank energies

        :return smoothed, from compact AR spectrum, filter bank energies
        :rtype: np.ndarray, numpy vector of floats (size num_filt)

        """

        if self.results['sarfbe'] is None:

            self.results['sarfbe'] = np.dot(self.get_inv_wgh(), self.arwgh())

        else:

            pass

        return self.results['sarfbe']

    def smagwgh(self) -> np.ndarray:

        """
        Smoothed magnitude spectrum

        :return: computed from compact spectum magnitude spectrum
        :rtype: np.ndarray, numpy vector of floats (size _nfft/2+1)

        """

        if self.results['smagwgh'] is None:

            self.results['smagwgh'] = np.dot(self.get_pinv_tfb(), self.sfbe())

        else:

            pass

        return self.results['smagwgh']

    def sarmagwgh(self) -> np.ndarray:

        """
        Smoothed AR magnitude spectrum

        :return: computed from AR compact spectrum magnitude spectrum
        :rtype: np.ndarray, numpy vector of floats (size _nfft/2+1)

        """

        if self.results['sarmagwgh'] is None:

            self.results['sarmagwgh'] = np.dot(self.get_pinv_tfb(), self.sarfbe())

        else:

            pass

        return self.results['sarmagwgh']

    def spsdwgh(self) -> np.ndarray:

        """
        Smoothed PSD

        :return: computed from compact spectrum PSD
        :rtype: np.ndarray, numpy vector of floats (size _nfft/2+1)

        """

        if self.results['spsdwgh'] is None:

            self.results['spsdwgh'] = self.smagwgh() ** 2.

        else:

            pass

        return self.results['spsdwgh']

    def sarpsdwgh(self) -> np.ndarray:

        """
        Smoothed AR PSD

        :return: PSD computed from AR compact spectra
        :rtype: np.ndarray, numpy vector of floats (size _nfft/2+1)

        """

        if self.results['sarpsdwgh'] is None:

            self.results['sarpsdwgh'] = self.sarmagwgh() ** 2.

        else:

            pass

        return self.results['sarpsdwgh']

    def psd2wgh(self,psd : np.ndarray) -> np.ndarray:

        """
        PSD -> weighted compact spectrum

        :param psd: the PSD
        :type psd: np.ndarray, numpy vector of floats (size _nfft/2+1)

        :return: compact spectrum based on PSD
        :rtype: np.ndarray, numpy vector of floats (size num_filt)

        """

        mag = np.sqrt(psd)

        fbe = np.dot(self.get_tfb(),mag)

        return np.dot(self.get_wgh(),fbe)

    def psd2ar(self,psd : np.ndarray,LpcOrder : int) -> Tuple[np.ndarray, float]:
        """

        Converting PSD into LPC coefficients and excitation variance

        :param psd: left half of the PSD
        :type psd: numpy vector of floats (size _nfft/2+1)
        :param LpcOrder: AR model order
        :type LpcOrder: int

        :return: * (`vector of floats`) direct form AR coeff. with leading 1
                 * (`float`) the variance of the short term residual
        """

        D = len(psd)

        B = np.concatenate([psd,psd[D-2:0:-1]])

        xc = np.real(np.fft.ifft(B))

        xc = xc[:LpcOrder+1]

        a, var, k = spectrum.LEVINSON(xc)

        a = np.concatenate([[1],a])

        var = var/(2*D-2)

        return a, var


def synth(mag_enh : np.ndarray, angular_r : np.ndarray, an_win : np.ndarray):

    """
    Signal synthesis based on magnitude and angular spectra

    :param mag_enh: enhanced speech magnitude spectrum
    :type psd_enh: np.ndarray, numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: np.ndarray, numpy vector of floats
    :param an_win: windowing function
    :type an_win: np.ndarray numpy vector of floats

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    # X = np.sqrt( psd_enh )

    X = mag_enh

    X[-1] = 0

    enhDft = np.concatenate( (X, X[-2:0:-1]) ) * np.exp( 1j * angular_r )

    an_win = np.sqrt( an_win )

    enh_fs = an_win*np.real( np.fft.ifft( enhDft ) )

    return enh_fs

def enhance_mag( mag_r, psd_n, psd_s):

    """
    The Wiener filter in frequency domain

    :param mag_r: noisy, unprocessed signal frame magnitude spectrum
    :type psd_r: numpy vector of floats
    :param psd_n: noise, estimated noise frame PSD
    :type psd_n: numpy vector of floats
    :param psd_s: speech, estimated speech frame PSD
    :type psd_s: numpy vector of floats

    :return: enhanced speech PSD
    :rtype: numpy vector of floats

    """

    psd_r_smag = psd_s + psd_n

    mag_enh = np.maximum(psd_s, 1.e-6) / np.maximum(psd_r_smag, 1.e-4) * mag_r

    mag_enh = np.maximum(mag_enh,0.001*mag_r)

    mag_enh[-1] = 0 # filter out the most high frequency peak

    return mag_enh

def enhance( mag_r, psd_n, psd_s, angular_r, an_win ):

    """
    The Wiener filter returning the time frequency signal frame

    :param mag_r: noisy, unprocessed signal frame magnitude spectrum
    :type mag_r: numpy vector of floats
    :param psd_n: noise, estimated noise frame PSD
    :type psd_n: numpy vector of floats
    :param psd_s: speech, estimated speech frame PSD
    :type psd_s: numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: numpy vector of floats
    :param an_win: windowing function
    :type an_win: numpy vector of floats

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    mag_enh = enhance_mag(mag_r, psd_n, psd_s)

    enh_fs = synth(mag_enh,angular_r,an_win)

    return enh_fs #, psd_enh

def enhance1 ( mag_r, psd_n, psd_s, angular_r, an_win):
    """
    Perceptual enhancement (SETAP p. 245)

    :param psd_r: noisy, unprocessed signal frame PSD
    :type psd_r: numpy vector of floats
    :param psd_n: noise, estimated noise frame PSD
    :type psd_n: numpy vector of floats
    :param psd_s: speech, estimated speech frame PSD
    :type psd_s: numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: numpy vector of floats
    :param an_win: windowing function
    :type an_win: numpy vector of floats

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    e = psd_s/np.maximum(psd_n,1.e-4)+1.e-6

    g = mag_r**2/np.maximum(psd_n,1.e-4)+1.e-6

    v = e*g/(1+e)

    aux = np.maximum(.5*v,1.e-2)

    if np.sum(np.isnan(aux)) > 0:

        print('nan found')

        raw_input()

    gain = np.sqrt(v)/(np.sqrt(np.pi)*g)*1/scipy.special.i0e(aux)

    if np.sum(np.isnan(gain)) > 0:

        print (v)

        print (np.sqrt(v))

        print (gain)

        print (scipy.special.iv(0,aux))

        print ('nan found')

        raw_input()

    # plt.plot(gain)
    #
    # plt.show()

    mag_enh = gain*mag_r

    enh_fs = synth(mag_enh,angular_r,an_win)

    return enh_fs #, psd_enh

def enhance2(psd_r, psd_n, mt, angular_r, an_win, ro=0.05): # note that instead of psd_s we pass the masking treshold (mt)

    """
    Perceptual Wiener filter

    :param psd_r: PSD of the noisy signal frame
    :type psd_r: numpy vector of floats
    :param psd_n: PSD of noise (may be approximate, smoothed etc.)
    :type psd_n: numpy vector of floats
    :param mt: PSD masking threshold
    :type mt: numpy vector of floats
    :param angular_r: angular noisy signal frame spectrum
    :type angular_r: numpy vector of floats
    :param an_win: windowing function
    :type an_win: numpy vector of floats
    :param ro: (0.-1.) larger causes less signal modification
    :type ro: float

    :return: time domain enhanced signal frame (windowed)
    :rtype: numpy vector of floats

    """

    gain = np.minimum(np.sqrt(mt/psd_n)+ro,1)

    psd_enh = gain*psd_r

    enh_fs = synth(psd_enh,angular_r,an_win)

    return enh_fs #, psd_enh

def load_wav(file_ : str, target_sampling_rate : int = 16000) -> Tuple[np.ndarray,int,int]:
    """Loads sound in a variety of formats supported by libsndfile. Resamples the input sound to target_sampling_rate.

    :param file_: sound file path
    :type file_: str
    :param target_sampling_rate: target sampling rate
    :type target_sampling_rate: int
    :returns: * the resampled sound to target sampling rate
              * the original sound sampling rate
              * the target sampling rate
    :rtype: Tuple[np.ndarray,int,int]
    """

    if not os.path.exists(file_):
        raise ValueError(f"The file {file_} does not exist.")

    s, sr = sf.read(file_, dtype=np.float32)

    print(f"Start resmpling")

    if int(sr) != int(target_sampling_rate):
    
        s_resampled = scipy.signal.resample(s,np.uint32(np.round(target_sampling_rate/sr*s.shape[0])))

    else:

        s_resampled = s

    print(f"Resampling done")

    np.random.seed(1000) # causes the dither is same on each run

    s_resampled += np.random.randn(*s_resampled.shape)*1.e-6 # add dither to improve numerical behaviour

    return s_resampled, int(sr), target_sampling_rate

def save_wav(signal : np.ndarray, sampling_rate : int, file_name : str):

    if not os.path.exists(os.path.dirname(file_name)):
        try:
            os.makedirs(os.path.dirname(file_name))
        except OSError:
            print(f'Can not create the directory {os.path.dirname(file_name)}')
        else:
            print(f'Path {os.path.dirname(file_name)} succesfully created.')

    sf.write(file_name, signal, sampling_rate)



if __name__ == "__main__":

    pass
