import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sc_fft
import pandas as pd


class Time_Signal:
    # group all the elaboration tools in one class. 
    # this class is built to handle the data processing of a single measurement. 

    def __init__(self, signal, observation_time):
        self.T = observation_time
        self.signal = np.array(signal)
        self.N = self.signal.shape[0]

        ##
        self.Ts = self.T / self.N
        self.t = np.linspace(0, self.T, self.N)

        

        
    def to_frequency_domain(self, avoid_DC=False):
        fft_spectrum = self.fft()
        return Frequency_Signal(fft_spectrum, f_Nyquist=0.5/self.T) 

    def fft(self, signal=None, normalize=True, sides=1):
        if signal is None:
            signal = self.signal
        
        if normalize:
            norm="forward"
        else:
            norm="backward"

        fft1 = 2*sc_fft.rfft(signal, norm=norm)
        fft1[0] = fft1[0]/2
        fft2 = sc_fft.fftshift(sc_fft.fft(signal, norm=norm))


        if sides == 1:
            return fft1
        if sides == 2:
            return fft2
        if sides == 0:
            return fft1, fft2

    def plot_time_signal_magnitude(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(16,4), constrained_layout = True)
        ax.plot(self.t, np.abs(self.signal))
        ax.set_title("Absolute")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Signal")
        ax.grid(True)

    def plot_time_signal(self, ax= None):
        if ax is None:
            fig, ax = plt.subplots(2,1, figsize=(16,6), constrained_layout = True)

        ax[0].plot(self.t, np.real(self.signal))
        ax[0].set_title("REAL COMPONENT")

        ax[1].plot(self.t, np.imag(self.signal))
        ax[1].set_title("IMAGINARY COMPONENT")
        

        for axx in ax:
            axx.set_xlabel("Time [s]")
            axx.set_ylabel("Signal")
            axx.grid(True)


class Frequency_Signal_Base():
    """For the base frequency signal I pass both values and frequencies in two different arrays with the same length"""
    def __init__(self, signal, f):
        self.signal, self.f = self._order_freqs(signal, f)
        self.N = len(self.signal)
        self.f_max = self.f.max()
    
    @staticmethod
    def _order_freqs(signal, f):
        values = np.rec.fromarrays([signal, f], dtype=[("signal", np.complex128), ("frequencies", np.float64)])
        sorted_values = np.sort(values, order="frequencies")
        return sorted_values["signal"], sorted_values["frequencies"]
    
    def plot_mag_and_phase(self, ax = None, xscale="log", exclude_DC = False):
        if ax is None:
            fig, ax = plt.subplots(2,1,figsize=(12,6), constrained_layout = True)

        f = self.f
        signal = self.signal

        if exclude_DC:
            f = self.f[1:]
            signal = self.signal[1:]
        
        ax[0].stem(f, np.abs(signal))
        ax[0].set_xlabel("Frequencies [Hz]")
        ax[0].set_ylabel("Magnitude")
        ax[0].set_xscale(xscale)
        ax[0].grid(True)

        ax[0].set_title("MAGNITUDE")

        ax[1].plot(f, np.angle(signal, deg=True))
        ax[1].set_xlabel("Frequencies [Hz]")
        ax[1].set_ylabel("Phase [deg]")
        ax[1].set_xscale(xscale)
        ax[1].set_ylim([-180,180])
        ax[1].set_yticks(np.arange(-180, 180, 45))
        ax[1].grid(True)
        ax[1].set_title("PHASE")
    

    def transform_to_signal(self, mult_factor=1, f_Nyquist=None, f_min = np.finfo(np.float64).eps):      
        f = self.f
        values = self.signal

        if f_Nyquist is not None:
            f = np.concatenate([f, [f_Nyquist]])
            values = np.concatenate([values, [0]])
        
        lcm_frequencies = np.lcm.reduce(f.astype(np.int64)).astype(np.float64)
        factors = np.array([lcm_frequencies/kk for kk in f])

        N_of_intervals = np.lcm.reduce(factors.astype(np.int64))*mult_factor

        # where are the original frequencies in the new array
        my_indices = np.array([N_of_intervals/kk for kk in factors]).astype(int)

        # create the new frequency array that is uniformly spaced
        my_f = np.linspace(0, f.max(), my_indices[-1]+1) #number of points is the number of intervals plus 1
        
        # fill any missing points with zero value
        my_values = np.zeros(my_f.shape, dtype=np.complex128)
        
        # add the original values to the known freqs
        my_values[my_indices] = values

        return Frequency_Signal(my_values, my_f[-1], f_min = f_min)






class Frequency_Signal(Frequency_Signal_Base):
    """
    This class is used to handle frequency domain signals with 1 sided spectrum and normalized
    """

    def __init__(self, signal, f_Nyquist, f_min = 0):
        self.f_Nyquist = f_Nyquist
        f = np.linspace(f_min, self.f_Nyquist, int(len(signal)))

        super().__init__(signal, f)
        # two sided
        self.f2 = np.linspace(-self.f_Nyquist, self.f_Nyquist, int(2*self.N-1))


        self.fs = 2*self.f_Nyquist
        self.f_res = self.f_Nyquist/self.N



    def ifft(self, normalized_input=True):
        if normalized_input:
            norm="forward"
        else:
            norm="backward"

        my_fft = self.signal.copy()
        my_fft *= 0.5
        my_fft[0] = 2*my_fft[0]
        return np.real(sc_fft.irfft(my_fft, norm="forward"))


    def to_time_domain(self):
        return Time_Signal(observation_time = 1/self.fs, signal = self.ifft())

    
        
        












class Frequency_Signal_PostProcessing(Frequency_Signal):
    def __init__(self, signal, f_Nyquist):
        super().init(signal, f_Nyquist)
    
    def get_fft_table(self):     
        return pd.DataFrame(np.stack([self.f, self.signal], axis=1), columns=["f", "fft"]).astype({"f":"float32", "fft":"complex128"}).set_index("f")


    def apply_conv_FIR_filter(self, signal, filt):
        self.FIR_spectr = self.fft(filt, sides=2)

        #         self.FIR_coherent_gain = np.average(self.FIR_spectr)
        #         self.FIR_coherent_power_gain = np.sqrt(np.average(self.FIR_spectr**2))
        self.filt_signal = np.convolve(filt, signal, "same")

        #         self.filt_signal_amp = self.filt_signal/self.FIR_coherent_gain
        #         self.filt_signal_pwr = self.filt_signal/self.FIR_coherent_power_gain

        return self.filt_signal

    def apply_window(self, signal, window, coherence="amplitude"):
        self.coherent_gain = np.average(window)
        self.coherent_power_gain = np.sqrt(np.average(window ** 2))

        raw_windowed_signal = signal * window

        if coherence == "amplitude":
            windowed_signal = raw_windowed_signal / self.coherent_gain
        elif coherence == "power":
            windowed_signal = raw_windowed_signal / self.coherent_power_gain
        else:
            print("retuning raw window")
            windowed_signal = raw_windowed_signal

        return windowed_signal

    def inspect_filter(self, signal, FIR, window=None, window_coherence="amplitude", ax=None):
        if ax == None:
            fig, ax = plt.subplots(1, 2, figsize=(18, 6), dpi=120)

        ntaps = FIR.shape[0]
        f_FIR = np.linspace(0, self.f_Nyquist, int((ntaps + 1) // 2))
        FIR_fz = self.fft(FIR, sides=2, normalize=False)[-f_FIR.shape[0]:]

        filt = self.apply_conv_FIR_filter(self.signal, FIR)

        ax[0].plot(self.f1, self.fft(signal), label="raw signal")
        ax[0].plot(self.f1, self.fft(filt), label="filtered")
        ax[0].plot(f_FIR, FIR_fz, '-k', label="filter frequency response")

        ax[0].set_xscale("log")
        ax[0].set_xlim([10e6, 2.5e9])
        ax[0].grid(True, which="both")
        ax[0].set_xlabel("f[Hz]")
        ax[0].set_ylabel("Spectrum Magnitude [V]")
        ax[0].set_ylim([0, 1.5])

        ax[1].plot(self.f1, 20 * np.log10(self.fft(signal)), label="raw signal")
        ax[1].plot(self.f1, 20 * np.log10(self.fft(filt)), label="filtered")
        ax[1].plot(f_FIR, 20 * np.log10(FIR_fz), '-k', label="filter frequency response")

        ax[1].set_ylim([-40, 5])
        ax[1].set_ylabel("Spectrum Magnitude[dB]")
        ax[1].set_xscale("log")
        ax[1].set_xlim([10e6, 2.5e9])
        ax[1].grid(True, which="both")
        ax[1].set_xlabel("f[Hz]")

        if window is not None:
            windowed = self.fft(self.apply_window(filt, window, coherence=window_coherence))

            ax[0].plot(self.f1, windowed, label="windowed and filtered")
            ax[1].plot(self.f1, 20 * np.log10(windowed), label="windowed and filtered")

        ax[0].legend()
        ax[1].legend()