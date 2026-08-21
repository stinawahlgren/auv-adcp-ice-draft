"""
Functions for detecting lag between signals
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from xarray import DataArray
from pandas import date_range

def find_time_lag(t1, f1, t2, f2, time_step = 0.25, trim = 3600, interpolation_method = 'linear', verbose = False, f1_label = 'f1', f2_label = 'f2'):
    """
    Find the time lag between the two signals f1(t1) and f2(t2). The sign of the lag is
    so that removing the lag from t1 will result in synchronized signals.

    The lag is found by first interpolating the two signals to a common time grid, and then taking 
    argmax of the cross-correlation between interpolated f1 and f2. 
    parameters:
        time_step : time step in seconds of the common time grid
        trim      : trim seconds are removed from the start and end of the signals before finding the lag.
    """

    # Make time grid
    tmin = np.max([t1[0], t2[0]])   + np.timedelta64(trim,'s')
    tmax = np.min([t1[-1], t2[-1]]) - np.timedelta64(trim,'s')
    time = date_range(start=tmin, end=tmax, freq=f'{time_step}s')
    if len(time) < 3:
        raise ValueError("Signals don't overlap, is trim too large?")

    # xarray deals well with time interpolation, so we make a DataArray first and interpolate that
    s1 = DataArray(data = f1, dims = 'time', coords = {'time' : t1})
    s2 = DataArray(data = f2, dims = 'time', coords = {'time' : t2})

    # If there are several time steps with the same time, it will cause errors when interpolate. Remove them!
    s1 = remove_time_step_duplicates(s1)
    s2 = remove_time_step_duplicates(s2)

    # Interpolate to common time grid
    s1_interp = s1.interp(time=time, method=interpolation_method)
    s2_interp = s2.interp(time=time, method=interpolation_method)

    if verbose:
        # Original signals
        plt.figure(figsize=(8,2))
        s1.plot(label = f1_label)
        s2.plot(label = f2_label)
        plt.title('Original signals')
        plt.legend()
        plt.grid()
        plt.show()

        # Trimmed and interpolated signals
        _plot_signals(s1_interp,s2_interp, 'Preprocessed signals', s1_label=f1_label, s2_label = f2_label)
        
    # Find lag (in samples)
    lag__samples = detect_lag(s1_interp.values, s2_interp.values, verbose = verbose)

    # Convert lag to time unit
    lag = np.timedelta64(int(time_step*1e9), 'ns') * lag__samples
    print(f'Detected time lag : {lag}')
    
    return lag

def remove_time_step_duplicates(da):
    """
    Remove time step duplicates from a DataArray (da) with a time dimension called 'time'
    """
    has_unique_time = np.concatenate([[True], da.time.diff(dim='time').values > np.timedelta64(1,'ns')])
    return da.isel(time=has_unique_time)

def detect_lag(f1, f2, mode='same', verbose=False):

    x = signal.correlate(f1-np.mean(f1),f2-np.mean(f2), mode=mode)
    lags = signal.correlation_lags(f1.size,f2.size, mode=mode)
    
    detected_lag = lags[np.argmax(x)]

    if verbose: 
        plt.figure(figsize=(8,2))
        plt.plot(lags,x)
        plt.title('cross-correlation')
        plt.xlabel('lag (samples)')
        plt.grid()
        plt.show()
        print(f'Detected lag : {detected_lag} samples')
        
    return detected_lag


def _plot_signals(s1,s2, title, s1_label = 'f1', s2_label = 'f2'):
    N1 = len(s1.time.values)
    N2 = len(s2.time.values)

    zoom = (min(N1,N2) > 8000)

    if zoom:
        plt.figure(figsize=(8,3))
        plt.subplot(1,2,1)
        s1.plot(label = s1_label)
        s2.plot(label = s2_label)
        plt.grid()
        plt.title(title)
        plt.legend()

        plt.subplot(1,2,2)
        index = slice(int(min(N1,N2)/4), int(min(N1,N2)/4) + 4000)
        s1.isel(time=index).plot(label = s1_label)
        s2.isel(time=index).plot(label = s2_label)
        plt.grid()
        plt.title('zoomed')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        s1.plot(label = s1_label)
        s2.plot(label = s2_label)
        plt.grid()
        plt.title(title)
        plt.legend()   
        plt.show()
    return