import numpy as np
import pandas as pd
import gsw
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from scipy.integrate import quad as scipy_integrate_quad
from more_itertools import ilen, always_iterable

from depth_pressure_conversions import pressure_from_depth, read_ctd_data, mean_profile

def sound_speed_in_cavity(files ='data/auxiliary/CTD/CTD_NBP2202_*.txt',  cavity_criterion = 'Dotson', make_plot = True,
                          SA = 'AbsSal (g/kg)', CT =  'ConsTemp (deg)', p = 'Pressure (dB)', 
                          lat = 'Latitude (deg)', lon = 'Longitude (deg)'):
    """
    Returns a logarithmic fit to the mean profile of sound speed from CTD data

    Parameters:
        cavity_criterion : If given, only uses selected data to compute fit.
                           Options: 
                               'Dotson' : Data from Dotson ice shelf cavity (based on lon/lat)
                               'p200'   : Only use sound speed from depth deeper than 200 dBar
                               None     : Use all available data
    """
    def sound_speed_model(log_p, a, b):
        return a*log_p + b
    
    # Load CTD data
    sound_speed_all = sound_speed_from_ctd(filename = files, SA=SA, CT=CT, p=p, lat=lat, lon=lon)

    if cavity_criterion == 'Dotson':
        inside_cavity = is_inside_Dotson_cavity(sound_speed_all)
        sound_speed_used = sound_speed_all.where(inside_cavity)
    elif cavity_criterion == 'p200':
        inside_cavity = sound_speed_all.p>200
        sound_speed_used = sound_speed_all.where(inside_cavity)
    else:
        sound_speed_used = sound_speed_all

    # Compute mean profile
    bin_size = 10
    bins = np.arange(np.min(sound_speed_used.p),np.max(sound_speed_used.p),bin_size)
    (c, p) = mean_profile(sound_speed_used, var = 'c', p = 'p', bins=bins)
    sound_speed_profile  = pd.DataFrame({'p' : p,
                                         'c' : c})
    
    # Fit to model
    popt, pcov = curve_fit(sound_speed_model, 
                           np.log(sound_speed_profile.p), 
                           sound_speed_profile.c)
    
    def sound_speed_profile_model(p):
        c_min = np.min(sound_speed_profile.c)
        if ilen(always_iterable(p)) == 1: # p single value
            if p > 0:
                c = sound_speed_model(np.log(p), popt[0], popt[1])
                c = np.max([c_min, c])
            else:
                c = c_min
        else: # p array           
            c = np.zeros(p.shape)+c_min
            c[p>0] = sound_speed_model(np.log(p[p>0]), popt[0], popt[1])
            c[c<c_min] = c_min 
        return c
    
    # Make figure
    if make_plot:
        # Plot all measurements
        plt.scatter(sound_speed_all.c,
                    sound_speed_all.p,
                    s=1, c='silver', label = 'all measurements')
        # Plot used measurements
        plt.scatter(sound_speed_used.c,
                    sound_speed_used.p,
                    s=1, c='tab:blue', label = 'used measurements')
        
        # Plot mean profile
        ylim = plt.ylim()
        xlim = plt.xlim()
        ps = np.linspace(ylim[0], ylim[1], 100)
        plt.plot(sound_speed_profile.c, sound_speed_profile.p, 'k', label = 'mean profile')
        
        # Plot sound speed fit
        plt.plot(sound_speed_profile_model(ps), ps, 'r--', label = 'fitted model')
        
        plt.ylim(ylim[::-1])
        plt.xlim(xlim)
        plt.xlabel('speed of sound (m/s)')
        plt.ylabel('pressure (dbar)')
        plt.title('Sound speed profile')
        plt.legend()
        plt.grid(alpha=0.2)       

    return sound_speed_profile_model   

def is_inside_Dotson_cavity(ds):
    lon1 = -113.4
    lon2 = -112.75
    lat1 = -74.18
    lat2 = -74.26
    crit1 = (ds.lon > -113.4) & (ds.lon < -111.8) # correct ice shelf
    crit2 = ds.lat < [max(lat2, x) for x in lat1 + (lat2-lat1)/(lon2-lon1)*(ds.lon.values -lon1)] #south of limit
    return crit1 & crit2 

def sound_speed_from_ctd(filename = 'data/auxiliary/CTD/CTD_NBP2202_*.txt', 
                         SA = 'AbsSal (g/kg)', CT =  'ConsTemp (deg)', p = 'Pressure (dB)',
                         lat = 'Latitude (deg)', lon = 'Longitude (deg)'):
    ctd_data = read_ctd_data(filename)
    return pd.DataFrame({'c' : gsw.sound_speed(ctd_data[SA], 
                                               ctd_data[CT], 
                                               ctd_data[p]),
                         'p' : ctd_data[p],
                         'lon' : ctd_data[lon],
                         'lat' : ctd_data[lat],
                        }).dropna(axis=0)

def sound_speed_harmonic_mean(depth1, depth2, profile, N=10):  
    """
    Computes the harmonic mean of the sound speed along a depth profile, given a sound speed profile 
    as a function of pressure. 

    Parameters:
        depth1, depth2  : Depths to compute average between (unit: m)
        profile : Sound speed profile (a function that takes pressure (dbar) as input)
        N       : Number of points used to compute average
    """
    p1 = pressure_from_depth(depth1)
    p2 = pressure_from_depth(depth2)
    ps = np.linspace(p1, p2, N) 

    cs = profile(ps)
    
    return N/np.sum(1/cs)

def refine_vertical_distance(t0, sensor_depth, h0, theta0, sound_speed_model_wrt_p, N=1, make_plots = False, fast_integration = False):
    """
    Improve estimated vertical distance to detected interface using ray theory and depth-dependent sound speed.
    Based on eq 8 in Hovem 2013.
    
    Parameters:
        t0 : the measured time between sensor and detected interface
        sensor_depth : depth of the ADCP (m)
        theta0 : vertical angle of beam at sensor depth (90 = along z-axis)
        sound_speed_model_wrt_p :  function c(p)
        N : number of iterations
        make_plots : If true, plot estimated h for each iteration step
        fast_integration :  If true, use a quicker and less accurate integration
    """
    def integrand(z):
        """
        Eq 8 in Hovem 2013
        """
        p = pressure_from_depth(-z)
        c = sound_speed_model_wrt_p(p)        
        return 1/(c*np.sqrt(1-(xi*c)**2))

    z0 = -sensor_depth
    c0 = sound_speed_model_wrt_p(pressure_from_depth(-z0))
    xi = np.cos(np.radians(theta0))/c0
    c_av = sound_speed_harmonic_mean(z0, z0+h0, sound_speed_model_wrt_p)

    if make_plots:
        hs = np.zeros(N+1)
        hs[0] = h0
    
    h = h0
    for i in range(N):
        if fast_integration:
            tau = _quick_and_dirty_integration(integrand, z0, z0 + h, N = 20)
        else:
            tau = scipy_integrate_quad(integrand, z0, z0 + h)[0] 
        h = h + (t0 - tau)*c_av

        if make_plots:
            hs[i+1]   = h

    if make_plots:
        plt.plot(hs, '*-')
        plt.ylabel('vertical distance to interface (m)')
        plt.xlabel('iteration')
        plt.title('Iterative refinement using sound speed profile')
        plt.grid(alpha=0.2)
    return h

def _quick_and_dirty_integration(integrand, lower_limit, upper_limit, N = 20):
    """
    Fast, but not so accurate, integration
    """
    xs = np.linspace(lower_limit, upper_limit, N) 
    dx = (upper_limit - lower_limit)/N 
    I = 0
    for x in xs:
        I = I + integrand(x)*dx
    return I
