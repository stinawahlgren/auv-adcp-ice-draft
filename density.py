import pandas as pd
import numpy as np
import glob
import gsw
import matplotlib.pyplot as plt
from scipy.stats import linregress, binned_statistic


def z_from_p(p):
    """
    Converts from pressure (dbar) to z (m) using a linear model for in-situ density 
        rho(p) = rho0 + k*p
    based on all AUV hugin CTD measurements from the campaign.
    """
    # Linear density fit from in_situ_density_fit()
    rho0 = 1027.345
    k = 5.126e-3 
    rho_p = rho0 + k*p  
    
    g = 9.828
    k_scaled = k*1e-4 # convert pressure from dbar to Pa
    return -1/(k_scaled*g) * np.log(rho_p/rho0)

def p_from_z(z):
    """
    Converts from z (m) to pressure (dbar) using a linear model for in-situ density 
        rho(p) = rho0 + k*p
    based on all AUV hugin CTD measurements from the campaign.
    """    
    # Linear density fit from in_situ_density_fit()
    rho0 = 1027.345
    k = 5.126e-3 
    
    k_scaled = k*1e-4 # convert pressure from dbar to Pa
    g = 9.828
    return rho0/k_scaled * (np.exp(-k_scaled*g*z) - 1) * 1e-4

def density_from_ctd(filename = 'data/auxiliary/CTD/CTD_NBP2202_*.txt'):
    ctd_files = glob.glob(filename)
    ctd_data_list = [pd.read_csv(f) for f in ctd_files]
    ctd_data = pd.concat(ctd_data_list, axis=0, ignore_index = True)
    return pd.DataFrame({'rho' : gsw.density.rho(ctd_data['AbsSal (g/kg)'], 
                                                 ctd_data['ConsTemp (deg)'], 
                                                 ctd_data['Pressure (dB)']),
                         'p' : ctd_data['Pressure (dB)'],
                         'time' : pd.to_datetime(ctd_data['Time (Matlab format)']-719529,unit='D')
                        }).dropna(axis=0)

def in_situ_density_fit(files ='data/auxiliary/CTD/CTD_NBP2202_*.txt',  make_plot = True):
    """
    Fits a linear model rho = k*p + rho_0 to CTD measurements
    """
    # Compute in situ density
    rho = density_from_ctd(filename = files)

    # Compute mean profile
    bin_size = 10
    bins = np.arange(np.min(rho.p),np.max(rho.p),bin_size)
    res  = binned_statistic(rho.p.values,
                            rho.rho.values,
                            statistic='mean', 
                            bins= bins)
 
    # Make linear fit
    rho_fit = linregress((bins[:-1]+bins[1:])/2, 
                         res.statistic,
                        )

    if make_plot:
        rho.plot.scatter(x='rho', y='p', label='from CTD', s=1)
        xlim = plt.xlim()
        ylim = np.array(plt.ylim())
        plt.plot(rho_fit.slope*ylim + rho_fit.intercept, ylim, 'k--', label = 'linear fit')
        plt.xlim(xlim)
        plt.ylim(ylim[::-1])
        plt.grid(alpha=0.1)
        plt.legend()
        plt.xlabel('in situ density (kg/m^3)')
        plt.ylabel('pressure (dbar)')

    return rho_fit