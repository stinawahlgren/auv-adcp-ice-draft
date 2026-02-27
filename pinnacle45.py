import numpy as np
from xarray import DataArray

def beam_direction_instrument__Pin45():
    """
    Beam directions in instrument coordinate system for the upward-looking RDI Pinnacle 45 ADCP. 
    """
    beam_angle_instrument_vertical = 20 * np.pi/180 # in radians
    CV = np.cos(beam_angle_instrument_vertical)
    SV = np.sin(beam_angle_instrument_vertical)
    
    beam_direction__inst = DataArray(np.array([[SV, -SV, 0, 0],
                                               [0, 0, SV, -SV],
                                               [CV, CV, CV, CV]]),
                                     dims = ['inst', 'beam'],
                                     coords = {'inst': ['x', 'y', 'z'],
                                               'beam': [1,2,3,4]}
                                    )
    return beam_direction__inst


def roation_matrix_instrument2earth__Pin45(ds, pitch='pitch_adcp', roll='roll_adcp', heading='heading_adcp', time_dim='time'):
    """
    Computes a rotation matrix for each ping for the upward-looking RDI Pinnacle 45 ADCP.  
    Transforms from instrument-referenced coordinate system to earth-referenced coordinate system.
    
    Based on equation 18 in ADCP Coordinate Transformation, Pinnacle document, P/N 951-6079-00 
    (January 2010). 
    
    Parameters:
        ds: xarray Dataset with ADCP orientation (pitch, roll, heading) for each ping
        pitch/roll/heading: field in ds with instrument tilt/roll/heading
        time_dim: name of the time dimension in ds
    """
        
    P = ds[pitch] * np.pi/180
    R = ds[roll] * np.pi/180
    H = ds[heading] * np.pi/180

    CH = np.cos(H)
    SH = np.sin(H)
    CP = np.cos(P)
    SP = np.sin(P)
    CR = np.cos(R)
    SR = np.sin(R)

    M11 = CH*CR + SH*SP*SR
    M12 = SH*CP
    M13 = CH*SR - SH*SP*CR
    M21 = -SH*CR + CH*SP*SR
    M22 = CH*CP
    M23 = -SH*SR - CH*SP*CR
    M31 = -CP*SR
    M32 = SP
    M33 = CP*CR

    M = np.array( [[M11, M12, M13],
                   [M21, M22, M23],
                   [M31, M32, M33]] )

    M_da = DataArray(M,
                     dims = ('earth', 'inst', 'time'),
                     coords = {'earth': ['E', 'N', 'U'],
                               'inst': ['x','y','z'],
                               'time': ds.time}) 
    return M_da