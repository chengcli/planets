'''
Read radiation tendency from Leconte et al. (2024)

256_marker_q0.45_F350_lowevap_startexo_tracermode.nc
'''

from netCDF4 import Dataset
from matplotlib import pyplot as plt
import numpy as np

def read_leconte_2024():
    fname = '256_marker_q0.45_F350_lowevap_startexo_tracermode.nc'
    data = Dataset(fname, 'r')

    # radiative cooling
    dt_rad = data['DT_RAD'][:]
    print('dt_rad = ', dt_rad.shape)

    # mean radiative cooling
    dt_rad_mean = dt_rad.mean(axis=(0, 2, 3))
    print('dt_rad_mean = ', dt_rad_mean.shape)

    # pressure levels (stagged)
    plev = data['P_HYD_W'][:]
    print('plev = ', plev.shape)

    # mean pressure levels
    plev_mean = plev.mean(axis=(0, 2, 3))
    plyr_mean = np.sqrt(plev_mean[:-1] * plev_mean[1:])
    print('plev_mean = ', plev_mean.shape)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))

    ax.plot(dt_rad_mean, plyr_mean / 1e5, label='Leconte et al. (2024)')
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_xlabel('Radiative Tendency (K/s)')
    ax.set_ylabel('Pressure (bar)')

    plt.show()

if __name__ == "__main__":
    read_leconte_2024()
