#!/usr/bin/env python
import matplotlib
#matplotlib.use('Agg')
from netCDF4 import Dataset
from pylab import *
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

style.use('dark_background')

# initial profile
case = 'case-10/k2-18b.out2.00188'
data = Dataset(f'{case}.nc', 'r')

pres_axis = mean(data['press'][0,:,:,0], axis=1) / 1.e5
print(min(pres_axis), max(pres_axis))

# mixing ratio map
H2O = data['H2O'][0,:,:,0] * 1.e3

# vertical velocity map
vel1 = data['vel1'][0,:,:,0]

# cloud map
H2Oc = data['H2O(l)'][0,:,:,0] * 1.e3

# precipitation map
H2Op = data['H2O(l,p)'][0,:,:,0] * 1.e3

x1v = data['x1'][:] / 1.e3
x2v = data['x2'][:] / 1.e3

fig, axs = subplots(1, 2, 
                    figsize = (8, 6),
                    sharey = True,
                    gridspec_kw = {'width_ratios': [8,1]})
subplots_adjust(hspace = 0.10, wspace = 0.10)

dx = (x2v[-1] - x2v[0]) / (len(x2v) - 1)
dlnp = (log(pres_axis[0]) - log(pres_axis[-1])) / (len(pres_axis) - 1)

# mixing ratio map
X, Y = meshgrid(x2v, pres_axis)

# cloud map
print('H2Oc min/max:', np.nanmin(H2Oc), np.nanmax(H2Oc))
#clevels = logspace(0, 3, 7)

# precipitation map
#iH2Op = H2Op > 1.E-5
#xH2Op = (X.T)[iH2Op]
#yH2Op = (Y.T)[iH2Op]

# randomize the position of precipitation in the cell
#xH2Op += dx*(rand(len(xH2Op)) - 0.5)
#yH2Op *= exp(dlnp*(rand(len(yH2Op)) - 0.5))

# water mixing ratio
wlevel = arange(0., 52.50, 2.5)
ax = axs[0]
h2 = ax.contourf(X, Y, vel1,
                 np.linspace(-50., 50., 11),
                 cmap = 'inferno', extend = 'both')
print('H2Oc min/max:', np.nanmin(H2Oc), np.nanmax(H2Oc))
ax.contourf(X, Y, H2Oc, [10., 20., 40., 80., 160.],
            cmap = 'Greys_r',
            norm = LogNorm(), 
            alpha = 0.8)
#ax.scatter(xH2Op, yH2Op, marker = 'o', color = 'b', s = H2Op[iH2Op]*10.)

#ax.set_xlim([min(x2v), max(x2v)])
ax.set_ylim([max(pres_axis), 0.02])
ax.set_ylabel('Pressure (bar)', fontsize = 15)
ax.set_yscale('log')
ax.set_xlabel('Distance (km)', fontsize = 15)
#ax.text(20., 100, "Time = %.2f day" % time[j], fontsize = 15, color = 'k')

# mixing ratio average
ax = axs[1]
#ax.plot(mean(H2O, axis=0), pres_axis, '-', color = 'gold')
ax.plot(mean(H2Oc, axis=1), pres_axis, '--', color = 'gold')
#ax.set_xscale('log')
#ax.set_xlim([0., 50.])
ax.set_xlabel('g/kg')

# add colorbar
divider = make_axes_locatable(axs[0])
cax = divider.append_axes("right", size = "5%", pad = 0., aspect = 40)
colorbar(h2, cax = cax)

#show()
savefig('fig_k2-18b_case10.png', bbox_inches = 'tight')
close(fig)
