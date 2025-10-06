#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

sat_moist = np.genfromtxt("saturn_profile_moist.txt")
sat_pseudo = np.genfromtxt("saturn_profile_pseudo.txt")
sat_dry = np.genfromtxt("saturn_profile_dry.txt")
sat_neutral = np.genfromtxt("saturn_profile_neutral.txt")

fig = plt.figure(figsize=(6,8))
ax = fig.add_subplot(1,1,1)

# Plot moist adiabat
pres = sat_moist[:,2]
temp = sat_moist[:,3]
ax.plot(temp, pres, label="Moist Adiabat", color='blue')

# Plot pseudo adiabat
pres = sat_pseudo[:,2]
temp = sat_pseudo[:,3]
ax.plot(temp, pres, label="Pseudo Adiabat", color='orange')

# Plot dry adiabat
pres = sat_dry[:,2]
temp = sat_dry[:,3]
ax.plot(temp, pres, label="Dry Adiabat", color='green')

# Plot neutral density
pres = sat_neutral[:,2]
temp = sat_neutral[:,3]
ax.plot(temp, pres, label="Neutral", color='red')

ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlabel("Temperature (K)")
ax.set_ylabel("Pressure (bar)")
ax.set_ylim(1000, 0.1)

plt.show()
