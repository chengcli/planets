#!/usr/bin/env python3
"""
Skeleton plotting script for a simulation NetCDF.

Columns:
1) narrow  : line plot (X=cooling rate, Y=height)
2) wide    : 2D contour of water cloud (X=horizontal, Y=height)
3) narrow  : line plot of horizontal-mean water (X=mean water, Y=height)
"""

import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# -----------------------
# User-editable defaults
# -----------------------
NC_PATH = "case-7/k2-18b.out2.00174.nc"

# Variable names in your file — adjust to match your dataset
VAR_HEIGHT = "z"               # vertical coordinate (e.g., "z", "height", "level")
VAR_HORIZ  = "x"               # horizontal coordinate (e.g., "x", "lon", "i")
VAR_TIME   = "time"            # time coordinate (optional)
VAR_COOL   = "cooling_rate"    # cooling rate [e.g., K/s] shaped like (z) or (time, z)
VAR_WATER  = "qc"              # water cloud mixing ratio (e.g., kg/kg), shaped (z, x) or (time, z, x)

# Choose which time index to plot if time exists
TIME_INDEX = 0

# -----------------------
# Load data
# -----------------------
ds = xr.open_dataset(NC_PATH)

# coordinates
x1v = ds["x1"]
x2v = ds["x2"]

# variables
h2o = ds["H2O"].isel({"time": 0}).load()
h2o_cloud = ds["H2O(l)"].isel({"time": 0}).load()

print("x1v:", x1v)
print("x1v dims:", x1v.dims)
print("h2o shape:", h2o.shape)
print("h2o cloud shape:", h2o_cloud.shape)

# Compute horizontal mean profile of water (mean over X)
h2o_cloud_mean = h2o_cloud.mean(dim="x2", skipna=True)
print("h2o_cloud_mean shape:", h2o_cloud_mean.shape)

# -----------------------
# Figure layout
# -----------------------
fig = plt.figure(figsize=(10, 4.2), dpi=150)
gs = GridSpec(
    nrows=1, ncols=3, figure=fig,
    width_ratios=[1.1, 3.2, 1.1], wspace=0.25
)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

# -----------------------
# Plot 1: Cooling rate vs height
# -----------------------
#ax1.plot(cool, z)
#ax1.set_xlabel(f"{VAR_COOL} [{cool.attrs.get('units','')}]".strip())
#ax1.set_ylabel(f"{VAR_HEIGHT} [{z.attrs.get('units','')}]".strip())
#ax1.grid(True, alpha=0.3)

# Optional: draw a reference zero line if cooling can be ±
#if np.nanmin(cool.values) < 0 < np.nanmax(cool.values):
#    ax1.axvline(0, lw=0.8, alpha=0.6)

# -----------------------
# Plot 2: 2D contour of water (X–Z)
# -----------------------
# pcolormesh is a good default; use contourf if you prefer smoothed contours
h = ax2.contourf(x2v, x1v, h2o_cloud[:,:,0], levels=20, cmap="Blues")
cbar = fig.colorbar(h, ax=ax2, pad=0.02)
#cbar.set_label(f"{VAR_WATER} [{water.attrs.get('units','')}]".strip())

ax2.set_xlabel(f"Distance (km)")
#ax2.set_title("Water cloud field")

# -----------------------
# Plot 3: Horizontal mean water cloud vs height
# -----------------------
ax3.plot(h2o_cloud_mean * 1.e3, x1v)
ax3.set_xlabel(f"mean water cloud (g/kg)")
ax3.grid(True, alpha=0.3)

# Tight/clean layout
for ax in (ax1, ax2, ax3):
    ax.tick_params(axis="both", which="both", labelsize=8)

plt.tight_layout()
plt.show()

# Optionally save:
# fig.savefig("water_cloud_panel.png", bbox_inches="tight", dpi=200)
