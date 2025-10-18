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
NC_PATH = sys.argv[1] if len(sys.argv) > 1 else "simulation.nc"

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

# Resolve coordinates robustly
z = ds[VAR_HEIGHT]
x = ds[VAR_HORIZ]

# Select a 2D (z, x) slice for water and a 1D (z) slice for cooling rate
if VAR_TIME in ds[VAR_WATER].dims:
    water = ds[VAR_WATER].isel({VAR_TIME: TIME_INDEX}).load()
else:
    water = ds[VAR_WATER].load()

if VAR_TIME in ds[VAR_COOL].dims:
    cool = ds[VAR_COOL].isel({VAR_TIME: TIME_INDEX}).load()
else:
    cool = ds[VAR_COOL].load()

# Ensure shapes/coords align as expected
# Expect water: (z, x) and cool: (z)
# If your dims are named differently (e.g., ("level","i")), fix above names or transpose here:
if water.dims != (VAR_HEIGHT, VAR_HORIZ):
    water = water.transpose(VAR_HEIGHT, VAR_HORIZ)

# Compute horizontal mean profile of water (mean over X)
water_mean = water.mean(dim=VAR_HORIZ, skipna=True)

# -----------------------
# Figure layout
# -----------------------
fig = plt.figure(figsize=(10, 4.2), dpi=150)
gs = GridSpec(
    nrows=1, ncols=3, figure=fig,
    width_ratios=[1.1, 3.2, 1.1], wspace=0.25
)

ax_left  = fig.add_subplot(gs[0, 0])
ax_mid   = fig.add_subplot(gs[0, 1], sharey=ax_left)
ax_right = fig.add_subplot(gs[0, 2], sharey=ax_left)

# -----------------------
# Plot 1: Cooling rate vs height
# -----------------------
ax_left.plot(cool, z)
ax_left.set_xlabel(f"{VAR_COOL} [{cool.attrs.get('units','')}]".strip())
ax_left.set_ylabel(f"{VAR_HEIGHT} [{z.attrs.get('units','')}]".strip())
ax_left.grid(True, alpha=0.3)

# Optional: draw a reference zero line if cooling can be ±
if np.nanmin(cool.values) < 0 < np.nanmax(cool.values):
    ax_left.axvline(0, lw=0.8, alpha=0.6)

# -----------------------
# Plot 2: 2D contour of water (X–Z)
# -----------------------
# pcolormesh is a good default; use contourf if you prefer smoothed contours
pcm = ax_mid.pcolormesh(
    x, z, water,
    shading="auto"  # handles non-uniform coords
)
cbar = fig.colorbar(pcm, ax=ax_mid, pad=0.02)
cbar.set_label(f"{VAR_WATER} [{water.attrs.get('units','')}]".strip())

ax_mid.set_xlabel(f"{VAR_HORIZ} [{x.attrs.get('units','')}]".strip())
ax_mid.set_title("Water cloud field")

# -----------------------
# Plot 3: Horizontal mean water vs height
# -----------------------
ax_right.plot(water_mean, z)
ax_right.set_xlabel(f"mean({VAR_WATER})")
ax_right.grid(True, alpha=0.3)

# Tight/clean layout
for ax in (ax_left, ax_mid, ax_right):
    ax.tick_params(axis="both", which="both", labelsize=8)

fig.suptitle(Path(NC_PATH).name + (f"  |  {VAR_TIME}={TIME_INDEX}" if VAR_TIME in ds.dims else ""), y=1.02, fontsize=10)
plt.tight_layout()
plt.show()

# Optionally save:
# fig.savefig("water_cloud_panel.png", bbox_inches="tight", dpi=200)
