from typing import Tuple, Optional
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# Parallelization thresholds
_VERTICAL_INTERP_PARALLEL_THRESHOLD = 100  # Minimum columns for parallel vertical interpolation
_HORIZONTAL_REGRID_PARALLEL_THRESHOLD = 10  # Minimum slices for parallel horizontal regridding

def compute_z_from_p(
    p: np.ndarray,      # (P,) pressure levels [Pa], typically decreasing with height
    rho: np.ndarray,    # (T, X, Y, P) density [kg/m^3]
    grav: float,        # gravity [m/s^2]
) -> np.ndarray:
    """
    Convert pressure levels to geometric height at each (t,x,y) using
    Δz = Δp / (ρ̄ g), where ρ̄ is layer-mean density (trapezoidal in 1/ρ).

    Returns:
        z: (T, X, Y, P) height [m], with z[..., 0] = 0 at the bottom level and increasing upward.
    """
    # Ensure p is monotonic from bottom (largest p) to top (smallest p)
    p = np.asarray(p)
    if p.ndim != 1:
        raise ValueError("p must be 1D of shape (P,).")
    if not np.all(np.diff(p) < 0):
        # sort descending (bottom→top)
        order = np.argsort(-p)
        p = p[order]
        rho = rho[..., order]

    T, X, Y, P = rho.shape
    if p.shape[0] != P:
        raise ValueError("rho last dimension must match p length.")

    # Layer thickness Δz_k between p[k] and p[k+1]
    # Use trapezoid rule on 1/rho:  1/ρ̄ ≈ 0.5*(1/ρ_k + 1/ρ_{k+1})
    inv_rho = 1.0 / rho
    inv_rho_mid = 0.5 * (inv_rho[..., :-1] + inv_rho[..., 1:]) # (T,X,Y,P-1)
    dp = (p[:-1] - p[1:])[None, None, None, :]                 # (1,1,1,P-1), positive if p decreases upward
    dz_layers = (dp * inv_rho_mid) / grav                      # (T,X,Y,P-1)

    # z at full levels with z[...,0] = 0 (relative to bottom)
    z = np.zeros((T, X, Y, P), dtype=rho.dtype)
    z[..., 1:] = np.cumsum(dz_layers, axis=-1)
    return z


def _vertical_interp_single_column(args):
    """
    Helper function for parallel vertical interpolation of a single column.
    
    Args:
        args: Tuple of (index, z_col, v_col, z_out)
        
    Returns:
        Tuple of (index, interpolated_values)
    """
    i, zc, vc, z_out = args
    
    # Require strictly increasing z for np.interp
    # If any non-monotonic segments (rare, but can happen), we sort by z
    order = np.argsort(zc)
    z_sorted = zc[order]
    v_sorted = vc[order]
    
    # Remove duplicates in z (if any)
    mask = np.diff(z_sorted, prepend=z_sorted[0]-1) > 0
    z_sorted = z_sorted[mask]
    v_sorted = v_sorted[mask]
    
    if z_sorted.size >= 2:
        # np.interp requires increasing x; clip outside → returns boundary value,
        # so we post-mask to NaN for true "no extrapolation".
        vals = np.interp(z_out, z_sorted, v_sorted)
        vals[(z_out < z_sorted[0]) | (z_out > z_sorted[-1])] = np.nan
        return (i, vals)
    else:
        return (i, np.full(z_out.shape[0], np.nan, dtype=vc.dtype))


def vertical_interp_to_z(
    z_col: np.ndarray,     # (..., P) heights increasing with level index
    v_col: np.ndarray,     # (..., P) values on same levels
    z_out: np.ndarray,     # (Z,) target heights (monotonic increasing)
    n_jobs: Optional[int] = None,  # Number of parallel workers (None = auto, 1 = sequential)
) -> np.ndarray:
    """
    1D vertical interpolation along the last axis (P→Z) using np.interp per column.
    Extrapolation is not performed; outside domain → NaN.
    
    Args:
        z_col: (..., P) heights increasing with level index
        v_col: (..., P) values on same levels
        z_out: (Z,) target heights (monotonic increasing)
        n_jobs: Number of parallel workers (None=auto, 1=sequential, -1=all CPUs)
        
    Returns:
        out: (..., Z) interpolated values
    """
    # Shapes
    *lead, P = z_col.shape
    Z = z_out.shape[0]
    out = np.full((*lead, Z), np.nan, dtype=v_col.dtype)

    # Flatten leading dims to loop neatly
    z_flat = z_col.reshape(-1, P)
    v_flat = v_col.reshape(-1, P)
    out_flat = out.reshape(-1, Z)
    
    n_cols = z_flat.shape[0]
    
    # Determine number of jobs
    if n_jobs is None:
        # Auto: Use parallelization only if we have enough columns to make it worthwhile
        n_jobs = min(cpu_count(), max(1, n_cols // _VERTICAL_INTERP_PARALLEL_THRESHOLD)) if n_cols > _VERTICAL_INTERP_PARALLEL_THRESHOLD else 1
    elif n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    if n_jobs == 1:
        # Sequential execution
        for i in range(n_cols):
            zc = z_flat[i]
            vc = v_flat[i]

            # Require strictly increasing z for np.interp
            # If any non-monotonic segments (rare, but can happen), we sort by z
            order = np.argsort(zc)
            z_sorted = zc[order]
            v_sorted = vc[order]

            # Remove duplicates in z (if any)
            mask = np.diff(z_sorted, prepend=z_sorted[0]-1) > 0
            z_sorted = z_sorted[mask]
            v_sorted = v_sorted[mask]

            if z_sorted.size >= 2:
                # np.interp requires increasing x; clip outside → returns boundary value,
                # so we post-mask to NaN for true "no extrapolation".
                vals = np.interp(z_out, z_sorted, v_sorted)
                vals[(z_out < z_sorted[0]) | (z_out > z_sorted[-1])] = np.nan
                out_flat[i] = vals
            else:
                out_flat[i] = np.nan
    else:
        # Parallel execution
        args_list = [(i, z_flat[i], v_flat[i], z_out) for i in range(n_cols)]
        
        with Pool(processes=n_jobs) as pool:
            results = pool.map(_vertical_interp_single_column, args_list)
        
        # Collect results
        for i, vals in results:
            out_flat[i] = vals

    return out


def horizontal_regrid_xy(
    x: np.ndarray, y: np.ndarray, field: np.ndarray, x_out: np.ndarray, y_out: np.ndarray
) -> np.ndarray:
    """
    Regrid a 2D field f(x,y) defined on 1D grids x,y to a new regular grid x_out,y_out
    using SciPy RegularGridInterpolator (linear).

    Args:
        field: (X, Y) on the original grids.
    Returns:
        field_on_out: (X_out, Y_out)
    """
    interp = RegularGridInterpolator((x, y),
                                     field,
                                     method="cubic",
                                     bounds_error=False, 
                                     fill_value=np.nan)

    Xo, Yo = np.meshgrid(x_out, y_out, indexing="ij")  # (X_out, Y_out)
    pts = np.stack([Xo.ravel(), Yo.ravel()], axis=-1)
    Fo = interp(pts).reshape(Xo.shape)
    return Fo


def _regrid_horizontal_slice(args):
    """
    Helper function for parallel horizontal regridding of a single (time, height) slice.
    
    Args:
        args: Tuple of (ti, zi, slab, x, y, x_out, y_out)
        
    Returns:
        Tuple of (ti, zi, regridded_slab)
    """
    ti, zi, slab, x, y, x_out, y_out = args
    result = horizontal_regrid_xy(x, y, slab, x_out, y_out)
    return (ti, zi, result)


def regrid_txyz_from_txyp(
    data_txyp: np.ndarray,  # (T, X, Y, P) variable on (t,x,y,p)
    z_txyp: np.ndarray,     # (T, X, Y, P) height at (t,x,y,p)
    inp_coord: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    out_coord: Tuple[np.ndarray, np.ndarray, np.ndarray],
    grav: float,
    n_jobs: Optional[int] = None,  # Number of parallel workers (None = auto, 1 = sequential)
    ) -> np.ndarray:
    """
    Full pipeline: (t,x,y,p) → compute z(t,x,y,p) → interpolate vertically to z'
    → horizontally to (x',y') for each (t,z'). Both steps are parallelized.

    Args:
        data_txyp: (T, X, Y, P) variable on (t,x,y,p)
        z_txyp: (T, X, Y, P) height at (t,x,y,p)
        inp_coord: Tuple of (t, x, y, p) coordinate arrays
        out_coord: Tuple of (x_out, y_out, z_out) coordinate arrays
        grav: Gravity constant [m/s^2]
        n_jobs: Number of parallel workers (None=auto, 1=sequential, -1=all CPUs)

    Returns:
        data_txyz: (T, X', Y', Z') on (t, x', y', z')
    """
    t, x, y, p = inp_coord
    x_out, y_out, z_out = out_coord

    T, X, Y, P = data_txyp.shape
    if z_txyp.shape != data_txyp.shape:
        raise ValueError("rho_txyp must have the same shape as data_txyp (T,X,Y,P).")

    # (1) Vertical interpolation to z' per (t,x,y) column (parallelized)
    #     Output shape: (T, X, Y, Z)
    data_txyz_on_orig_xy = vertical_interp_to_z(z_txyp, data_txyp, z_out, n_jobs=n_jobs)  # (T,X,Y,Z)

    # (2) Horizontal regrid for each (t,z') slice (parallelized)
    Xo, Yo, Zo = len(x_out), len(y_out), len(z_out)
    data_txyz = np.full((T, Xo, Yo, Zo), np.nan, dtype=data_txyp.dtype)
    
    # Determine number of jobs for horizontal regridding
    n_slices = T * Zo
    if n_jobs is None:
        # Auto: Use parallelization only if we have enough slices to make it worthwhile
        n_jobs_horiz = min(cpu_count(), max(1, n_slices // _HORIZONTAL_REGRID_PARALLEL_THRESHOLD)) if n_slices > _HORIZONTAL_REGRID_PARALLEL_THRESHOLD else 1
    elif n_jobs == -1:
        n_jobs_horiz = cpu_count()
    elif n_jobs < 1:
        n_jobs_horiz = 1
    else:
        n_jobs_horiz = n_jobs

    if n_jobs_horiz == 1:
        # Sequential execution
        for ti in range(T):
            for zi in range(Zo):
                slab = data_txyz_on_orig_xy[ti, :, :, zi]   # (X, Y)
                data_txyz[ti, :, :, zi] = horizontal_regrid_xy(x, y, slab, x_out, y_out)
    else:
        # Parallel execution
        args_list = [
            (ti, zi, data_txyz_on_orig_xy[ti, :, :, zi], x, y, x_out, y_out)
            for ti in range(T) for zi in range(Zo)
        ]
        
        with Pool(processes=n_jobs_horiz) as pool:
            results = pool.map(_regrid_horizontal_slice, args_list)
        
        # Collect results
        for ti, zi, result in results:
            data_txyz[ti, :, :, zi] = result

    return data_txyz


# --------------------------
# Example usage (skeleton):
# --------------------------
# T, X, Y, P = 4, 50, 60, 30
# t = np.arange(T)
# x = np.linspace(-10.0, 10.0, X)
# y = np.linspace(30.0, 50.0, Y)
# p = np.linspace(100000.0, 10000.0, P)  # 1000→100 hPa, descending = bottom→top
#
# # Fake inputs
# rho = 1.2 + 0.1*np.random.rand(T, X, Y, P)         # kg/m^3
# var = np.random.rand(T, X, Y, P)                   # some variable on p-levels
#
# # Target grids
# x_out = np.linspace(-9.5, 9.5, 64)
# y_out = np.linspace(30.5, 49.5, 72)
# z_out = np.linspace(0.0, 15000.0, 40)              # 0–15 km
#
# data_txyz, t_o, x_o, y_o, z_o = regrid_txyz_from_txyp(
#     var, rho, t, x, y, p, x_out, y_out, z_out, g=9.80665
# )

