from typing import Tuple

import numpy as np
from scipy.interpolate import RegularGridInterpolator

def compute_dz_from_plev(
    plev: np.ndarray,   # (P,) pressure levels [Pa], typically decreasing with height
    rho: np.ndarray,    # (T, P, Lat, Lon) density [kg/m^3]
    grav: float,        # gravity [m/s^2]
) -> np.ndarray:
    """
    Compute layer thickness from pressure levels at each (t, lat, lon) using
    Δz = Δp / (ρ̄ g), where ρ̄ is layer-mean density (trapezoidal in 1/ρ).

    Args:
        plev: (P,) pressure levels [Pa], typically decreasing with height
        rho: (T, P, Lat, Lon) density [kg/m^3]
        grav: gravity [m/s^2]

    Returns:
        dz: (T, P-1, Lat, Lon) layer thickness [m] between pressure levels
    """
    # Ensure plev is monotonic from bottom (largest p) to top (smallest p)
    plev = np.asarray(plev)
    if plev.ndim != 1:
        raise ValueError("plev must be 1D of shape (P,).")
    if not np.all(np.diff(plev) < 0):
        # sort descending (bottom→top)
        order = np.argsort(-plev)
        plev = plev[order]
        rho = np.moveaxis(rho, 1, -1)[..., order]
        rho = np.moveaxis(rho, -1, 1)

    T, P, Lat, Lon = rho.shape
    if plev.shape[0] != P:
        raise ValueError("rho second dimension must match plev length.")

    # Layer thickness Δz_k between plev[k] and plev[k+1]
    # Use trapezoid rule on 1/rho:  1/ρ̄ ≈ 0.5*(1/ρ_k + 1/ρ_{k+1})
    inv_rho = 1.0 / rho
    inv_rho_mid = 0.5 * (inv_rho[:, :-1, :, :] + inv_rho[:, 1:, :, :])  # (T, P-1, Lat, Lon)
    dp = (plev[:-1] - plev[1:])[None, :, None, None]                    # (1, P-1, 1, 1), positive if p decreases upward
    dz_layers = (dp * inv_rho_mid) / grav                               # (T, P-1, Lat, Lon)

    return dz_layers


def compute_z_from_p(
    p: np.ndarray,      # (P,) pressure levels [Pa], typically decreasing with height
    rho: np.ndarray,    # (T, X, Y, P) density [kg/m^3]
    grav: float,        # gravity [m/s^2]
) -> np.ndarray:
    """
    Legacy function: Convert pressure levels to geometric height at each (t,x,y) using
    Δz = Δp / (ρ̄ g), where ρ̄ is layer-mean density (trapezoidal in 1/ρ).
    
    Note: This function is kept for backward compatibility. It assumes data in
    (T, X, Y, P) format. For new code using ECMWF ERA5 data in (T, P, Lat, Lon)
    format, use compute_dz_from_plev() and compute_heights_from_dz().

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


def compute_heights_from_dz(
    dz: np.ndarray,     # (T, P-1, Lat, Lon) layer thickness [m]
    topo: np.ndarray,   # (Lat, Lon) topographic elevation [m]
) -> np.ndarray:
    """
    Compute heights at pressure levels from layer thicknesses and topography.

    Args:
        dz: (T, P-1, Lat, Lon) layer thickness [m] between pressure levels
        topo: (Lat, Lon) topographic elevation [m] above reference

    Returns:
        z: (T, P, Lat, Lon) height [m] at pressure levels, with z[..., 0, :, :] = topo
    """
    T, P_minus_1, Lat, Lon = dz.shape
    P = P_minus_1 + 1

    # Initialize heights with bottom at topographic elevation
    z = np.zeros((T, P, Lat, Lon), dtype=dz.dtype)
    z[:, 0, :, :] = topo[None, :, :]  # Bottom level at topographic elevation

    # Add layer thicknesses cumulatively
    z[:, 1:, :, :] = topo[None, None, :, :] + np.cumsum(dz, axis=1)

    return z


def vertical_interp_to_z(
    z_col: np.ndarray,     # (..., P) heights increasing with level index
    v_col: np.ndarray,     # (..., P) values on same levels
    z_out: np.ndarray,     # (Z,) target heights (monotonic increasing)
    bounds_error: bool = True,  # If True, raise error when extrapolation would occur
) -> np.ndarray:
    """
    1D vertical interpolation along the last axis (P→Z) using np.interp per column.
    
    Args:
        z_col: (..., P) heights increasing with level index
        v_col: (..., P) values on same levels
        z_out: (Z,) target heights (monotonic increasing)
        bounds_error: If True, raise error when extrapolation would occur
        
    Returns:
        out: (..., Z) interpolated values
        
    Raises:
        ValueError: If bounds_error=True and extrapolation would occur
    """
    # Shapes
    *lead, P = z_col.shape
    Z = z_out.shape[0]
    out = np.full((*lead, Z), np.nan, dtype=v_col.dtype)

    # Flatten leading dims to loop neatly
    z_flat = z_col.reshape(-1, P)
    v_flat = v_col.reshape(-1, P)
    out_flat = out.reshape(-1, Z)

    extrapolation_detected = False
    for i in range(z_flat.shape[0]):
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
            # Check for extrapolation
            if np.any(z_out < z_sorted[0]) or np.any(z_out > z_sorted[-1]):
                extrapolation_detected = True
                
            # np.interp requires increasing x; clip outside → returns boundary value,
            # so we post-mask to NaN for true "no extrapolation".
            vals = np.interp(z_out, z_sorted, v_sorted)
            vals[(z_out < z_sorted[0]) | (z_out > z_sorted[-1])] = np.nan
            out_flat[i] = vals
        else:
            out_flat[i] = np.nan

    if bounds_error and extrapolation_detected:
        raise ValueError(
            f"Vertical interpolation would require extrapolation. "
            f"Target z range [{z_out.min():.2f}, {z_out.max():.2f}] exceeds "
            f"available data range."
        )

    return out


def latlon_to_xy(
    lats: np.ndarray,   # (Lat,) latitude [degrees]
    lons: np.ndarray,   # (Lon,) longitude [degrees]
    planet_radius: float,  # planet radius [m]
    lat_center: float = None,  # center latitude for projection [degrees]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert latitude/longitude to local Cartesian coordinates (Y, X) in meters.

    Args:
        lats: (Lat,) latitude array [degrees]
        lons: (Lon,) longitude array [degrees]
        planet_radius: radius of the planet [m]
        lat_center: center latitude for projection [degrees]. If None, uses midpoint of lats.

    Returns:
        Y: (Lat,) Y-coordinate in meters (North-South direction)
        X: (Lon,) X-coordinate in meters (East-West direction)
    """
    if lat_center is None:
        lat_center = 0.5 * (lats[0] + lats[-1])

    # Convert to radians
    lats_rad = np.radians(lats)
    lons_rad = np.radians(lons)
    lat_center_rad = np.radians(lat_center)

    # Y is simply arc length from center latitude
    Y = (lats_rad - lat_center_rad) * planet_radius

    # X uses the cosine of the center latitude for the projection
    lon_center_rad = np.radians(0.5 * (lons[0] + lons[-1]))
    X = (lons_rad - lon_center_rad) * planet_radius * np.cos(lat_center_rad)

    return Y, X


def horizontal_regrid_xy(
    x: np.ndarray, y: np.ndarray, field: np.ndarray, x_out: np.ndarray, y_out: np.ndarray,
    bounds_error: bool = True
) -> np.ndarray:
    """
    Regrid a 2D field f(x,y) defined on 1D grids x,y to a new regular grid x_out,y_out
    using SciPy RegularGridInterpolator (cubic).

    Args:
        x: (X,) input x-coordinate
        y: (Y,) input y-coordinate
        field: (X, Y) on the original grids.
        x_out: (X_out,) output x-coordinate
        y_out: (Y_out,) output y-coordinate
        bounds_error: If True, raise error when interpolation becomes extrapolation

    Returns:
        field_on_out: (X_out, Y_out)
    """
    # Check bounds to prevent extrapolation
    if bounds_error:
        if x_out.min() < x.min() or x_out.max() > x.max():
            raise ValueError(
                f"Output x range [{x_out.min():.2f}, {x_out.max():.2f}] exceeds "
                f"input x range [{x.min():.2f}, {x.max():.2f}]. Extrapolation is not allowed."
            )
        if y_out.min() < y.min() or y_out.max() > y.max():
            raise ValueError(
                f"Output y range [{y_out.min():.2f}, {y_out.max():.2f}] exceeds "
                f"input y range [{y.min():.2f}, {y.max():.2f}]. Extrapolation is not allowed."
            )

    interp = RegularGridInterpolator((x, y),
                                     field,
                                     method="cubic",
                                     bounds_error=False, 
                                     fill_value=np.nan)

    Xo, Yo = np.meshgrid(x_out, y_out, indexing="ij")  # (X_out, Y_out)
    pts = np.stack([Xo.ravel(), Yo.ravel()], axis=-1)
    Fo = interp(pts).reshape(Xo.shape)
    
    # Check if any NaNs were introduced (indicating out of bounds)
    if bounds_error and np.any(np.isnan(Fo)):
        raise ValueError("Interpolation resulted in NaN values, indicating extrapolation occurred.")
    
    return Fo


def regrid_pressure_to_height(
    var_tpll: np.ndarray,      # (T, P, Lat, Lon) variable on pressure-lat-lon grid
    rho_tpll: np.ndarray,      # (T, P, Lat, Lon) air density [kg/m^3]
    topo_ll: np.ndarray,       # (Lat, Lon) topographic elevation [m]
    plev: np.ndarray,          # (P,) pressure levels [Pa]
    lats: np.ndarray,          # (Lat,) latitude [degrees]
    lons: np.ndarray,          # (Lon,) longitude [degrees]
    x1f: np.ndarray,           # (Z,) vertical height coordinate [m]
    x2f: np.ndarray,           # (Y,) horizontal Y-coordinate [m]
    x3f: np.ndarray,           # (X,) horizontal X-coordinate [m]
    planet_grav: float,        # gravity constant [m/s^2]
    planet_radius: float,      # radius of the planet [m]
    bounds_error: bool = True, # If True, raise error when extrapolation would occur
) -> np.ndarray:
    """
    Complete regridding pipeline from ECMWF ERA5 pressure-lat-lon data to distance grids.
    
    Steps:
    1. Compute layer thickness from pressure levels and density
    2. Add layer thickness to topographic elevation to get heights at pressure levels
    3. Interpolate variables vertically to uniform height grid
    4. Convert lat/lon to local Cartesian coordinates (Y, X)
    5. Interpolate horizontally to desired output grid, matching centers
    
    Args:
        var_tpll: (T, P, Lat, Lon) variable on pressure-lat-lon grid
        rho_tpll: (T, P, Lat, Lon) air density [kg/m^3]
        topo_ll: (Lat, Lon) topographic elevation [m]
        plev: (P,) pressure levels [Pa], typically decreasing with height
        lats: (Lat,) latitude array [degrees]
        lons: (Lon,) longitude array [degrees]
        x1f: (Z,) vertical height coordinate [m]
        x2f: (Y,) horizontal Y-coordinate [m]
        x3f: (X,) horizontal X-coordinate [m]
        planet_grav: gravity constant [m/s^2]
        planet_radius: radius of the planet [m]
        bounds_error: If True, raise error when extrapolation would occur
        
    Returns:
        var_tzyx: (T, Z, Y, X) variable on distance grids
        
    Raises:
        ValueError: If output domain exceeds input domain when bounds_error=True
    """
    T, P, Lat, Lon = var_tpll.shape
    
    # Validate input shapes
    if rho_tpll.shape != var_tpll.shape:
        raise ValueError("rho_tpll must have the same shape as var_tpll")
    if topo_ll.shape != (Lat, Lon):
        raise ValueError(f"topo_ll shape {topo_ll.shape} must match (Lat={Lat}, Lon={Lon})")
    if plev.shape[0] != P:
        raise ValueError(f"plev length {plev.shape[0]} must match P dimension {P}")
    if lats.shape[0] != Lat:
        raise ValueError(f"lats length {lats.shape[0]} must match Lat dimension {Lat}")
    if lons.shape[0] != Lon:
        raise ValueError(f"lons length {lons.shape[0]} must match Lon dimension {Lon}")
    
    # Step 1: Compute layer thickness from pressure levels
    dz_tpll = compute_dz_from_plev(plev, rho_tpll, planet_grav)  # (T, P-1, Lat, Lon)
    
    # Step 2: Add layer thickness to topographic elevation
    z_tpll = compute_heights_from_dz(dz_tpll, topo_ll)  # (T, P, Lat, Lon)
    
    # Step 3: Vertical interpolation to uniform height grid
    # Reshape to (T, Lat, Lon, P) for vertical_interp_to_z
    var_tllp = np.moveaxis(var_tpll, 1, -1)  # (T, Lat, Lon, P)
    z_tllp = np.moveaxis(z_tpll, 1, -1)      # (T, Lat, Lon, P)
    
    # Interpolate vertically
    var_tllz = vertical_interp_to_z(z_tllp, var_tllp, x1f, bounds_error=bounds_error)  # (T, Lat, Lon, Z)
    
    # Step 4: Convert lat/lon to local Cartesian coordinates
    lat_center = 0.5 * (lats[0] + lats[-1])
    Y_coord, X_coord = latlon_to_xy(lats, lons, planet_radius, lat_center)
    
    # Step 5: Match centers of input and output domains
    # Center of input domain
    Y_center_in = 0.5 * (Y_coord[0] + Y_coord[-1])
    X_center_in = 0.5 * (X_coord[0] + X_coord[-1])
    
    # Center of output domain
    Y_center_out = 0.5 * (x2f[0] + x2f[-1])
    X_center_out = 0.5 * (x3f[0] + x3f[-1])
    
    # Shift output coordinates to match input center
    x2f_shifted = x2f + (Y_center_in - Y_center_out)
    x3f_shifted = x3f + (X_center_in - X_center_out)
    
    # Check if output domain is within input domain when bounds_error is True
    if bounds_error:
        if x2f_shifted.min() < Y_coord.min() or x2f_shifted.max() > Y_coord.max():
            raise ValueError(
                f"Output Y domain [{x2f_shifted.min():.2f}, {x2f_shifted.max():.2f}] m "
                f"exceeds input Y domain [{Y_coord.min():.2f}, {Y_coord.max():.2f}] m. "
                f"Interpolated domain must be embedded within or equal to original domain."
            )
        if x3f_shifted.min() < X_coord.min() or x3f_shifted.max() > X_coord.max():
            raise ValueError(
                f"Output X domain [{x3f_shifted.min():.2f}, {x3f_shifted.max():.2f}] m "
                f"exceeds input X domain [{X_coord.min():.2f}, {X_coord.max():.2f}] m. "
                f"Interpolated domain must be embedded within or equal to original domain."
            )
    
    # Step 6: Horizontal interpolation for each time and height level
    Z = len(x1f)
    Y_out = len(x2f)
    X_out = len(x3f)
    var_tzyx = np.full((T, Z, Y_out, X_out), np.nan, dtype=var_tpll.dtype)
    
    for ti in range(T):
        for zi in range(Z):
            slab = var_tllz[ti, :, :, zi]  # (Lat, Lon)
            # Note: horizontal_regrid_xy expects (X, Y) order, so we transpose
            slab_t = slab.T  # (Lon, Lat) -> (X, Y) ordering
            var_tzyx[ti, zi, :, :] = horizontal_regrid_xy(
                X_coord, Y_coord, slab_t, x3f_shifted, x2f_shifted, bounds_error=bounds_error
            ).T  # Transpose back to (Y, X)
    
    return var_tzyx


def regrid_topography(
    topo_ll: np.ndarray,       # (Lat, Lon) topographic elevation [m]
    lats: np.ndarray,          # (Lat,) latitude [degrees]
    lons: np.ndarray,          # (Lon,) longitude [degrees]
    x2f: np.ndarray,           # (Y,) horizontal Y-coordinate [m]
    x3f: np.ndarray,           # (X,) horizontal X-coordinate [m]
    planet_radius: float,      # radius of the planet [m]
    bounds_error: bool = True, # If True, raise error when extrapolation would occur
) -> np.ndarray:
    """
    Regrid topographic elevation from lat/lon to distance grids.
    
    Args:
        topo_ll: (Lat, Lon) topographic elevation [m]
        lats: (Lat,) latitude array [degrees]
        lons: (Lon,) longitude array [degrees]
        x2f: (Y,) horizontal Y-coordinate [m]
        x3f: (X,) horizontal X-coordinate [m]
        planet_radius: radius of the planet [m]
        bounds_error: If True, raise error when extrapolation would occur
        
    Returns:
        topo_yx: (Y, X) topographic elevation on distance grids
        
    Raises:
        ValueError: If output domain exceeds input domain when bounds_error=True
    """
    Lat, Lon = topo_ll.shape
    
    # Validate input shapes
    if lats.shape[0] != Lat:
        raise ValueError(f"lats length {lats.shape[0]} must match Lat dimension {Lat}")
    if lons.shape[0] != Lon:
        raise ValueError(f"lons length {lons.shape[0]} must match Lon dimension {Lon}")
    
    # Convert lat/lon to local Cartesian coordinates
    lat_center = 0.5 * (lats[0] + lats[-1])
    Y_coord, X_coord = latlon_to_xy(lats, lons, planet_radius, lat_center)
    
    # Match centers
    Y_center_in = 0.5 * (Y_coord[0] + Y_coord[-1])
    X_center_in = 0.5 * (X_coord[0] + X_coord[-1])
    Y_center_out = 0.5 * (x2f[0] + x2f[-1])
    X_center_out = 0.5 * (x3f[0] + x3f[-1])
    
    x2f_shifted = x2f + (Y_center_in - Y_center_out)
    x3f_shifted = x3f + (X_center_in - X_center_out)
    
    # Check bounds
    if bounds_error:
        if x2f_shifted.min() < Y_coord.min() or x2f_shifted.max() > Y_coord.max():
            raise ValueError(
                f"Output Y domain [{x2f_shifted.min():.2f}, {x2f_shifted.max():.2f}] m "
                f"exceeds input Y domain [{Y_coord.min():.2f}, {Y_coord.max():.2f}] m."
            )
        if x3f_shifted.min() < X_coord.min() or x3f_shifted.max() > X_coord.max():
            raise ValueError(
                f"Output X domain [{x3f_shifted.min():.2f}, {x3f_shifted.max():.2f}] m "
                f"exceeds input X domain [{X_coord.min():.2f}, {X_coord.max():.2f}] m."
            )
    
    # Horizontal interpolation
    # horizontal_regrid_xy expects (X, Y) order
    topo_t = topo_ll.T  # (Lon, Lat) -> (X, Y) ordering
    topo_yx = horizontal_regrid_xy(
        X_coord, Y_coord, topo_t, x3f_shifted, x2f_shifted, bounds_error=bounds_error
    ).T  # Transpose back to (Y, X)
    
    return topo_yx


def regrid_txyz_from_txyp(
    data_txyp: np.ndarray,  # (T, X, Y, P) variable on (t,x,y,p)
    z_txyp: np.ndarray,     # (T, X, Y, P) height at (t,x,y,p)
    inp_coord: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    out_coord: Tuple[np.ndarray, np.ndarray, np.ndarray],
    grav: float,
    ) -> np.ndarray:
    """
    Legacy function: Full pipeline: (t,x,y,p) → compute z(t,x,y,p) → interpolate vertically to z'
    → horizontally to (x',y') for each (t,z').
    
    Note: This function is kept for backward compatibility. For new code, use
    regrid_pressure_to_height() which handles the full ECMWF ERA5 pipeline.

    Returns:
        data_txyz: (T, X', Y', Z') on (t, x', y', z')
        t, x_out, y_out, z_out (passed-through for convenience)
    """
    t, x, y, p = inp_coord
    x_out, y_out, z_out = out_coord

    T, X, Y, P = data_txyp.shape
    if z_txyp.shape != data_txyp.shape:
        raise ValueError("z_txyp must have the same shape as data_txyp (T,X,Y,P).")

    # (1) Vertical interpolation to z' per (t,x,y) column
    #     Output shape: (T, X, Y, Z)
    data_txyz_on_orig_xy = vertical_interp_to_z(z_txyp, data_txyp, z_out, bounds_error=False)  # (T,X,Y,Z)

    # (2) Horizontal regrid for each (t,z') slice
    Xo, Yo, Zo = len(x_out), len(y_out), len(z_out)
    data_txyz = np.full((T, Xo, Yo, Zo), np.nan, dtype=data_txyp.dtype)

    for ti in range(T):
        for zi in range(Zo):
            slab = data_txyz_on_orig_xy[ti, :, :, zi]   # (X, Y)
            data_txyz[ti, :, :, zi] = horizontal_regrid_xy(x, y, slab, x_out, y_out, bounds_error=False)

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

