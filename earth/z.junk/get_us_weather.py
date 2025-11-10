import torch
import numpy as np
from netCDF4 import Dataset
from ecmwf_api.regrid import (
        compute_z_from_p,
        regrid_txyz_from_txyp
        )

def load_white_sand_weather(filename: str):
    module = torch.jit.load(filename)
    state_dict = module.state_dict()
    data = {k: v.numpy() for k, v in state_dict.items()}
    print(data.keys())
    data["temp"] = data.pop("t")
    data["vel3"] = data.pop("u")
    data["vel2"] = data.pop("v")
    #data["vel1"] = np.zeros_like(data["vel2"])  # no vertical velocity provided
    data["pres"] = data.pop("levels_hpa") * 100.
    data["time"] = data.pop("time_unix")
    data["lat"] = data.pop("latitude")
    data["lon"] = data.pop("longitude")
    print(f"time = {data['time']}")
    print(f"pressure shape = {data['pres'].shape}")

    return data

def interpolate_to_grid(data: dict[str, np.ndarray],
                        x3f: torch.Tensor,
                        x2f: torch.Tensor,
                        x1f: torch.Tensor) -> dict[str, np.ndarray]:
    """
    Interpolate weather data to the specified grid.

    Args:
        x3f: 1D tensor of x3 face coordinates (meters), excluding ghost cells
        x2f: 1D tensor of x2 face coordinates (meters), excluding ghost cells
        x1f: 1D tensor of x1 face coordinates (meters), excluding ghost cells
    """

    # constants
    radius_earth = 6371.e3  # radius of the Earth (m)
    grav_earth = 9.81  # m/s^2
    rgas_earth = 287.05  # J/(kg·K) for dry air

    # center latitude and longitude
    lat_center = 0.5 * (data["lat"][0] + data["lat"][-1])
    lon_center = 0.5 * (data["lon"][0] + data["lon"][-1])

    # convert to distance in meters
    lon_m = (data["lon"] / 180.0) * np.pi * radius_earth * np.cos(np.radians(lat_center))
    lat_m = data["lat"] / 180.0 * np.pi * radius_earth

    # adjust lon_span_m to be centered around zero
    lon_m -= 0.5 * (lon_m[0] + lon_m[-1])
    lat_m -= 0.5 * (lat_m[0] + lat_m[-1])

    # center of computational grid
    x2_center = 0.5 * (x2f[0] + x2f[-1])
    x3_center = 0.5 * (x3f[0] + x3f[-1])

    # match center and re-compute data coordinates
    x3_coord = x3_center + lon_m
    x2_coord = x2_center + lat_m

    print("x3_coord =", x3_coord)
    print("x2_coord =", x2_coord)

    time, pres = data["time"].copy(), data["pres"].copy()

    # compute density at pressure levels (...,np)
    data["rho"] = data["pres"] / (rgas_earth * data["temp"])
    data["pres"] = data["rho"] * (rgas_earth * data["temp"])

    # output grid data
    data_out = {}

    # interpolate field from (time, x3_coord, x2_coord, pres) -> (time, x3v, x2v, x1v)

    x3v = 0.5 * (x3f[:-1] + x3f[1:])
    x2v = 0.5 * (x2f[:-1] + x2f[1:])
    x1v = 0.5 * (x1f[:-1] + x1f[1:])

    # Build z(t,x,y,p)
    z_txyp = compute_z_from_p(pres, data["rho"], grav_earth) # (T,X,Y,P)

    for key in ["temp", "rho", "pres", "vel2", "vel3"]:
        print(f"Interpolating {key} ...")
        data_out[key] = regrid_txyz_from_txyp(
            data[key],
            z_txyp,
            (time, x3_coord, x2_coord, pres),
            (x3v, x2v, x1v),
            grav_earth,
        )
        print(f"{key} shape = {data_out[key].shape}")

    print(f"lon_m = {data['lon']} deg, {lon_m/1.e3} km")
    print(f"lat_m = {data['lat']} deg, {lat_m/1.e3} km")

    return data_out

# write weather data to netcdf file
def write_weather_to_netcdf(weather_data, filename: str,
                            resolution: str):
    with Dataset(filename, "w", format="NETCDF4") as ncfile:
        # permute data from (time, x3, x2, x1) to (time, x1, x2, x3)
        temp = weather_data["temp"].transpose(0, 3, 2, 1)
        vel2 = weather_data["vel2"].transpose(0, 3, 2, 1)
        vel3 = weather_data["vel3"].transpose(0, 3, 2, 1)
        pres = weather_data["pres"]
        ntime, nx1, nx2, nx3 = temp.shape

        # Define dimensions
        ncfile.createDimension("time", temp.shape[0])
        ncfile.createDimension("x1", temp.shape[1])
        ncfile.createDimension("x2", temp.shape[2])
        ncfile.createDimension("x3", temp.shape[3])

        # coordinate variables
        tvar = ncfile.createVariable("time", "f4", ("time",))
        zvar = ncfile.createVariable("x1", "f4", ("x1",))
        yvar = ncfile.createVariable("x2", "f4", ("x2",))
        xvar = ncfile.createVariable("x3", "f4", ("x3",))

        tvar.units = "seconds since 2025-01-01 00:00:00"
        tvar.axis = "T"
        xvar.units = "meters"
        xvar.axis = "X"
        yvar.units = "meters"
        yvar.axis = "Y"
        zvar.units = "pa"
        zvar.axis = "Z"

        tvar[:] = np.arange(temp.shape[0]).astype("f4")
        if pres.ndim == 1:
            zvar[:] = pres.astype("f4")
        else:
            zvar[:] = pres.mean(axis=(0,1,2)).astype("f4")
        yvar[:] = np.arange(temp.shape[2]).astype("f4")
        xvar[:] = np.arange(temp.shape[3]).astype("f4")
        
        # Create variables
        temp_var = ncfile.createVariable("temp", "f4", ("time", "x1", "x2", "x3"))
        temp_var.units = "K"
        temp_var.long_name = "Air Temperature"
        temp_var[:] = temp

        pres_var = ncfile.createVariable("pres", "f4", ("time", "x1", "x2", "x3"))
        pres_var.units = "Pa"
        pres_var.long_name = "Air Pressure"
        if pres.ndim == 1:
            # broadcast pressure (nx1,) to (ntime, nx1, nx2, nx3)
            pres_var[:] = pres[None, :, None, None].astype("f4")
        else:
            pres_var[:] = pres.transpose(0, 3, 2, 1).astype("f4")

        vel2_var = ncfile.createVariable("vel2", "f4", ("time", "x1", "x2", "x3"))
        vel2_var.units = "m/s"
        vel2_var.long_name = "Velocity component in Y-direction"
        vel2_var[:] = vel2

        vel3_var = ncfile.createVariable("vel3", "f4", ("time", "x1", "x2", "x3"))
        vel3_var.units = "m/s"
        vel3_var.long_name = "Velocity component in X-direction"
        vel3_var[:] = vel3

        # Global attributes
        ncfile.description = "4D weather data loaded from TorchScript"
        ncfile.source = "Generated by write_weather_to_netcdf function"

def create_weather_input(fname: str,
                         x3f: torch.Tensor,
                         x2f: torch.Tensor,
                         x1f: torch.Tensor,
                         nghost: int):
    fname = "era5_by_pressure_modules_2025_Jan_01_AAA.pt"
    data = load_white_sand_weather(fname)

    data_out = interpolate_to_grid(data,
                        x3f=x3f[nghost:-nghost],
                        x2f=x2f[nghost:-nghost],
                        x1f=x1f[nghost:-nghost])

    # convert to torch tensors and add ghost cells
    result = {}

    for key in ["rho", "vel2", "vel3", "pres"]:
        arr = data_out[key]
        ntime, nx3, nx2, nx1 = arr.shape
        tensor = torch.full((ntime, nx3 + 2*nghost, nx2 + 2*nghost, nx1 + 2*nghost),
                            0., dtype=torch.float64)
        tensor[:, nghost:-nghost, nghost:-nghost, nghost:-nghost] = torch.from_numpy(arr)
        result[key] = tensor
        print(f"{key} shape = {result[key].shape}")

        # fill ghost cells by nearest neighbor
        result[key][:, :nghost, :, :] = result[key][:, nghost:nghost+1, :, :]
        result[key][:, -nghost:, :, :] = result[key][:, -nghost-1:-nghost, :, :]
        result[key][:, :, :nghost, :] = result[key][:, :, nghost:nghost+1, :]
        result[key][:, :, -nghost:, :] = result[key][:, :, -nghost-1:-nghost, :]
        result[key][:, :, :, :nghost] = result[key][:, :, :, nghost:nghost+1]
        result[key][:, :, :, -nghost:] = result[key][:, :, :, -nghost-1:-nghost]

    return result


def test_load_white_sand_weather():
    fname = "era5_by_pressure_modules_2025_Jan_01_AAA.pt"
    data = load_white_sand_weather(fname)
    data_out = interpolate_to_grid(data,
                        x3f=torch.linspace(-20.E3, 20.E3, 201),
                        x2f=torch.linspace(-20.E3, 20.E3, 201),
                        x1f=torch.linspace(0., 10.E3, 101))
    write_weather_to_netcdf(data, "white_sand_weather_4d.nc", "40m")
    write_weather_to_netcdf(data_out, "white_sand_weather_4d_grid.nc", "40m")

if __name__ == "__main__":
    test_load_white_sand_weather()
