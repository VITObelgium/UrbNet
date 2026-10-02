# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import calendar
import json
import math
import os
import shlex
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rioxr
import torch
import xarray as xr
from loguru import logger
from netCDF4 import Dataset as NetCDFDataset
from netCDF4 import num2date
from pvlib.location import Location
from pyproj import Transformer
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, uniform_filter
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset as TorchDataset

from urbnet.abudhabi.settings import (
    AD_INPUT_FOLDER_PREFIX,
    CITIES_WITH_URBCLIM_DATA,
    NORMALIZATION_CHANNEL_SPEC_TEMPERATURE,
    NUMBER_OF_PREDICTOR_CHANNELS_HUMIDITY,
    NUMBER_OF_PREDICTOR_CHANNELS_TEMPERATURE,
    NUMBER_OF_PREDICTOR_CHANNELS_WBGT,
    OVERALL_FOLDER_PREFIX,
    OUTPUT_FOLDER_ROOT,
    UAE_CITIES,
    UAE_INPUT_FOLDER_PREFIX,
)

HOUR_DEVIATION_FROM_UTC = 4

CACHE_SIZE = len(CITIES_WITH_URBCLIM_DATA) * (2024 - 2015 + 1)


def execute(command, shell=True):
    return subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
        shell=shell,
    )


def calculate_clear_sky_rs(latitude, longitude, year=2024, altitude=0, utc_offset_minutes=30):
    """
    Compute hourly clear-sky GHI for a whole year using a fixed UTC offset (e.g. UTC+30 min).

    Parameters:
        latitude (float): Latitude in degrees.
        longitude (float): Longitude in degrees.
        year (int): Year to compute.
        altitude (float): Site altitude in meters.
        utc_offset_minutes (int): Time offset in minutes from UTC (e.g. 30 for UTC+00:30).

    Returns:
        pandas.DataFrame: Hourly clear-sky radiation (GHI, DNI, DHI) with timestamped index.
    """
    # Create a fixed offset timezone (e.g. UTC+00:30)
    offset = timezone(timedelta(minutes=utc_offset_minutes))

    # Create time range for full year in the custom time zone
    times = pd.date_range(
        start=f"{year}-01-01 00:00", end=f"{year}-12-31 23:00", freq="1h", tz=offset
    )

    # Create Location object with dummy IANA time zone (we only use it for solar position, not local time)
    location = Location(latitude, longitude, tz="UTC", altitude=altitude)

    # pvlib assumes timestamps are localized — we pass ours with offset explicitly
    clear_sky = location.get_clearsky(times, model="simplified_solis")

    # Re-attach the custom timezone, just in case pvlib drops it
    clear_sky.index = clear_sky.index.tz_convert(offset)

    return clear_sky


def patches_per_timestep(H, W, patch_size):
    return math.ceil(H / patch_size) * math.ceil(W / patch_size)


def patches_per_timestamp_for_city(city, patch_size):
    ndvi_city = load_ndvi(city)
    _, H, W = ndvi_city.shape
    final_height = H - 20
    final_width = W - 20
    return patches_per_timestep(final_height, final_width, patch_size)


@lru_cache(maxsize=CACHE_SIZE)
def load_era5(city, year):
    logger.info(f"Loading ERA 5 input data for {city} in {year}")
    meteo_file = OVERALL_FOLDER_PREFIX / "urbsim" / city / "meteo" / f"{city}_meteo_{year}.nc"
    ds_meteo = xr.open_dataset(meteo_file)
    # combine with the meteorological parameters in one xarray
    ds = xr.Dataset(
        data_vars={
            "T2M": ("time", ds_meteo.T[:, 0].data),
            "U10M": ("time", ds_meteo.U[:, 0].data),
            "V10M": ("time", ds_meteo.V[:, 0].data),
            "Q2M": ("time", ds_meteo.Q[:, 0].data),
            "Rs": ("time", ds_meteo.Rs[:].data),
            "CSF": ("time", ds_meteo.Rs[:].data),
        },
        coords={
            "time": pd.to_datetime(
                dict(
                    year=ds_meteo.YEAR.values,
                    month=ds_meteo.MONTH.values,
                    day=ds_meteo.DAY.values,
                    hour=ds_meteo.HOUR.values,
                    minute=ds_meteo.MINUTE.values,
                )
            )
        },
    )

    # data is available every 20 mins, we only need the hours
    hour_mask = ds.time.dt.minute == 0
    ds_hourly = ds.sel(time=ds.time[hour_mask])

    # get the center coordinate of the city
    terrain_file = OVERALL_FOLDER_PREFIX / "urbsim" / city / "terrain" / f"{city}_surface.nc"
    ds_terrain = xr.open_dataset(terrain_file)
    city_lat = ds_terrain.LAT.data.mean()
    city_lon = ds_terrain.LON.data.mean()
    # city_alt = ds_terrain.DEM.data.mean() -- not used

    # mask radiation with clear sky radiation to get cloud cover
    rs_yearly = calculate_clear_sky_rs(
        latitude=city_lat, longitude=city_lon, altitude=0, year=year
    )  # altitude to 0
    if len(ds_hourly.Rs.values) != len(rs_yearly.ghi.values):
        raise ValueError(
            f"Mismatch in number of time steps: {len(ds_hourly.Rs.values)} vs {len(rs_yearly.ghi.values)}"
        )
    else:
        rs = ds_hourly.Rs.values
        ghi = rs_yearly.ghi.values

        # Default to 1
        cs_frac = np.ones_like(rs)

        # Define valid (non-NaN and non-zero denominator) mask
        valid_mask = (ghi != 0) & ~np.isnan(rs) & ~np.isnan(ghi)

        # Regular division where valid
        cs_frac[valid_mask] = rs[valid_mask] / ghi[valid_mask]

        # Special handling where GHI == 0
        zero_mask = (ghi == 0) & ~np.isnan(rs)
        cs_frac[zero_mask & (rs < 0)] = 0
        cs_frac[zero_mask & (rs > 0)] = 1

        # Clamp to [0, 1]
        cs_frac[cs_frac > 1] = 1
        cs_frac[cs_frac < 0] = 0

        ds_hourly.CSF.values = cs_frac

    # Add wind speed (WS)
    ds_hourly["WIND_SPEED"] = xr.apply_ufunc(
        np.sqrt,
        ds_hourly.U10M**2 + ds_hourly.V10M**2,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )

    # Add the wind direction, calculated from U10M and V10M
    ds_hourly["WIND_DIR"] = xr.apply_ufunc(
        np.arctan2,
        ds_hourly.V10M,
        ds_hourly.U10M,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )

    # Add sin and cos of the wind direction (wind_dir is already in radians)
    ds_hourly["WIND_DIR_SIN"] = xr.apply_ufunc(
        np.sin,
        ds_hourly.WIND_DIR,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )

    ds_hourly["WIND_DIR_COS"] = xr.apply_ufunc(
        np.cos,
        ds_hourly.WIND_DIR,
        input_core_dims=[["time"]],
        output_core_dims=[["time"]],
        vectorize=True,
    )

    # also get the fraction of clear sky radiation as parameter
    max_rs_cs = 1364
    ds_hourly["Rs"] = ds_hourly["Rs"] / max_rs_cs

    # Q2M is in g/kg, we convert to kg/kg
    ds_hourly["Q2M"] = ds_hourly["Q2M"] / 1000.0

    return ds_hourly


@lru_cache(maxsize=len(CITIES_WITH_URBCLIM_DATA))
def load_ndvi(city):
    logger.info(f"Loading NDVI data for {city}")
    months = [calendar.month_abbr[i].lower() for i in range(1, 13)]
    for idx, month in enumerate(months):
        ndvi_filename = f"{OVERALL_FOLDER_PREFIX}/urbsim/{city}/terrain/log/temp_veg_{month}.tif"
        ds = rioxr.open_rasterio(ndvi_filename)
        ds = ds.rename({"band": "month"})
        ds = ds.assign_coords(month=[idx + 1])
        if idx == 0:
            ndvi_data = ds.copy()
        else:
            ndvi_data = xr.concat([ndvi_data, ds], dim="month")
        ndvi_data.values[ndvi_data.values > 1] = 1
        ndvi_data.values[ndvi_data.values < -1] = -1
    return ndvi_data


@lru_cache(maxsize=CACHE_SIZE)
def load_urbclim(city, year):
    logger.info(f"Loading UrbClim data for {city} in {year}")
    urbclim_filename = f"{OVERALL_FOLDER_PREFIX}/urbsim/{city}/urbclim/{year}/{city}_output.nc"
    ds = xr.open_dataset(urbclim_filename, chunks={"t": 100})
    # urbclim data is flipped along the y-axis
    ds = ds.reindex(y=list(reversed(ds.y)))
    # WBGT is in local time (timediff for UAE = 4h to UTC)
    wbgt_filename = f"{OVERALL_FOLDER_PREFIX}/urbsim/{city}/wbgt/WBGT_{year}.nc"
    ds_WBGT = xr.open_dataset(wbgt_filename, chunks={"time": 100})
    ds_WBGT = ds_WBGT.rename({"time": "t"})
    ds_WBGT = ds_WBGT.assign_coords(
        t=ds_WBGT.t.values - np.timedelta64(HOUR_DEVIATION_FROM_UTC, "h")
    )
    # small mismatch between coordinates of UrbClim and WBGT
    ds_WBGT = ds_WBGT.assign_coords(x=ds.x.values, y=ds.y.values)
    ds_all = xr.merge([ds, ds_WBGT])
    return ds_all


@lru_cache(maxsize=len(CITIES_WITH_URBCLIM_DATA))
def load_dtc(city):
    logger.info(f"Loading distance to coast data for {city}")
    distance_to_coast_filename = (
        OUTPUT_FOLDER_ROOT / "derived" / city / "distance_to_coast" / f"{city}.tif"
    )
    if not os.path.exists(distance_to_coast_filename):
        # Load ndvi to get the extent and resolution
        ndvi = load_ndvi(city)
        min_x = ndvi.x.values.min()
        min_y = ndvi.y.values.min()
        max_x = ndvi.x.values.max()
        max_y = ndvi.y.values.max()
        res = ndvi.x.values[1] - ndvi.x.values[0]

        # Run gdal
        os.makedirs(distance_to_coast_filename.parent, exist_ok=True)
        source_distance_to_coast = (
            UAE_INPUT_FOLDER_PREFIX / "distance_to_coast" / "full_domain_EPSG3857.tif"
        )
        gdal_cmd = (
            "gdalwarp "
            + f"-te {min_x - 50} {min_y - 50} {max_x + 50} {max_y + 50} "
            + f"-tr {res} {res} "
            + f"{shlex.quote(str(source_distance_to_coast))} "
            + shlex.quote(str(distance_to_coast_filename))
        )
        execute(gdal_cmd)
    distance_coast_data = rioxr.open_rasterio(distance_to_coast_filename)
    # remove the sharp edges
    distance_coast_data.values = gaussian_filter(distance_coast_data.values, sigma=4)
    return distance_coast_data


def get_radiation_all_times(
    n_circ_class_clusters,
    start_year,
    end_year,
    operational_forecast_folder,
):
    """
    Add radiation information to the cluster classification
    """
    if "ECMWF" in operational_forecast_folder or "ICON" in operational_forecast_folder:
        df_clusters = pd.read_csv(f"{operational_forecast_folder}/clusters.csv")
    else:
        df_clusters = pd.read_csv(
            OVERALL_FOLDER_PREFIX
            / "input_data"
            / "circulation_domain"
            / f"circulation_type_classification_{n_circ_class_clusters}_clusters.csv"
        )
    df_clusters["time"] = pd.to_datetime(df_clusters["time"])
    df_clusters.set_index("time", inplace=True)

    # add another layer stating the day or night based on radiation
    for year in range(start_year, end_year + 1):
        CSR_year = calculate_clear_sky_rs(24.47, 54.37, year=year)
        CSR_year["day_night"] = CSR_year["ghi"].apply(lambda x: "day" if x > 1 else "night")
        CSR_year.index = CSR_year.index.tz_localize(None)
        if year == start_year:
            df_clusters = df_clusters.join(CSR_year)
        else:
            df_clusters = df_clusters.combine_first(CSR_year[CSR_year.index.isin(df_clusters.index)])
    return df_clusters.drop(columns=["ghi", "dhi", "dni"])


@lru_cache(maxsize=10)
def get_all_times(
    n_circ_class_clusters,
    circ_class_cluster_idx,
    use_individual_model_per_cluster,
    use_day_night_model,
    time_of_the_day,
    start_year,
    end_year,
):
    """
    Get all the times from we which we want to sample
    See https://confluence.vito.be/spaces/CLIM/pages/359728485/Weather+system+classification
    """
    sample_clusters = get_radiation_all_times(
        n_circ_class_clusters, start_year, end_year, operational_forecast_folder="None"
    )

    if use_individual_model_per_cluster:
        # Alternative, subset each circulation pattern separately
        sample_clusters = sample_clusters[(sample_clusters["cluster"] == circ_class_cluster_idx)]
    if use_day_night_model:
        if time_of_the_day == "day":
            sample_clusters = sample_clusters[(sample_clusters["day_night"] == "day")]
        elif time_of_the_day == "night":
            sample_clusters = sample_clusters[(sample_clusters["day_night"] == "night")]

    sample_times = sample_clusters.index.tolist()
    return sample_times


def build_dataset_index(
    cities: List[str],
    patch_size: int,
    start_year,
    end_year,
    n_circ_class_clusters,
    circ_class_cluster_idx,
    use_individual_model_per_cluster,
    use_day_night_model,
    time_of_the_day,
):
    """
    Build full 'index' of all possible samples that can be drawn.
    For every time, for every cit, there will be a different number of patches depending on the size of the city
    """
    times = get_all_times(
        n_circ_class_clusters,
        circ_class_cluster_idx,
        use_individual_model_per_cluster,
        use_day_night_model,
        time_of_the_day,
        start_year,
        end_year,
    )

    patches_per_city = {city: patches_per_timestamp_for_city(city, patch_size) for city in cities}

    return pd.DataFrame(
        [
            {"time": time, "city": city}
            for time in times
            for city, nb_patches in patches_per_city.items()
            for _ in range(nb_patches)
        ]
    )


@lru_cache(maxsize=10)
def get_land_use_data(city):
    return rioxr.open_rasterio(
        OVERALL_FOLDER_PREFIX / "urbsim" / city / "terrain" / "log" / "temp_ua.tif",
    ).astype(int)


@lru_cache(maxsize=10)
def get_land_use_data_encoded(city):
    landuse_data = get_land_use_data(city)
    landuse_data = landuse_data.where(landuse_data != 53, 51)
    # One-hot encode the categorical data so the channels are added in the last dimension
    # Dimensions will become (height, width, channels)
    landuse_classes = [10, 20, 30, 40, 50, 51, 52, 60, 80, 90, 95, 200]
    encoder = OneHotEncoder(
        sparse_output=False,
        categories=[landuse_classes],
        handle_unknown="ignore",
    )
    landuse_encoded = encoder.fit_transform(
        np.squeeze(landuse_data.values).flatten().reshape(-1, 1)
    ).reshape(len(landuse_data.y), len(landuse_data.x), -1)
    return landuse_encoded


@lru_cache(maxsize=10)
def get_digital_elevation_data(city, year):
    digital_elevation_data = rioxr.open_rasterio(
        OVERALL_FOLDER_PREFIX / "urbsim" / city / "terrain" / "log" / "temp_dem.tif",
    )
    ERA5 = xr.open_dataset(
        OVERALL_FOLDER_PREFIX / "urbsim" / city / "meteo" / f"{city}_meteo_{year}.nc"
    )
    return digital_elevation_data - float(ERA5.Altitude)


@lru_cache(maxsize=10)
def get_building_fraction_data(city):
    building_fraction_data = rioxr.open_rasterio(
        OVERALL_FOLDER_PREFIX / "urbsim" / city / "terrain" / "log" / "temp_buildings.tif"
    )
    # The building fraction data is in percentage, so we divide by 100 to get a fraction
    return building_fraction_data / 100


@lru_cache(maxsize=10)
def get_soil_sealing_data(city):
    return rioxr.open_rasterio(
        OVERALL_FOLDER_PREFIX / "urbsim" / city / "terrain" / "log" / "temp_soil.tif"
    )


def get_patch(time, city, patch_size, use_day_night_model, output_var, rng):
    meteo_data = load_era5(city=city, year=time.year).sel(time=time)
    ndvi_data = load_ndvi(city)
    ndvi_data_for_time = ndvi_data.values[time.month - 1, :, :]
    urbclim_data = load_urbclim(city=city, year=time.year).sel(t=[time])
    distance_coast_data = load_dtc(city)
    landuse_encoded = get_land_use_data_encoded(city)
    digital_elevation_data = get_digital_elevation_data(city, time.year)
    building_fraction_data = get_building_fraction_data(city)
    # soil_sealing_data = get_soil_sealing_data(city)
    # time_of_day = build_time_of_day(time, use_day_night_model)

    _, H, W = ndvi_data.shape

    def convert_to_4d(arr):
        return arr.values[..., np.newaxis]

    def convert_scalar_to_4d(scalar, h, w):
        # Go from a single number to a (1, H, W, 1) array
        return np.full((1, h, w, 1), scalar)

    X = np.concatenate(
        (
            # SBN = should be normalized
            convert_to_4d(distance_coast_data),  # SBN
            convert_to_4d(digital_elevation_data),  # SBN
            convert_to_4d(building_fraction_data),
            # convert_to_4d(soil_sealing_data),
            convert_scalar_to_4d(meteo_data.T2M, h=H, w=W),  # SBN
            convert_scalar_to_4d(meteo_data.Q2M, h=H, w=W),  # SBN
            convert_scalar_to_4d(meteo_data.WIND_SPEED, h=H, w=W),  # SBN
            convert_scalar_to_4d(meteo_data.WIND_DIR_SIN, h=H, w=W),
            convert_scalar_to_4d(meteo_data.WIND_DIR_COS, h=H, w=W),
            convert_scalar_to_4d(meteo_data.CSF, h=H, w=W),
            convert_scalar_to_4d(meteo_data.Rs, h=H, w=W),
            ndvi_data_for_time[np.newaxis, ..., np.newaxis],
            landuse_encoded[np.newaxis, ...],
        ),
        axis=-1,
    )

    if output_var == "temperature":
        y = convert_to_4d(urbclim_data.T2M)
        # humidity is excluded as a predictor in this case
        X = np.delete(X, 4, axis=-1)
    elif output_var == "wbgt":
        y = convert_to_4d(urbclim_data.WBGT)
    elif output_var == "humidity":
        y = convert_to_4d(urbclim_data.QV2M)
    else:
        raise ValueError(f"Unsupported output_var: {output_var}")

    # Cut off sides
    X = X[:, 10:-10, 10:-10, :]
    y = y[:, 10:-10, 10:-10]

    # Generate random starting index for patches
    h = rng.integers(0, H - 20 - patch_size + 1)
    w = rng.integers(0, W - 20 - patch_size + 1)

    X_patch = X[:, h : h + patch_size, w : w + patch_size, :]
    y_patch = y[:, h : h + patch_size, w : w + patch_size, :]

    return X_patch[0], y_patch[0]


def get_time_index(time_var, target_time):
    times = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    time_idx = np.where(np.array(times) == target_time)[0]
    if len(time_idx) == 0:
        raise ValueError(f"Time {target_time} not found in file.")
    return time_idx[0]


def get_time_indices(time_var, target_times):
    single = not isinstance(target_times, (list, tuple, np.ndarray))
    if single:
        target_times = [target_times]
    times = num2date(time_var[:], units=time_var.units, calendar=time_var.calendar)
    times = np.array(times)
    indices = []
    for t in target_times:
        idx = np.where(times == t)[0]
        if len(idx) == 0:
            raise ValueError(f"Time {t} not found in file.")
        indices.append(idx[0])
    return indices[0] if single else indices


def load_target_era5(var_name, time, target_domain, input_folder):
    file_name = f"{target_domain}_100m_{var_name}_{time.year}{str(time.month).zfill(2)}{str(time.day).zfill(2)}.nc"
    nc_path = input_folder / "ERA5_samples" / file_name
    with NetCDFDataset(nc_path, "r") as nc:
        idx = get_time_indices(nc.variables["time"], time)
        return np.flipud(nc.variables[var_name][idx, :, :].filled(0))


def load_target_era5_list(var_name, times, target_domain, input_folder):
    # Group requested times by (year, month, day) so we open each file once
    by_day = defaultdict(list)  # (y,m,d) -> list of (out_idx, datetime)
    for out_idx, t in enumerate(times):
        by_day[(t.year, t.month, t.day)].append((out_idx, t))

    results = [None] * len(times)
    for (y, m, d), items in by_day.items():
        sample_time = items[0][1]
        file_name = f"{target_domain}_100m_{var_name}_{sample_time.year}{str(sample_time.month).zfill(2)}{str(sample_time.day).zfill(2)}.nc"
        nc_path = input_folder / "ERA5_samples" / file_name

        with NetCDFDataset(nc_path, "r") as nc:
            tvar = nc.variables["time"]
            req_times = [t for _, t in items]
            idxs = get_time_indices(tvar, req_times)
            # Read each requested timestep and place it back in the correct position
            for (out_idx, _), i in zip(items, idxs):
                arr = nc.variables[var_name][i, :, :].filled(0)
                results[out_idx] = np.flipud(arr)
    return np.stack(results, axis=0)


def load_target_era5_list_singlefile(var_name, times, input_folder):
    file_name = f"{var_name}_100m.nc"
    nc_path = f"{input_folder}/{var_name}/{file_name}"

    with NetCDFDataset(nc_path, "r") as nc:
        tvar = nc.variables["time"]
        idxs = get_time_indices(tvar, times)  # list/np.array of length len(times)

        # Read all requested timesteps in one go: (n_times, y, x)
        data = nc.variables[var_name][idxs, :, :].filled(0)

    # Flip along y for every timestep (equivalent to np.flipud per slice)
    data = np.flip(data, axis=1)
    return data


def safe_read(src, replacement_value=np.nan):
    data = src.read(1)
    nodata = src.meta.get("nodata", None)
    if nodata is not None:
        data = np.where(data == nodata, replacement_value, data)
    else:
        data = np.where(np.isnan(data), replacement_value, data)
    return data


def load_target_ndvi(time, input_folder):
    # Infer the month from the time (%b)
    month = time.strftime("%b").lower()
    file_name = f"ndvi_{month}_100m.tif"
    tif_path = input_folder / "ndvi" / file_name
    with rasterio.open(tif_path) as src:
        return safe_read(src, replacement_value=0.0)


def load_target_dem(input_folder, target_domain):
    tif_path = input_folder / "DEM" / "DEM.tif"
    with rasterio.open(tif_path) as src:
        DEM = safe_read(src, replacement_value=0.0)
    ERA5_height_path = input_folder / "ERA5_samples" / f"{target_domain}_100m_z.nc"
    ERA5_height = xr.open_dataset(ERA5_height_path).z.values[0, :, :]
    ERA5_height = np.flipud(ERA5_height)  # Flip the height data to match DEM orientation
    ERA5_height = uniform_filter(ERA5_height, size=250)
    ERA5_height = uniform_filter(ERA5_height, size=250)
    return DEM - ERA5_height


def get_metadata(input_folder):
    tif_path = input_folder / "buildings" / "buildings.tif"
    with rasterio.open(tif_path) as src:
        meta = src.meta.copy()
        cols, rows = np.meshgrid(np.arange(meta["width"]), np.arange(meta["height"]))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons = np.array(xs)
        lats = np.array(ys)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon_deg, lat_deg = transformer.transform(lons, lats)
        lat_grid = lat_deg.reshape(rows.shape[0], rows.shape[1])
        lon_grid = lon_deg.reshape(rows.shape[0], rows.shape[1])
    return meta, lat_grid, lon_grid


def load_target_building_fraction(input_folder):
    tif_path = input_folder / "buildings" / "buildings.tif"
    with rasterio.open(tif_path) as src:
        # The building fraction data is in percentage, so we divide by 100 to get a fraction
        return safe_read(src, replacement_value=0.0) / 100


# def load_uae_soil_sealing():
#     tif_path = UAE_INPUT_FOLDER_PREFIX / "sealing" / "sealing.tif"
#     with rasterio.open(tif_path) as src:
#         return safe_read(src, replacement_value=0.0)


def load_target_land_use(input_folder, target_domain):
    tif_path = input_folder / "land_use" / f"{target_domain}_100m_EPSG3857_ready.tif"
    with rasterio.open(tif_path) as src:
        land_use_data = safe_read(src, replacement_value=200).astype(int)
        # remove the solar panels class (not well represented)
        land_use_data[land_use_data == 53] = 51
        # TODO MERGE THIS WITH RIOXARRAY VERSION TO AVOID DUPLICATION
        # WATCH OUT WITH DIFFERENT ORDERING OF DIMENSIONS
        # One-hot encode the categorical data so the channels are added in the last dimension
        # Dimensions will become (height, width, channels)
        landuse_classes = [10, 20, 30, 40, 50, 51, 52, 60, 80, 90, 95, 200]
        encoder = OneHotEncoder(
            sparse_output=False,
            categories=[landuse_classes],
            handle_unknown="ignore",
        )
        landuse_encoded = encoder.fit_transform(land_use_data.flatten().reshape(-1, 1))
        landuse_encoded = landuse_encoded.reshape(land_use_data.shape[0], land_use_data.shape[1], -1)
        return np.transpose(landuse_encoded, (2, 0, 1))


def load_target_dtc(input_folder):
    tif_path = input_folder / "distance_to_coast" / "full_domain_EPSG3857.tif"
    with rasterio.open(tif_path) as src:
        data = safe_read(src)
        data = gaussian_filter(data, sigma=4)
        return data


def convert_scalar_to_3d(scalar, h, w):
    return np.full((1, h, w), scalar, dtype=np.float32)


def prepend_dim(arr):
    return arr[np.newaxis, ...]


def get_inference_input(time, use_day_night_model, target_domain):
    # TODO Are timestamps being handled properly?
    # What is the input time zone?
    # What timezone are the files?
    # Shift wrt UTC needed?
    if target_domain == "UAE":
        INPUT_FOLDER_PREFIX = UAE_INPUT_FOLDER_PREFIX
    elif target_domain == "AD":
        INPUT_FOLDER_PREFIX = AD_INPUT_FOLDER_PREFIX
    else:
        raise ValueError(f"Unsupported target_domain: {target_domain}")

    logger.info("Constructing inference input data for all of UAE")
    era5_t = load_target_era5(
        "t", time=time, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
    )
    era5_u = load_target_era5(
        "u", time=time, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
    )
    era5_v = load_target_era5(
        "v", time=time, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
    )
    era5_q = load_target_era5(
        "q", time=time, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
    )
    era5_ssrd = load_target_era5(
        "ssrd", time=time, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
    )

    # calculate wind speed and direction
    era5_wind_speed = np.sqrt(era5_u**2 + era5_v**2)
    era5_wind_dir_sin = np.sin(np.arctan2(era5_v, era5_u))
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250)
    era5_wind_dir_cos = np.cos(np.arctan2(era5_v, era5_u))
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250)

    # smooth out the gridded boxy ERA-5 data (t & ssrd two times, wind three times)
    era5_t = uniform_filter(era5_t, size=250)
    era5_t = uniform_filter(era5_t, size=250)
    era5_q = uniform_filter(era5_q, size=250)
    era5_q = uniform_filter(era5_q, size=250)
    era5_ssrd = uniform_filter(era5_ssrd, size=250)
    era5_ssrd = uniform_filter(era5_ssrd, size=250)
    era5_wind_speed = uniform_filter(era5_wind_speed, size=250)
    era5_wind_speed = uniform_filter(era5_wind_speed, size=250)
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250)
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250)
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250)
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250)
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250)
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250)

    # get all the other input data
    ndvi = load_target_ndvi(time, input_folder=INPUT_FOLDER_PREFIX)
    dtc_data = load_target_dtc(input_folder=INPUT_FOLDER_PREFIX)
    digital_elevation_data = load_target_dem(
        input_folder=INPUT_FOLDER_PREFIX, target_domain=target_domain
    )
    building_fraction_data = load_target_building_fraction(input_folder=INPUT_FOLDER_PREFIX)
    # soil_sealing_data = load_uae_soil_sealing()
    landuse_encoded = load_target_land_use(
        input_folder=INPUT_FOLDER_PREFIX, target_domain=target_domain
    )
    # time_of_day = build_time_of_day(time, use_day_night_model)
    Rs = era5_ssrd / 1364  # normalize to maximum solar constant
    meta, lat_grid, lon_grid = get_metadata(input_folder=INPUT_FOLDER_PREFIX)

    # get spatially explicit clear sky radiation
    # we only take a subset, as calculating for each location individually is
    # extremely slow
    unique_lats = np.linspace(np.min(lat_grid), np.max(lat_grid), 200)
    ave_lon = np.mean(lon_grid)
    offset = timezone(timedelta(minutes=30))
    dt_with_tz = time.replace(tzinfo=offset)
    ghi_vals = np.zeros_like(unique_lats)
    for idx, lat in enumerate(unique_lats):
        location = Location(lat, ave_lon, tz="UTC", altitude=0)
        # pvlib assumes timestamps are localized — we pass ours with offset explicitly
        clear_sky = location.get_clearsky(dt_with_tz, model="simplified_solis")
        clear_sky.index = clear_sky.index.tz_convert(offset)
        ghi_vals[idx] = clear_sky.iloc[0].ghi

    interp_func = interp1d(unique_lats, ghi_vals, bounds_error=False, fill_value="extrapolate")
    cs_radiation = interp_func(lat_grid)
    cs_frac = era5_ssrd / cs_radiation
    cs_frac[cs_frac > 1] = 1
    cs_frac[cs_frac < 0] = 0
    cs_frac[np.isnan(cs_frac)] = 1

    # NOTE: it's important that this is in the same order as in the training data
    X = np.concatenate(
        (
            prepend_dim(dtc_data),
            prepend_dim(digital_elevation_data),
            prepend_dim(building_fraction_data),
            # prepend_dim(soil_sealing_data),
            prepend_dim(era5_t),
            prepend_dim(era5_q),
            prepend_dim(era5_wind_speed),
            prepend_dim(era5_wind_dir_sin),
            prepend_dim(era5_wind_dir_cos),
            prepend_dim(cs_frac),
            # convert_scalar_to_3d(time_of_day, h=ndvi.shape[0], w=ndvi.shape[1]),
            prepend_dim(Rs),
            prepend_dim(ndvi),
            landuse_encoded,
        ),
        axis=0,
    )

    return X, meta


def get_production_input(times, target_domain, operational_forecast_folder):
    if target_domain == "UAE":
        INPUT_FOLDER_PREFIX = UAE_INPUT_FOLDER_PREFIX
    elif target_domain == "AD":
        INPUT_FOLDER_PREFIX = AD_INPUT_FOLDER_PREFIX
    else:
        raise ValueError(f"Unsupported target_domain: {target_domain}")

    logger.info("Constructing inference input data for all of UAE")
    if "ECMWF" in operational_forecast_folder or "ICON" in operational_forecast_folder:
        era5_t = load_target_era5_list_singlefile(
            "2t", times=times, input_folder=operational_forecast_folder
        )
        era5_u = load_target_era5_list_singlefile(
            "10u", times=times, input_folder=operational_forecast_folder
        )
        era5_v = load_target_era5_list_singlefile(
            "10v", times=times, input_folder=operational_forecast_folder
        )
        era5_q = load_target_era5_list_singlefile(
            "q2m", times=times, input_folder=operational_forecast_folder
        )
        era5_ssrd = load_target_era5_list_singlefile(
            "ssrd", times=times, input_folder=operational_forecast_folder
        )
    else:
        era5_t = load_target_era5_list(
            "t", times=times, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
        )
        era5_u = load_target_era5_list(
            "u", times=times, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
        )
        era5_v = load_target_era5_list(
            "v", times=times, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
        )
        era5_q = load_target_era5_list(
            "q", times=times, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
        )
        era5_ssrd = load_target_era5_list(
            "ssrd", times=times, target_domain=target_domain, input_folder=INPUT_FOLDER_PREFIX
        )

    # calculate wind speed and direction
    era5_wind_speed = np.sqrt(era5_u**2 + era5_v**2)
    era5_wind_dir_sin = np.sin(np.arctan2(era5_v, era5_u))
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250, axes=(1, 2))
    era5_wind_dir_cos = np.cos(np.arctan2(era5_v, era5_u))
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250, axes=(1, 2))

    # smooth out the gridded boxy ERA-5 data (t & ssrd two times, wind three times)
    era5_t = uniform_filter(era5_t, size=250, axes=(1, 2))
    era5_t = uniform_filter(era5_t, size=250, axes=(1, 2))
    era5_q = uniform_filter(era5_q, size=250, axes=(1, 2))
    era5_q = uniform_filter(era5_q, size=250, axes=(1, 2))
    era5_ssrd = uniform_filter(era5_ssrd, size=250, axes=(1, 2))
    era5_ssrd = uniform_filter(era5_ssrd, size=250, axes=(1, 2))
    era5_wind_speed = uniform_filter(era5_wind_speed, size=250, axes=(1, 2))
    era5_wind_speed = uniform_filter(era5_wind_speed, size=250, axes=(1, 2))
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250, axes=(1, 2))
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250, axes=(1, 2))
    era5_wind_dir_sin = uniform_filter(era5_wind_dir_sin, size=250, axes=(1, 2))
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250, axes=(1, 2))
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250, axes=(1, 2))
    era5_wind_dir_cos = uniform_filter(era5_wind_dir_cos, size=250, axes=(1, 2))

    # get all the other input data
    months = np.unique([m.month for m in times])
    ndvi = {}
    for month in months:
        time = datetime(2024, month, 15, 11, 0, tzinfo=timezone.utc)
        ndvi[month] = load_target_ndvi(time, input_folder=INPUT_FOLDER_PREFIX)
    dtc_data = load_target_dtc(input_folder=INPUT_FOLDER_PREFIX)
    digital_elevation_data = load_target_dem(
        input_folder=INPUT_FOLDER_PREFIX, target_domain=target_domain
    )
    building_fraction_data = load_target_building_fraction(input_folder=INPUT_FOLDER_PREFIX)
    # soil_sealing_data = load_uae_soil_sealing()
    landuse_encoded = load_target_land_use(
        input_folder=INPUT_FOLDER_PREFIX, target_domain=target_domain
    )
    # time_of_day = build_time_of_day(time, use_day_night_model)
    Rs = era5_ssrd / 1364  # normalize to maximum solar constant
    meta, lat_grid, lon_grid = get_metadata(input_folder=INPUT_FOLDER_PREFIX)

    # get spatially explicit clear sky radiation
    # we only take a subset, as calculating for each location individually is
    # extremely slow
    unique_lats = np.linspace(np.min(lat_grid), np.max(lat_grid), 200)
    ave_lon = np.mean(lon_grid)
    offset = timezone(timedelta(minutes=30))
    dt_with_tz = [time.replace(tzinfo=offset) for time in times]
    ghi_vals = np.zeros((len(unique_lats), len(times)))
    cs_radiation = np.zeros_like(era5_ssrd)
    for idy, t in enumerate(dt_with_tz):
        for idx, lat in enumerate(unique_lats):
            location = Location(lat, ave_lon, tz="UTC", altitude=0)
            # pvlib assumes timestamps are localized — we pass ours with offset explicitly
            clear_sky = location.get_clearsky(t, model="simplified_solis")
            clear_sky.index = clear_sky.index.tz_convert(offset)
            ghi_vals[idx, idy] = clear_sky.iloc[0].ghi

        interp_func = interp1d(
            unique_lats, ghi_vals[:, idy], bounds_error=False, fill_value="extrapolate"
        )
        cs_radiation[idy, :, :] = interp_func(lat_grid)
    cs_frac = era5_ssrd / cs_radiation
    cs_frac[cs_frac > 1] = 1
    cs_frac[cs_frac < 0] = 0
    cs_frac[np.isnan(cs_frac)] = 1

    return (
        dtc_data,
        digital_elevation_data,
        building_fraction_data,
        era5_t,
        era5_q,
        era5_wind_speed,
        era5_wind_dir_sin,
        era5_wind_dir_cos,
        cs_frac,
        Rs,
        ndvi,
        landuse_encoded,
        meta,
    )


@lru_cache()
def calculate_mean_std(channel_name) -> Tuple[np.float32, np.float32]:
    logger.info(f"Calculating mean and std for {channel_name}")
    match channel_name:
        case "distance_to_coast":
            data = load_target_dtc(input_folder=UAE_INPUT_FOLDER_PREFIX)
            return np.mean(data), np.std(data)
        case "digital_elevation":
            data = load_target_dem(input_folder=UAE_INPUT_FOLDER_PREFIX, target_domain="UAE")
            return np.mean(data), np.std(data)
        case "temperature":
            t2m_stats_filename = UAE_INPUT_FOLDER_PREFIX / "ERA5_statistics" / "T2M_UAE.nc"
            ds = xr.open_dataset(t2m_stats_filename, chunks={"t": 100})
            mean = np.float32(ds["t"].mean().compute().values)
            std = np.float32(ds["t"].std().compute().values)
            return mean, std
        case "humidity":
            q2m_stats_filename = UAE_INPUT_FOLDER_PREFIX / "ERA5_statistics" / "QV2M_UAE.nc"
            ds = xr.open_dataset(q2m_stats_filename, chunks={"t": 100})
            mean = np.float32(ds["q"].mean().compute().values)
            std = np.float32(ds["q"].std().compute().values)
            return mean, std
        case "wind_speed":
            ws_stats_filename = UAE_INPUT_FOLDER_PREFIX / "ERA5_statistics" / "WS_UAE.nc"
            ds = xr.open_dataset(ws_stats_filename, chunks={"t": 100})
            mean = np.float32(ds["u"].mean().compute().values)
            std = np.float32(ds["u"].std().compute().values)
            return mean, std
        case "wbgt":
            # not possible to calculate for whole UAE so we use the cities
            means = []
            stds = []
            for city in UAE_CITIES:
                ds = load_urbclim(city, year=2019)
                means.append(ds.WBGT.mean().values)
                stds.append(ds.WBGT.std().values)
            mean = np.float32(np.mean(means))
            var = np.mean(np.float32(stds) ** 2 + (np.float32(means) - mean) ** 2)
            std = np.float32(np.sqrt(var))
            return mean, std
        case _:
            raise NotImplementedError(f"Normalization for {channel_name} is not implemented.")


def compute_normalization(
    channel_names: List[str] = [
        "distance_to_coast",
        "digital_elevation",
        "temperature",
        "humiditywind_speed",
    ],
):
    norm_data = {}
    for channel_name in channel_names:
        mean, std = calculate_mean_std(channel_name)
        norm_data[channel_name] = {
            "mean": mean.item(),
            "std": std.item(),
        }

    return norm_data


def load_normalization(
    x_channels: Dict[int, str] = NORMALIZATION_CHANNEL_SPEC_TEMPERATURE["x_channels"],
    y_channels: Dict[int, str] = NORMALIZATION_CHANNEL_SPEC_TEMPERATURE["y_channels"],
    load_from_file: bool = True,
):
    all_required_channels = set(x_channels.values()).union(set(y_channels.values()))
    norm_data_filepath = OUTPUT_FOLDER_ROOT / "normalization_data.json"
    legacy_norm_data_filepath = UAE_INPUT_FOLDER_PREFIX / "normalization_data.json"
    norm_data_load_filepath = norm_data_filepath
    if not norm_data_load_filepath.is_file() and legacy_norm_data_filepath.is_file():
        norm_data_load_filepath = legacy_norm_data_filepath
    norm_data = None

    if load_from_file and norm_data_load_filepath.is_file():
        logger.info(f"Loading normalization data from {norm_data_load_filepath}")
        with open(norm_data_load_filepath, "r") as f:
            norm_data = json.load(f)
        missing_channels = set(all_required_channels) - set(norm_data.keys())
        if missing_channels:
            logger.warning(f"Normalization data is missing some channels: {missing_channels}.")
            norm_data = None

    if norm_data is None:
        logger.warning(f"Computing normalization for {all_required_channels}...")
        norm_data = compute_normalization(channel_names=list(all_required_channels))
        logger.info(f"Storing normalization data to {norm_data_filepath} for later reuse.")
        norm_data_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_data_filepath, "w") as f:
            json.dump(norm_data, f)

    # Convert to structure used throughout the codebase
    return {
        "x_means": {ch: norm_data[name]["mean"] for ch, name in x_channels.items()},
        "x_stds": {ch: norm_data[name]["std"] for ch, name in x_channels.items()},
        "y_means": {ch: norm_data[name]["mean"] for ch, name in y_channels.items()},
        "y_stds": {ch: norm_data[name]["std"] for ch, name in y_channels.items()},
    }


class JITDataset(TorchDataset):
    def __init__(
        self,
        cities,
        patch_size,
        n_circ_class_clusters,
        circ_class_cluster_idx,
        use_individual_model_per_cluster,
        use_day_night_model,
        time_of_the_day,
        start_year,
        end_year,
        output_var,
        nb_patches,
        norm_data,
        rng_seed=42,
    ):
        dataset_index = build_dataset_index(
            cities=cities,
            patch_size=patch_size,
            n_circ_class_clusters=n_circ_class_clusters,
            circ_class_cluster_idx=circ_class_cluster_idx,
            use_individual_model_per_cluster=use_individual_model_per_cluster,
            use_day_night_model=use_day_night_model,
            time_of_the_day=time_of_the_day,
            start_year=start_year,
            end_year=end_year,
        )
        total_available_patches = len(dataset_index)
        if nb_patches > total_available_patches:
            logger.warning(
                f"Requested number of patches ({nb_patches}) is larger than the available ({total_available_patches}). Using {total_available_patches} patches instead."
            )
            self.nb_patches = total_available_patches
        else:
            self.nb_patches = nb_patches
        self.patch_size = patch_size
        self.use_day_night_model = use_day_night_model
        self.rng_seed = rng_seed
        self.norm_data = norm_data
        self.output_var = output_var

        self.sampled_index = dataset_index.sample(
            n=self.nb_patches, random_state=rng_seed
        ).reset_index(drop=True)

    def __len__(self):
        return self.nb_patches

    def __getitem__(self, idx):
        # Set random number generator so timestep is sampled same way
        rng = np.random.default_rng(self.rng_seed + idx)

        time, city = self.sampled_index.iloc[idx]
        x, y = get_patch(
            time=time,
            city=city,
            patch_size=self.patch_size,
            use_day_night_model=self.use_day_night_model,
            output_var=self.output_var,
            rng=rng,
        )

        for ch, mean in self.norm_data["x_means"].items():
            x[:, :, ch] = (x[:, :, ch] - mean) / self.norm_data["x_stds"][ch]

        for ch, mean in self.norm_data["y_means"].items():
            y[:, :, ch] = (y[:, :, ch] - mean) / self.norm_data["y_stds"][ch]

        # HCW -> CHW
        if self.output_var == "temperature":
            assert x.shape == (
                self.patch_size,
                self.patch_size,
                NUMBER_OF_PREDICTOR_CHANNELS_TEMPERATURE,
            )
        elif self.output_var == "wbgt":
            assert x.shape == (self.patch_size, self.patch_size, NUMBER_OF_PREDICTOR_CHANNELS_WBGT)
        elif self.output_var == "humidity":
            assert x.shape == (
                self.patch_size,
                self.patch_size,
                NUMBER_OF_PREDICTOR_CHANNELS_HUMIDITY,
            )
        else:
            raise ValueError(f"Unsupported output_var: {self.output_var}")
        X_torch = torch.from_numpy(x).float().permute(2, 0, 1)
        y_torch = torch.from_numpy(y).float().permute(2, 0, 1)

        return X_torch, y_torch
