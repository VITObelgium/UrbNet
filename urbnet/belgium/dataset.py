# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import calendar
import math
from functools import lru_cache
from typing import List, Literal

import numpy as np
import pandas as pd
import rasterio
import rioxarray as rioxr
import torch
import xarray as xr
from loguru import logger
from scipy.ndimage import uniform_filter
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset

from urbnet.belgium.settings import (
    INFERENCE_INPUT_FOLDER_PREFIX,
    METEO_VAR_NAMES,
    NUMBER_OF_TARGET_CHANNELS,
    OVERALL_FOLDER_PREFIX,
    get_number_of_predictor_channels,
)
from urbnet.visualize import visualize_bands

# Step 1: Define the CORINE to UrbClim mapping based on your table
CORINE_TO_URBCLIM = {
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
    6: 3,
    10: 4,
    11: 4,
    34: 5,
    8: 6,
    9: 6,
    30: 6,
    31: 6,
    38: 6,
    39: 6,
    7: 7,
    18: 7,
    26: 7,
    27: 7,
    32: 7,
    35: 7,
    36: 7,
    37: 7,
    12: 8,
    13: 8,
    14: 8,
    19: 8,
    20: 8,
    21: 8,
    15: 9,
    28: 9,
    29: 9,
    33: 9,
    16: 10,
    17: 10,
    22: 10,
    23: 11,
    25: 11,
    24: 12,
    40: 13,
    41: 14,
    42: 15,
    43: 15,
    44: 15,
}


def take_along_dim(var, interval_idx, dim="interval"):
    if dim in var.dims:
        interval_axis = var.dims.index(dim)
        # 2. Expand interval_idx along interval axis (insert axis)
        gather_idx = np.expand_dims(interval_idx.data, axis=interval_axis)

        # 3. Now take along that axis
        gathered = np.take_along_axis(var.data, gather_idx, axis=interval_axis).squeeze(
            axis=interval_axis
        )

        # New dims: remove 'interval' from dims
        new_dims = list(var.dims)
        new_dims.pop(interval_axis)

        # Create result DataArray
        ds_out_day = xr.DataArray(
            gathered,
            dims=new_dims,
            coords={dim: var.coords[dim] for dim in new_dims if dim in var.coords},
            attrs=var.attrs,
        )
    else:
        # Pass through non-interval-dependent variables
        ds_out_day = var
    return ds_out_day


def shape_for_region(region):
    """
    Get the shape of the NDVI data for a given region.
    This is used to determine the height and width of the patches.
    """
    ndvi_for_region = load_ndvi(region, month=1)
    _, H, W = ndvi_for_region.shape
    return H, W


def build_dataset_index(
    regions: List[str],
    start_year,
    end_year,
    patch_size,
    rng_seed,
):
    """
    Build full 'index' of all samples that can be drawn
    For every time, for every region, we may have a different number of patches per region?
    """
    rng = np.random.default_rng(rng_seed)
    times = pd.date_range(
        start=f"{start_year}-01-01",
        end=f"{end_year}-12-31",
        freq="1d",
        tz="UTC",
    )
    times = times[times.day_of_year != 1]
    times_list = times.to_list()

    def build_rows_for_time_region(time, region):
        H, W = shape_for_region(region)
        # NOTE: the -20 here is because we're cutting off 10 pixels on each side in get_patch
        nb_patches = math.ceil((H - 20) / patch_size) * math.ceil((W - 20) / patch_size)
        py = rng.integers(0, H - 20 - patch_size + 1, nb_patches)
        px = rng.integers(0, W - 20 - patch_size + 1, nb_patches)
        return pd.DataFrame(
            {
                "time": np.repeat(time, nb_patches),
                "region": np.repeat(region, nb_patches),
                "H": np.repeat(H, nb_patches),
                "W": np.repeat(W, nb_patches),
                "py": py,
                "px": px,
            }
        )

    # Concatenate the list of DataFrames into a single DataFrame
    return pd.concat(
        [build_rows_for_time_region(time, region) for time in times_list for region in regions],
        ignore_index=True,
    )


@lru_cache()
def load_era5_year(region, region_name, year):
    meteo_file = (
        OVERALL_FOLDER_PREFIX / "urbsim_runs" / region / "meteo" / f"{region_name}_meteo_{year}.nc"
    )
    ds_meteo = xr.open_dataset(meteo_file)
    # NOTE: ERA5 is in UTC, so in principle we need to shift to Belgian time zone
    # However, minimum temperature happens before sunset, so we can skip time zone alignment.

    ds = xr.Dataset(
        data_vars={
            "T2M": ("time", ds_meteo.T[:, 0].data),
            "U10M": ("time", ds_meteo.U[:, 0].data),
            "V10M": ("time", ds_meteo.V[:, 0].data),
            "Rs": ("time", ds_meteo.Rs[:].data),
            # "CSF": ("time", ds_meteo.Rs[:].data),
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

    return ds

def process_era5_day_and_previous_day(ds_select_day, ds_select_previous_day, steps_per_day):
    """
    function expects, dataarrays with variables 'T2M',"U10M", "V10M",and "Rs" in ds_select_day and "Rs" in ds_select_previous_day

    """
    import numpy as np
    import xarray as xr

    ds_select_day["WIND_SPEED"] = np.sqrt(ds_select_day.U10M**2 + ds_select_day.V10M**2)
    ds_select_day["WIND_DIR"] = np.arctan2(ds_select_day.U10M, ds_select_day.V10M)
    ds_select_day["WIND_DIR_SIN"] = np.sin(ds_select_day.WIND_DIR)
    ds_select_day["WIND_DIR_COS"] = np.cos(ds_select_day.WIND_DIR)
    ds_select_day_reshaped = reshape_into_groups(ds_select_day, "time", steps_per_day)
    ds_select_day_reshaped = ds_select_day_reshaped.assign_coords(
        {"time": ds_select_day_reshaped.time.astype("datetime64[D]").astype("datetime64[ns]")}
    )

    ds_select_previous_day_reshaped = reshape_into_groups(
        ds_select_previous_day, "time", steps_per_day
    )
    #
    ds_select_previous_day_reshaped = ds_select_previous_day_reshaped.assign_coords(
        {
            "time": ds_select_previous_day_reshaped.time.astype("datetime64[D]").astype(
                "datetime64[ns]"
            )
        }
    )
    T2M_isnull = ds_select_day_reshaped["T2M"].mean("period").isnull()

    ds_select_day_reshaped_T2M_for_idx = xr.where(T2M_isnull, 0, ds_select_day_reshaped["T2M"])
    idx_at_Tmin = ds_select_day_reshaped_T2M_for_idx.argmin(dim="period")
    idx_at_Tmax = ds_select_day_reshaped_T2M_for_idx.argmax(dim="period")

    # get wind values at Tmin and at Tmax
    ds_out_day = xr.Dataset()
    for var_name in [
        "WIND_SPEED",
        "WIND_DIR_SIN",
        "WIND_DIR_COS",
        "T2M",
    ]:  # reshaped_ds.data_vars.items():
        var = ds_select_day_reshaped[var_name]

        ds_out_day[var_name + "_at_Tmin"] = take_along_dim(var, idx_at_Tmin, "period")
        ds_out_day[var_name + "_at_Tmax"] = take_along_dim(var, idx_at_Tmax, "period")

    max_rs_cs = 1364  # TODO: range may need to be revised for mean radiation

    for var_name in ["WIND_SPEED", "WIND_DIR_SIN", "WIND_DIR_COS", "T2M"]:
        ds_out_day[var_name + "_mean"] = ds_select_day_reshaped[var_name].mean(dim="period")

    ds_out_day["Rs_mean"] = ds_select_day_reshaped["Rs"].mean(dim="period") / max_rs_cs

    # radiation from the previous day. This is needed because Tmin is affected by the radiative effects from the previous day
    ds_out_day["Rs_mean_prev_day"] = xr.zeros_like(ds_out_day["Rs_mean"])
    # we overwrite the values, but we keep the time stamp
    ds_out_day["Rs_mean_prev_day"].values = (
        ds_select_previous_day_reshaped["Rs"].mean(dim="period") / max_rs_cs
    ).values

    return ds_out_day


def load_era5_day(region, region_name, time):
    ds_full_year = load_era5_year(region, region_name, year=time.year)

    # we select particular day and the previous day
    ds_select_day = ds_full_year.isel(time=ds_full_year.time.dt.dayofyear == time.day_of_year)
    ds_select_previous_day = ds_full_year.isel(
        time=ds_full_year.time.dt.dayofyear == (time.day_of_year - 1)
    )

    return process_era5_day_and_previous_day(ds_select_day, ds_select_previous_day, steps_per_day=72)


@lru_cache
def load_ndvi(region, month):
    month_short_str = str(calendar.month_abbr[month]).lower()
    ndvi_filename = (
        f"{OVERALL_FOLDER_PREFIX}/urbsim_runs/{region}/terrain/log/temp_veg_{month_short_str}.tif"
    )
    ndvi_darr = rioxr.open_rasterio(ndvi_filename)
    assert isinstance(ndvi_darr, xr.DataArray)
    ndvi = ndvi_darr.values
    if np.isnan(ndvi).any():
        logger.warning(
            f"NaN found in ndvi for {region} in month {month_short_str}, replacing with mean."
        )
        ndvi = np.where(np.isnan(ndvi), np.nanmean(ndvi), ndvi)
    ndvi[ndvi > 1] = 1
    ndvi[ndvi < -1] = -1
    return ndvi


@lru_cache()
def load_urbclim_min_temp_for_year_v2(region, region_name, year):
    urbclim_filename = (
        f"{OVERALL_FOLDER_PREFIX}/urbsim_runs/{region}/urbclim/{year}/{region_name}_output.nc"
    )
    ds = xr.open_dataset(urbclim_filename, chunks={"t": 240})
    ds_flipped = ds.reindex(y=list(reversed(ds.y)))
    floored_dates = ds_flipped.t.dt.floor("D")
    tmin = ds_flipped.T2M.groupby(floored_dates).min("t")
    tmin = tmin.rename({"floor": "time"})
    return tmin


def load_urbclim_min_temp_for_time_v2(region, region_name, time):
    tmin = load_urbclim_min_temp_for_year_v2(region, region_name, time.year)
    # Ensure the time is a pandas.Timestamp with no time component (date only)
    return tmin.sel(time=time.normalize().to_datetime64()).values


# This cache is separate for every training sample, so grows fast!
# NOTE this caching breaks the training for some reason, with y values that are out of whack.
# @lru_cache(maxsize=2000)
def load_urbclim_for_time_v1(region, region_name, time, target_var):
    urbclim_filename = (
        f"{OVERALL_FOLDER_PREFIX}/urbsim_runs/{region}/urbclim/{time.year}/{region_name}_output.nc"
    )
    ds = xr.open_dataset(urbclim_filename, chunks={"t": 240})
    # urbclim data is flipped along the y-axis
    ds_flipped = ds.reindex(y=list(reversed(ds.y)))
    # NOTE: UrbClim is in UTC, so in principle we need to shift to Belgian time zone
    # However, minimum temperature happens before sunset, so we can skip time zone alignment.
    ds_for_day = ds_flipped.sel(
        t=ds_flipped.t.dt.floor("D") == time.normalize().to_datetime64()
    ).rename({"t": "time"})

    # align time dimensions of era5 with urbclim
    ds_era5_select_day_hourly = ds_era5_select_day_hourly.assign_coords(
        {
            "time": ds_era5_select_day_hourly.time.astype("datetime64[h]").values.astype(
                ds_for_day.time.dtype
            )
        }
    )

    match target_var:
        case "t_min":
            return ds_for_day.T2M.min(dim="time").values
        case "t_max":
            return ds_for_day.T2M.max(dim="time").values
        case "t_mean":
            return ds_for_day.T2M.mean(dim="time").values


@lru_cache()
def get_land_use_data(region):
    if region == "Belgium":
        tif_path = (
            OVERALL_FOLDER_PREFIX
            / "input_data"
            / "urbsim-config-belgium"
            / "terrain"
            / "log"
            / "temp_ua.tif"
        )
    else:
        tif_path = OVERALL_FOLDER_PREFIX / "urbsim_runs" / region / "terrain" / "log" / "temp_ua.tif"
    arr = rioxr.open_rasterio(tif_path)
    assert isinstance(arr, xr.DataArray)
    return arr.astype(int).compute()


@lru_cache()
def get_land_use_data_encoded(region):
    corine_data = get_land_use_data(region)

    urbclim_landuse_data = np.zeros_like(corine_data, dtype=np.uint8)
    for corine_val, urbclim_val in CORINE_TO_URBCLIM.items():
        urbclim_landuse_data[corine_data == corine_val] = urbclim_val

    # One-hot encode the categorical data so the channels are added in the last dimension
    # Dimensions will become (height, width, channels)
    encoder = OneHotEncoder(
        sparse_output=False,
        categories=[list(range(1, len(set(CORINE_TO_URBCLIM.values())) + 1))],
        handle_unknown="ignore",
    )

    flat_landuse_urbclim = np.squeeze(urbclim_landuse_data).flatten().reshape(-1, 1)
    # Fit and transform, then reshape directly to (H, W, channels)
    landuse_encoded = encoder.fit_transform(flat_landuse_urbclim).reshape(
        urbclim_landuse_data.shape[1], urbclim_landuse_data.shape[2], -1
    )
    # Move the channels to the first dimension
    landuse_encoded_shifted = np.moveaxis(landuse_encoded, -1, 0)
    # TODO This assertion does not work for all of Belgium
    # assert np.all(np.sum(landuse_encoded_shifted, axis=0) == 1), (
    #     "One-hot encoding failed, not all channels sum to 1."
    # )
    return landuse_encoded_shifted


@lru_cache()
def get_digital_elevation_data(region):
    de_darr = rioxr.open_rasterio(
        OVERALL_FOLDER_PREFIX / "urbsim_runs" / region / "terrain" / "log" / "temp_dem.tif",
    )
    assert isinstance(de_darr, xr.DataArray)
    de = de_darr.values
    # NOTE Subtract the mean (Hendrik knows why)
    return de - de.mean()


@lru_cache()
def get_building_fraction_data(region):
    bf_darr = rioxr.open_rasterio(
        OVERALL_FOLDER_PREFIX / "urbsim_runs" / region / "terrain" / "log" / "temp_buildings.tif"
    )
    assert isinstance(bf_darr, xr.DataArray)
    # The building fraction data is in percentage, so we divide by 100 to get a fraction
    return bf_darr.values / 100


@lru_cache()
def get_soil_sealing_data(region):
    ss_darr = rioxr.open_rasterio(
        OVERALL_FOLDER_PREFIX / "urbsim_runs" / region / "terrain" / "log" / "temp_soil.tif"
    )
    assert isinstance(ss_darr, xr.DataArray)
    return ss_darr.values / 100


def reshape_into_groups(ds: xr.Dataset, dim="time", period=3) -> xr.Dataset:
    # 1. Infer constants
    # steps_per_day = 24 * 60 // interval_minutes  # 72 for 20-minute intervals

    total_steps = ds.sizes[dim]
    n_days = total_steps // period
    valid_steps = n_days * period

    # 2. Trim to full days onl:'tim'y
    ds_trimmed = ds.isel(time=slice(0, valid_steps))

    # 3. Reshape all data variables
    def reshape_var(da: xr.DataArray) -> xr.DataArray:
        reshaped = da.data.reshape(
            da.shape[: da.dims.index(dim)] + (n_days, period) + da.shape[(da.dims.index(dim) + 1) :]
        )  # preserve extra dims
        return xr.DataArray(
            reshaped,
            dims=da.dims[: da.dims.index(dim)]
            + (dim, "period")
            + da.dims[(da.dims.index(dim) + 1) :],  # replace 'time' with ('date', 'interval')
            attrs=da.attrs,
        )

    ds_reshaped = ds_trimmed.map(reshape_var)
    ds_reshaped[dim] = reshape_var(ds_trimmed[dim]).mean("period")

    # # 4. Build new coordinates
    # start_date = pd.Timestamp(ds_trimmed.time[0].values).normalize()
    # date_coords = pd.date_range(start=start_date, periods=n_days, freq="D")
    # interval_coords = pd.timedelta_range(start="00:00:00", freq=f"{interval_minutes}min", periods=period)

    # # 5. Assign new coords and drop old time
    # ds_reshaped = ds_reshaped.assign_coords(
    #     date=("date", date_coords),
    #     interval=("interval", interval_coords)
    # ).rename({'date':'time'})

    return ds_reshaped


def get_patch(target_var, time, region, patch_size, px, py):
    # Result should be a tuple (X, y) where
    # X is a 3D numpy array with shape (H, W, C) and
    # y is a 2D numpy array with shape (H, W)

    region_name = region.split("_")[0].capitalize()

    # See urbnet.abudhabi.dataset for examples

    meteo_data = load_era5_day(region=region, region_name=region_name, time=time)
    # # resulting variables:
    # ['Tmin','Tmax','WIND_SPEED_at_Tmin','WIND_SPEED_at_Tmax','WIND_DIR_SIN_at_Tmin','WIND_DIR_SIN_at_Tmax','WIND_DIR_COS_at_Tmin','WIND_DIR_COS_at_Tmax','Rs_mean','Rs_mean_prev_day']:

    ndvi_data = load_ndvi(region, month=time.month)

    urbclim_data = load_urbclim_for_time_v1(
        region=region,
        region_name=region_name,
        time=time,
        target_var=target_var,
    )
    # distance_coast_data = load_dtc(region)
    landuse_encoded = get_land_use_data_encoded(region)
    digital_elevation_data = get_digital_elevation_data(region)
    building_fraction_data = get_building_fraction_data(region)
    soil_sealing_data = get_soil_sealing_data(region)

    _, H, W = ndvi_data.shape

    def convert_to_4d(arr):
        return arr[..., np.newaxis]

    def convert_scalar_to_3d(scalar, h, w):
        # Go from a single number to a (1, H, W, 1) array
        return np.full((1, h, w), scalar)

    X = np.concatenate(
        (
            # SBN = should be normalized
            digital_elevation_data,  # SBN
            building_fraction_data,
            soil_sealing_data,
            # 10 meteo variables
            *tuple(
                [
                    convert_scalar_to_3d(meteo_data[var_name], h=H, w=W)
                    for var_name in METEO_VAR_NAMES[target_var]
                ]
            ),  # SBN
            ndvi_data,
            landuse_encoded,
        ),
        axis=0,
    )

    # multiple urbclim data variables
    y = np.array(urbclim_data[np.newaxis, ...])
    # y = np.concat([data[np.newaxis] for data in urbclim_data_vars],axis=0)

    # NOTE: Suggestion Dirk; cut off sides
    X = X[:, 10:-10, 10:-10]
    y = y[:, 10:-10, 10:-10]

    X_patch = X[:, py : py + patch_size, px : px + patch_size]
    y_patch = y[:, py : py + patch_size, px : px + patch_size]

    return X_patch, y_patch


# TODO further extend this to properly calculate aggregates (argmin for U, V)
def load_inference_era5(time):
    variable_mapping = {"T2M": "t", "U10M": "u", "V10M": "v", "Rs": "ssrd"}

    date_times = [time, time - pd.Timedelta(days=1)]

    ds_select_days = []
    for date_time in date_times:
        ds_select_days.append(xr.Dataset())
        for var, read_var in variable_mapping.items():
            file_name = f"BELGIUM_100m_{read_var}_{date_time:%Y%m%d}.nc"
            nc_path = INFERENCE_INPUT_FOLDER_PREFIX / "ERA5_samples" / file_name
            ds_select_days[-1][var] = xr.open_dataset(nc_path)[read_var]

    # NOTE: ERA5 is in UTC, so in principle we need to shift to Belgian time zone
    # However, minimum temperature happens before sunset, so we can skip time zone alignment.

    return process_era5_day_and_previous_day(ds_select_days[0], ds_select_days[1], steps_per_day=24)

def safe_read(src, replacement_value=np.nan):
    data = src.read(1)
    nodata = src.meta.get("nodata", None)
    if nodata is not None:
        # TODO what value to give NODATA? Right now, nan.
        data = np.where(data == nodata, replacement_value, data)
    return data


def load_inference_ndvi(time):
    # Infer the month from the time (%b)
    month = time.strftime("%b").lower()
    file_name = f"temp_veg_{month}.tif"
    tif_path = (
        OVERALL_FOLDER_PREFIX
        / "input_data"
        / "urbsim-config-belgium"
        / "terrain"
        / "log"
        / file_name
    )
    with rasterio.open(tif_path) as src:
        return safe_read(src)


def load_inference_dem():
    tif_path = (
        OVERALL_FOLDER_PREFIX
        / "input_data"
        / "urbsim-config-belgium"
        / "terrain"
        / "log"
        / "temp_dem.tif"
    )
    with rasterio.open(tif_path) as src:
        # TODO subtract mean here as well? Form of convolution, pixel per pixel
        dem = safe_read(src)
        dem_delta = dem - uniform_filter(dem, size=250)
        return dem_delta


def load_inference_building_fraction():
    tif_path = (
        OVERALL_FOLDER_PREFIX
        / "input_data"
        / "urbsim-config-belgium"
        / "terrain"
        / "log"
        / "temp_buildings.tif"
    )
    with rasterio.open(tif_path) as src:
        # The building fraction data is in percentage, so we divide by 100 to get a fraction
        return safe_read(src, replacement_value=0.0) / 100


def load_inference_soil_sealing():
    tif_path = (
        OVERALL_FOLDER_PREFIX
        / "input_data"
        / "urbsim-config-belgium"
        / "terrain"
        / "log"
        / "temp_soil.tif"
    )
    with rasterio.open(tif_path) as src:
        return safe_read(src, replacement_value=0.0) / 100


# def load_inference_dtc():
#     tif_path = UAE_INPUT_FOLDER_PREFIX / "distance_to_coast" / "full_domain_EPSG3857.tif"
#     with rasterio.open(tif_path) as src:
#         data = safe_read(src)
#         data = gaussian_filter(data, sigma=4)
#         return data


def convert_scalar_to_3d(scalar, h, w):
    return np.full((1, h, w), scalar, dtype=np.float32)


def get_inference_input(time, target_var):
    # TODO Are timestamps being handled properly?
    # What is the input time zone?
    # What timezone are the files?
    # Shift wrt UTC needed?

    logger.info("Constructing inference input data for all of BELGIUM")
    meteo_data = load_inference_era5(time=time)
    # era5_u = load_inference_era5("u", time=time)
    # era5_v = load_inference_era5("v", time=time)
    # era5_ssrd = load_inference_era5("ssrd", time=time)

    # era5_wind_speed = np.sqrt(era5_u**2 + era5_v**2)
    # era5_wind_dir_sin = np.sin(np.arctan2(era5_v, era5_u))
    # era5_wind_dir_cos = np.cos(np.arctan2(era5_v, era5_u))

    ndvi = load_inference_ndvi(time)
    # dtc_data = load_inference_dtc()
    digital_elevation_data = load_inference_dem()
    building_fraction_data = load_inference_building_fraction()
    soil_sealing_data = load_inference_soil_sealing()
    landuse_encoded = get_land_use_data_encoded("Belgium")

    # NOTE: it's important that this is in the same order as in the training data

    X = np.concatenate(
        (
            # prepend_dim(dtc_data),
            prepend_dim(digital_elevation_data),
            prepend_dim(building_fraction_data),
            prepend_dim(soil_sealing_data),
            *tuple(
                [np.flipud(meteo_data[var_name].values) for var_name in METEO_VAR_NAMES[target_var]]
            ),
            # prepend_dim(era5_wind_speed),
            # prepend_dim(era5_wind_dir_sin),
            # prepend_dim(era5_wind_dir_cos),
            # prepend_dim(era5_ssrd),
            # convert_scalar_to_3d(time_of_day, h=ndvi.shape[0], w=ndvi.shape[1]),
            prepend_dim(ndvi),
            landuse_encoded,
        ),
        axis=0,
    )
    X[X > 1.0e35] = np.nan

    return X


def prepend_dim(arr):
    return arr[np.newaxis, ...]


def compute_normalization(
    regions: List[str],
    target_var: str,
    patch_size: int,
    nb_patches,
    start_year,
    end_year,
    x_channels,
    y_channels,
    rng_seed,
):
    dataset_index = build_dataset_index(
        regions=regions,
        start_year=start_year,
        end_year=end_year,
        patch_size=patch_size,
        rng_seed=rng_seed,
    )

    x_sum = {ch: 0.0 for ch in x_channels}
    x_sum_sq = {ch: 0.0 for ch in x_channels}
    y_sum = {ch: 0.0 for ch in y_channels}
    y_sum_sq = {ch: 0.0 for ch in y_channels}
    count = 0

    # Sample nb_patches rows from the dataset index
    sampled_rows = dataset_index.sample(n=nb_patches, random_state=rng_seed).reset_index(drop=True)

    # Loop over the sampled rows, extract patches, and compute means and stds
    for _, row in sampled_rows.iterrows():
        time = row["time"]
        region = row["region"]
        py = row["py"]
        px = row["px"]
        X, y = get_patch(
            target_var=target_var,
            time=time,
            region=region,
            patch_size=patch_size,
            py=py,
            px=px,
        )

        _, H, W = X.shape
        n = H * W
        count += n

        for ch in x_channels:
            vals = X[ch].ravel()
            x_sum[ch] += vals.sum()
            x_sum_sq[ch] += np.square(vals).sum()

        for ch in y_channels:
            vals = y[ch].ravel()
            y_sum[ch] += vals.sum()
            y_sum_sq[ch] += np.square(vals).sum()

    x_means = {ch: x_sum[ch] / count for ch in x_channels}
    x_stds = {ch: np.sqrt(x_sum_sq[ch] / count - (x_sum[ch] / count) ** 2) for ch in x_channels}
    y_means = {ch: y_sum[ch] / count for ch in y_channels}
    y_stds = {ch: np.sqrt(y_sum_sq[ch] / count - (y_sum[ch] / count) ** 2) for ch in y_channels}

    norm_data = {
        "x_means": x_means,
        "x_stds": x_stds,
        "y_means": y_means,
        "y_stds": y_stds,
    }
    return norm_data


class JITDataset(Dataset):
    def __init__(
        self,
        regions,
        target_var,
        patch_size,
        start_year,
        end_year,
        nb_patches,
        norm_data,
        rng_seed,
        type: Literal["train", "val"],
    ):
        dataset_index = build_dataset_index(
            regions=regions,
            start_year=start_year,
            end_year=end_year,
            patch_size=patch_size,
            rng_seed=rng_seed,
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
        self.norm_data = norm_data
        self.target_var = target_var
        self.sampled_index = dataset_index.sample(
            n=self.nb_patches, random_state=rng_seed
        ).reset_index(drop=True)
        self.type = type

    def __len__(self):
        return self.nb_patches

    def __getitem__(self, idx):
        row = self.sampled_index.iloc[idx]
        time = row["time"]
        region = row["region"]
        py = row["py"]
        px = row["px"]

        X, y = get_patch(
            target_var=self.target_var,
            time=time,
            region=region,
            patch_size=self.patch_size,
            py=py,
            px=px,
        )

        if self.type == "train" and idx < 3 and False:
            visualize_bands(X, y, filepath=f"{idx}_{time:%Y%m%d}_{region}_{py}_{px}.png")

        for ch, mean in self.norm_data["x_means"].items():
            X[ch, :, :] = (X[ch, :, :] - mean) / self.norm_data["x_stds"][ch]

        for ch, mean in self.norm_data["y_means"].items():
            y[ch, :, :] = (y[ch, :, :] - mean) / self.norm_data["y_stds"][ch]

        if self.type == "train" and idx < 3 and False:
            visualize_bands(X, y, filepath=f"{idx}_{time:%Y%m%d}_{region}_{py}_{px}_norm.png")

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError(
                f"NaN found for time {time}, region {region} at ({py}, {px}): X {np.argwhere(np.isnan(X))}, y {np.argwhere(np.isnan(y))}"
            )

        assert X.shape == (
            get_number_of_predictor_channels(self.target_var),
            self.patch_size,
            self.patch_size,
        )
        assert y.shape == (NUMBER_OF_TARGET_CHANNELS, self.patch_size, self.patch_size)
        X_torch = torch.from_numpy(X).float()
        y_torch = torch.from_numpy(y).float()

        return X_torch, y_torch


if __name__ == "__main__":
    X, y = get_patch(
        target_var="t_min",
        time=pd.Timestamp("2016-04-15", tz="UTC"),
        region="antwerp_25042025",
        patch_size=100,
        px=10,
        py=10,
    )
    visualize_bands(X, y, filepath="test_patch.png")
