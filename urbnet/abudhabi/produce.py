# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from loguru import logger

from urbnet.abudhabi.dataset import (
    get_production_input,
    get_radiation_all_times,
    load_normalization,
    prepend_dim,
)
from urbnet.abudhabi.model import SimpleCNN
from urbnet.abudhabi.settings import (
    AD_INPUT_FOLDER_PREFIX,
    MLFLOW_EXPERIMENT_NAME,
    NORMALIZATION_CHANNEL_SPEC_HUMIDITY,
    NORMALIZATION_CHANNEL_SPEC_TEMPERATURE,
    NORMALIZATION_CHANNEL_SPEC_WBGT,
    NUMBER_OF_PREDICTOR_CHANNELS_HUMIDITY,
    NUMBER_OF_PREDICTOR_CHANNELS_TEMPERATURE,
    NUMBER_OF_PREDICTOR_CHANNELS_WBGT,
    OUTPUT_FOLDER_ROOT,
    UAE_INPUT_FOLDER_PREFIX,
)
from urbnet.utils import (
    fetch_mlflow_run,
    load_mlflow_model_weights,
    mlflow_available,
)

# from urbnet.visualize import visualize_prediction


def produce(
    run_names: dict,
    times: list[datetime],
    output_var: str,
    target_domain: str,
    operational_forecast_folder: str,
    experiment_name: str = MLFLOW_EXPERIMENT_NAME,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    models = {}
    for key, run_name in run_names.items():
        mlflow_run = fetch_mlflow_run(experiment_name=experiment_name, run_name=run_name)
        params = mlflow_run.data.params

        # NOTE: Model architecture assumed fixed and known (we're only loading model weights)
        logger.info("Constructing model")
        if output_var == "T2M":
            model = SimpleCNN(
                in_channels=int(
                    params.get(
                        "number_of_predictor_channels", NUMBER_OF_PREDICTOR_CHANNELS_TEMPERATURE
                    )
                )
            ).to(device)
        elif output_var == "WBGT":
            model = SimpleCNN(
                in_channels=int(
                    params.get("number_of_predictor_channels", NUMBER_OF_PREDICTOR_CHANNELS_WBGT)
                )
            ).to(device)
        elif output_var == "QV2M":
            model = SimpleCNN(
                in_channels=int(
                    params.get("number_of_predictor_channels", NUMBER_OF_PREDICTOR_CHANNELS_HUMIDITY)
                )
            ).to(device)
        else:
            raise ValueError(f"Unsupported output_var: {output_var}")
        logger.info("Downloading model weights and loading in")
        state_dict = load_mlflow_model_weights(
            mlflow_run=mlflow_run, device=device
        )
        model.load_state_dict(state_dict)
        model.eval()
        models[key] = model

    (
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
    ) = get_production_input(
        times=times,
        target_domain=target_domain,
        operational_forecast_folder=operational_forecast_folder,
    )

    tile_size = 800
    trim = models[next(iter(models))].num_conv_layers
    stride = tile_size - 2 * trim
    if output_var == "T2M":
        norm_data = load_normalization(
            x_channels=NORMALIZATION_CHANNEL_SPEC_TEMPERATURE["x_channels"],
            y_channels=NORMALIZATION_CHANNEL_SPEC_TEMPERATURE["y_channels"],
        )
    elif output_var == "WBGT":
        norm_data = load_normalization(
            x_channels=NORMALIZATION_CHANNEL_SPEC_WBGT["x_channels"],
            y_channels=NORMALIZATION_CHANNEL_SPEC_WBGT["y_channels"],
        )
    elif output_var == "QV2M":
        norm_data = load_normalization(
            x_channels=NORMALIZATION_CHANNEL_SPEC_HUMIDITY["x_channels"],
            y_channels=NORMALIZATION_CHANNEL_SPEC_HUMIDITY["y_channels"],
        )
    else:
        raise ValueError(f"Unsupported output_var: {output_var}")
    T, H, W = era5_t.shape[0], era5_t.shape[1], era5_t.shape[2]
    y_np = np.full((T, H, W), np.nan)

    # first identify which model to use in which time step
    df_clusters = get_radiation_all_times(
        n_circ_class_clusters=8,
        start_year=times[0].year,
        end_year=times[-1].year,
        operational_forecast_folder=operational_forecast_folder,
    )
    times_naive = [pd.Timestamp(t).tz_convert("UTC").tz_localize(None) for t in times]
    df_filtered = df_clusters.loc[df_clusters.index.intersection(times_naive)]
    if len(models) == 1:
        df_filtered["model_key"] = list(models.keys())[0]
    elif len(models) == 2:  # day-night
        df_filtered["model_key"] = df_filtered["day_night"]
    elif len(models) == 16:  # day-night-cluster
        df_filtered["model_key"] = (
            df_filtered["day_night"] + "-" + df_filtered["cluster"].astype(str)
        )
    else:
        raise ValueError(f"Unsupported number of models: {len(models)}")

    for idx, time in enumerate(times):
        logger.info(f"Producing prediction for time {time} ({idx + 1}/{len(times)})")

        # pick the model for this time step
        model = models[df_filtered.iloc[idx]["model_key"]]
        # NOTE: it's important that this is in the same order as in the training data
        X_np = np.concatenate(
            (
                prepend_dim(dtc_data),
                prepend_dim(digital_elevation_data),
                prepend_dim(building_fraction_data),
                # prepend_dim(soil_sealing_data),
                prepend_dim(era5_t[idx, :, :]),
                prepend_dim(era5_q[idx, :, :]),
                prepend_dim(era5_wind_speed[idx, :, :]),
                prepend_dim(era5_wind_dir_sin[idx, :, :]),
                prepend_dim(era5_wind_dir_cos[idx, :, :]),
                prepend_dim(cs_frac[idx, :, :]),
                # convert_scalar_to_3d(time_of_day, h=ndvi.shape[0], w=ndvi.shape[1]),
                prepend_dim(Rs[idx, :, :]),
                prepend_dim(ndvi[time.month]),
                landuse_encoded,
            ),
            axis=0,
        )

        if output_var == "T2M":
            X_np = np.concatenate((X_np[:4], X_np[5:]), axis=0)  # remove specific humidity channel

        logger.info(
            f"Sliding over {H}x{W} image with {tile_size}x{tile_size} tiles and stride {stride}"
        )

        tile_coords = []

        for ypos in range(0, H, stride):
            for xpos in range(0, W, stride):
                tile_coords.append((ypos, xpos))

                logger.info(
                    f"Processing tile ({ypos} - {ypos + tile_size}, {xpos} - {xpos + tile_size}) size {tile_size}x{tile_size}"
                )
                # NOTE: This .copy() is important! Without it, Xtile_np will be a mere view of X_np
                # that will be normalized multiple times for regions where tiles overlap.
                Xtile_np = X_np[:, ypos : ypos + tile_size, xpos : xpos + tile_size].copy()

                for ch, mean in norm_data["x_means"].items():
                    Xtile_np[ch, :, :] = (Xtile_np[ch, :, :] - mean) / norm_data["x_stds"][ch]

                Xtile_t = torch.from_numpy(Xtile_np).float().to(device).unsqueeze(0)
                with torch.no_grad():
                    ypred = model(Xtile_t)
                    ytile = ypred.cpu().numpy() * norm_data["y_stds"][0] + norm_data["y_means"][0]
                    # Check that the resulting tile is trimmed as expected
                    assert ytile.shape == (
                        1,
                        1,
                        Xtile_t.shape[2] - 2 * trim,
                        Xtile_t.shape[3] - 2 * trim,
                    )
                y0 = ypos + trim
                x0 = xpos + trim
                # Deal with incomplete tiles at the edges
                y1 = min(ypos + tile_size - trim, H - trim)
                x1 = min(xpos + tile_size - trim, W - trim)
                logger.info(
                    f"> Output tile domain: ({y0} - {y1}, {x0} - {x1}), size {y1 - y0}x{x1 - x0}"
                )
                # Check that the output tile perfectly fits in the specified region of the output array
                assert y1 - y0 == ytile.shape[2]
                assert x1 - x0 == ytile.shape[3]
                # Place the output tile in the correct position in the output array
                y_np[idx, y0:y1, x0:x1] = ytile[0, 0, :, :]

    # visualise only land
    ## first get rid of the oceans
    mask = (
        (np.squeeze(landuse_encoded[8, :, :]) == 1)
        | (np.squeeze(landuse_encoded[9, :, :]) == 1)
        | (np.squeeze(landuse_encoded[11, :, :]) == 1)
    )
    y_np[:, mask] = np.nan

    ## next get rid of everything outside AD bounding box
    if target_domain == "UAE":
        filename = f"{UAE_INPUT_FOLDER_PREFIX}/{target_domain}_extent_EPSG3857.tif"
    elif target_domain == "AD":
        filename = f"{AD_INPUT_FOLDER_PREFIX}/{target_domain}_extent_EPSG3857.tif"
    else:
        raise ValueError(f"Unsupported target_domain: {target_domain}")
    with rasterio.open(filename) as src:
        mask = src.read(1)
    y_np[:, mask == 0] = np.nan

    # Kelvin to degrees
    if output_var == "T2M":
        y_np = y_np - 273.15

    # output as netcdf file
    # Unpack raster geometry
    transform = meta["transform"]  # Affine(a, b, c, d, e, f)
    a, b, c, d, e, f = transform.a, transform.b, transform.c, transform.d, transform.e, transform.f
    W, H = meta["width"], meta["height"]  # note: width = x size (cols), height = y size (rows)

    # Pixel-center coordinates (x to the east, y to the north; e < 0 means y decreases downward)
    x = c + a * (np.arange(W) + 0.5)  # length W
    y = f + e * (np.arange(H) + 0.5)  # length H (descending if e < 0)

    # put time in the right format
    np_times = np.array(times, dtype="datetime64[ns]")

    # Build DataArray (dims: time, y, x)
    da = xr.DataArray(
        y_np, dims=("time", "y", "x"), coords={"time": np_times, "x": x, "y": y}, name=output_var
    )

    # Dataset + CF-ish spatial metadata
    ds = xr.Dataset({output_var: da})

    # A GDAL/CF-compatible CRS variable (safe, widely readable)
    crs_wkt = meta["crs"].to_wkt() if hasattr(meta["crs"], "to_wkt") else str(meta["crs"])
    ds["spatial_ref"] = xr.DataArray(
        0,
        attrs={
            "spatial_ref": crs_wkt,  # GDAL-style
            "crs_wkt": crs_wkt,  # some readers expect this key
            "epsg_code": "EPSG:3857",
        },
    )
    ds[output_var].attrs["grid_mapping"] = "spatial_ref"
    if output_var == "WBGT":
        ds[output_var].attrs["long_name"] = "Wet Bulb Globe Temperature"
    elif output_var == "T2M":
        ds[output_var].attrs["long_name"] = "2m Air Temperature"
    elif output_var == "QV2M":
        ds[output_var].attrs["long_name"] = "2m Specific Humidity"
    ds[output_var].attrs["units"] = "°C"

    # Store the geotransform as a GDAL extension (helps some tools)
    # GDAL GeoTransform order: (c, a, b, f, d, e)
    ds[output_var].attrs["GeoTransform"] = f"{c} {a} {b} {f} {d} {e}"

    # some general attributes
    ds.attrs = {
        "title": f"{output_var} for Abu Dhabi Emirate",
        "institution": "Flemish Institute for Technological Research (VITO)",
        "source": "UrbNet Deep Learning Model",
        "contact": "niels.souverijns@vito.be",
        "crs": "EPSG:3857",
    }

    # Encoding: keep your nodata, dtype, and add compression
    if output_var == "QV2M":
        scale_factor = 1e-5
    else:
        scale_factor = 0.01
    encoding = {
        output_var: {
            "scale_factor": scale_factor,
            "chunksizes": (1, 500, 500),
            "dtype": "int32",
            "_FillValue": -32768,
            "missing_value": -32768,
            "zlib": True,
            "complevel": 9,
        },
        "time": {
            "units": "hours since 1970-01-01 00:00:00",
            "calendar": "standard",
            "dtype": "float64",
        },
    }

    # Write NetCDF
    logger.info("Writing NetCDF output")
    if "ECMWF" in operational_forecast_folder or "ICON" in operational_forecast_folder:
        output_folder = operational_forecast_folder
    else:
        output_folder = (
            f"{OUTPUT_FOLDER_ROOT}/prod-{list(run_names.keys())[0]}-"
            + f"{list(run_names.values())[0]}"
        )
    os.makedirs(output_folder, exist_ok=True)
    output_name = (
        f"{output_folder}/{target_domain}_{output_var}_"
        + f"{times[0].year}{str(times[0].month).zfill(2)}{str(times[0].day).zfill(2)}T{str(times[0].hour).zfill(2)}-"
        + f"{times[-1].year}{str(times[-1].month).zfill(2)}{str(times[-1].day).zfill(2)}T{str(times[-1].hour).zfill(2)}.nc"
    )
    ds.to_netcdf(output_name, encoding=encoding)
    logger.info(f"Written output to {output_name}")

    # visualize_prediction(
    #     y_np,
    #     output_file=f"{OUTPUT_FOLDER_ROOT}/{run_name}/{target_domain}_{time.strftime('%Y%m%d_%H%M')}.png",
    #     time=time,
    #     meta=meta,
    # )

    # TODO: integrate the resulting prediction into the existing filesystem or MLFlow
    assert mlflow_available()


if __name__ == "__main__":
    start = datetime(2024, 11, 15, 11, 0, tzinfo=timezone.utc)  # always start at hour 1
    end = datetime(2024, 11, 15, 23, 0, tzinfo=timezone.utc)
    delta = timedelta(hours=1)
    dt_list = []
    current = start
    while current <= end:
        dt_list.append(current)
        current += delta
    produce(
        run_names={"all": "v188"},
        times=dt_list,
        target_domain="UAE",
        output_var="WBGT",
        operational_forecast_folder="not_applicable",
    )
