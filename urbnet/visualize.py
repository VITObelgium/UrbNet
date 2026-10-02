# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import datetime as dt
import math
from pathlib import Path
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import rasterio
import torch
from loguru import logger
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.utils.data import DataLoader

from urbnet.utils import mlflow_available


def run_example_predictions(
    model,
    dataloader: DataLoader,
    nb_batches: int,
    norm_data: dict[str, dict[int, float]],
    version,
    output_folder_run,
    device,
    type: Literal["val", "train"] = "val",
    output_var: str = "temperature",
):
    logger.info(f"Running example predictions on {nb_batches} batches of {type} dataset")
    nb_batches = 65  # approximately 1000 samples (16*65=1040)
    val_iter = iter(dataloader)
    for batch_num in range(nb_batches):
        try:
            X_batch, y_true = next(val_iter)
            y_true = model.trim_output(y_true)
        except StopIteration:
            break  # No more batches
        X_batch = X_batch.to(device)
        y_true = y_true.to(device)
        model.eval()
        with torch.no_grad():
            y_pred = model(X_batch)

        # Move tensors to CPU and convert to numpy
        y_pred_np = y_pred.cpu().numpy()
        y_true_np = y_true.cpu().numpy()

        for ch, mean in norm_data["y_means"].items():
            y_pred_np[:, ch, :, :] = y_pred_np[:, ch, :, :] * norm_data["y_stds"][0] + mean
            y_true_np[:, ch, :, :] = y_true_np[:, ch, :, :] * norm_data["y_stds"][0] + mean

        for band_index in range(y_true_np.shape[1]):
            visualize_predictions(
                true_imgs=y_true_np[:, band_index],
                pred_imgs=y_pred_np[:, band_index],
                batch_num=batch_num,
                version=version,
                output_folder_run=output_folder_run,
                type=type,
                band_index=band_index,
                output_var=output_var,
            )


def visualize_bands(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    band_names: Optional[List[str]] = None,
    filepath: Path | str = "input_bands.png",
):
    """
    Visualize a single input band as an image.
    :param arr: 2D numpy array representing the input band.
    """
    arr = np.concatenate((X, y), axis=0) if y is not None else X
    n_subplots = arr.shape[0]
    n_rows = min(n_subplots, 5)
    n_cols = math.ceil(n_subplots / n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)

    if band_names is None:
        x_bands = [f"X[{i}]" for i in range(X.shape[0])]
        y_bands = [f"y[{i}]" for i in range(y.shape[0])] if y is not None else []
        band_names = x_bands + y_bands
    else:
        assert len(band_names) == n_subplots, (
            f"Expected {n_subplots} band names, got {len(band_names)} (number of bands in X and y)"
        )

    for idx in range(n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        if idx < n_subplots:
            im = ax.imshow(arr[idx], cmap="plasma")
            ax.set_title(band_names[idx])
            ax.axis("off")
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
        else:
            ax.axis("off")

    logger.info(f"Storing input bands visualization at {filepath}")
    plt.tight_layout()
    plt.savefig(filepath, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visualize_predictions(
    true_imgs: np.ndarray,
    pred_imgs: np.ndarray,
    batch_num: int,
    version: str,
    output_folder_run: Path,
    type: Literal["val", "train"],
    band_index: int,
    output_var: str = "temperature",
):
    # Show at most 8 predictions per image
    max_rows_per_fig = 8
    total_rows = true_imgs.shape[0]
    num_figs = math.ceil(total_rows / max_rows_per_fig)

    diff_vmin = -5
    diff_vmax = 5
    if output_var == "humidity":
        diff_vmin = -0.005
        diff_vmax = 0.005

    for fig_idx in range(num_figs):
        start_idx = fig_idx * max_rows_per_fig
        end_idx = min(start_idx + max_rows_per_fig, total_rows)
        n_rows = end_idx - start_idx

        output_file = f"{output_folder_run}/{version}_prediction_{type}_batch_{batch_num}_part_{fig_idx:02d}_band_{band_index}.png"
        logger.info(
            f"Storing prediction on {type} for batch {batch_num} part {fig_idx} at {output_file}"
        )

        fig, axes = plt.subplots(n_rows, 3, figsize=(4 * 3 + 1.5, 4 * n_rows), squeeze=False)
        # plt.subplots_adjust(wspace=0, hspace=0)
        # Set column titles once at the top
        for col, title in enumerate(["Ground Truth (K)", "Prediction (K)", "Difference"]):
            axes[0, col].set_title(title)

        for i in range(n_rows):
            idx = start_idx + i
            # Ground Truth
            ax_gt = axes[i, 0]
            im1 = ax_gt.imshow(true_imgs[idx], cmap="plasma")
            vmin_gt, vmax_gt = im1.get_clim()
            ax_gt.axis("off")
            divider_gt = make_axes_locatable(ax_gt)
            cax_gt = divider_gt.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im1, cax=cax_gt)

            # Prediction
            ax_pred = axes[i, 1]
            im2 = ax_pred.imshow(pred_imgs[idx], cmap="plasma", vmin=vmin_gt, vmax=vmax_gt)
            ax_pred.axis("off")
            divider_pred = make_axes_locatable(ax_pred)
            cax_pred = divider_pred.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im2, cax=cax_pred)

            # Difference
            ax_diff = axes[i, 2]
            im3 = ax_diff.imshow(
                true_imgs[idx] - pred_imgs[idx], cmap="coolwarm_r", vmin=diff_vmin, vmax=diff_vmax
            )
            ax_diff.axis("off")
            divider_diff = make_axes_locatable(ax_diff)
            cax_diff = divider_diff.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im3, cax=cax_diff)

        # Remove all spacing between subplots
        plt.subplots_adjust(wspace=0.1, hspace=0.07, left=0, right=0.92, top=1, bottom=0)
        plt.savefig(output_file, dpi=300, bbox_inches="tight", pad_inches=0)
        if mlflow_available():
            mlflow.log_artifact(output_file)
        plt.close(fig)

    if type == "val":
        # also save the validation samples as geotiff files for further analysis
        for i in range(true_imgs.shape[0]):
            output_true_tif = f"{output_folder_run}/{version}_prediction_{type}_batch_{batch_num}_sample_{i:02d}_true_img.tif"
            output_pred_tif = f"{output_folder_run}/{version}_prediction_{type}_batch_{batch_num}_sample_{i:02d}_pred_img.tif"
            logger.info(
                f"Exporting true and prediction sample {i} to GeoTIFF: {output_true_tif} & {output_pred_tif}"
            )
            # fill the nodata values
            nodata_value = -9999
            true_array_filled = np.where(np.isnan(true_imgs[i]), nodata_value, true_imgs[i])
            pred_array_filled = np.where(np.isnan(pred_imgs[i]), nodata_value, pred_imgs[i])
            meta = {
                "height": true_imgs[i].shape[0],
                "width": true_imgs[i].shape[1],
                "count": 1,
                "dtype": true_array_filled.dtype,
                "nodata": nodata_value,
                "compress": "deflate",
                "interleave": "band",
                "tiled": True,
            }
            with rasterio.open(output_true_tif, "w", **meta) as dst:
                dst.write(true_array_filled, 1)
            with rasterio.open(output_pred_tif, "w", **meta) as dst:
                dst.write(pred_array_filled, 1)


def visualize_prediction(
    y_np,
    time,
    output_file="produce.png",
    meta=None,
):
    logger.info(f"Visualizing prediction and saving to {output_file}")
    fig, ax = plt.subplots(figsize=(6, 6))

    im = ax.imshow(y_np - 273.15, cmap="plasma")
    ax.axis("off")

    # Create colorbar with the same height as the image
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label("Temperature (°C)", fontsize=8)
    cbar.ax.tick_params(labelsize=8)  # Set tick label size

    # Adjust title fontsize
    if type(time) is dt.date:
        time = dt.datetime(*time.timetuple()[:6])

    ax.set_title(
        f"Temperature for {time.strftime('%Y-%m-%d at %H:%M ')}{time.tzname() if time.tzinfo else ''}",
        fontsize=8,
        pad=10,
    )

    plt.savefig(output_file, dpi=600, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    # also export to a geotiff file
    if meta is not None:
        output_file_tif = output_file.replace(".png", ".tif")
        logger.info(f"Exporting prediction to GeoTIFF: {output_file_tif}")
        # fill the nodata values
        nodata_value = -9999
        array_filled = np.where(np.isnan(y_np), nodata_value, y_np)
        meta.update(
            {
                "height": y_np.shape[0],
                "width": y_np.shape[1],
                "count": 1,
                "dtype": array_filled.dtype,
                "nodata": nodata_value,
                "compress": "deflate",
                "interleave": "band",
                "tiled": True,
            }
        )
        with rasterio.open(output_file_tif, "w", **meta) as dst:
            dst.write(array_filled, 1)
