# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import inspect
import os
from typing import Literal

from loguru import logger

from urbnet.belgium.model import train_vectorborne
from urbnet.belgium.settings import (
    REGIONS_WITH_URBCLIM_DATA,
    get_experiment_name,
    get_output_folder,
)
from urbnet.utils import get_next_version, mlflow_run


def run(
    regions_train: list[str],
    target_var: Literal["t_min", "t_max", "t_mean"],
    nb_patches_train: int,
    nb_patches_norm: int,
    regions_val: list[str],
    nb_patches_val: int,
    patch_size: int = 100,
    start_year: int = 2005,
    end_year: int = 2015,
    learning_rate: float = 0.001,
    batch_size: int = 16,
    nb_epochs: int = 15,
):
    # Check that all cities are in the list of available cities
    for region in regions_train + regions_val:
        if region not in REGIONS_WITH_URBCLIM_DATA:
            raise ValueError(
                f"No UrbClim data available for {region}; choose from {REGIONS_WITH_URBCLIM_DATA}"
            )

    # throw warning if there are cities in cities_val that appear in cities_train
    common_regions = set(regions_train) & set(regions_val)
    if common_regions:
        logger.warning(
            f"Cities {common_regions} are present in both training and validation sets. "
            "This may lead to overfitting."
        )

    # Automatically infer next version
    output_folder_root = get_output_folder(target_var)
    version = get_next_version(output_folder_root)

    # Build dictionary of parameters from function arguments
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Failed to obtain current frame for argument inspection.")
    args, _, _, values = inspect.getargvalues(frame)
    params = {arg: values[arg] for arg in args}

    experiment_name = get_experiment_name(target_var)
    output_folder_run = output_folder_root / version
    logger.info(f"Creating run output folder: {output_folder_run}")
    output_folder_run.mkdir(parents=True, exist_ok=False)
    # Ensure group write permissions (chmod g+w)
    os.chmod(output_folder_run, os.stat(output_folder_run).st_mode | 0o20)

    with mlflow_run(version, experiment_name, params):
        train_vectorborne(
            regions_train=regions_train,
            target_var=target_var,
            nb_patches_train=nb_patches_train,
            nb_patches_norm=nb_patches_norm,
            regions_val=regions_val,
            nb_patches_val=nb_patches_val,
            patch_size=patch_size,
            start_year=start_year,
            end_year=end_year,
            nb_epochs=nb_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            version=version,
            output_folder_run=output_folder_run,
        )

    logger.info("Run completed successfully.")
    return version


if __name__ == "__main__":
    regions_train = [
        "antwerp_25042025",
        "gent_08052025",
        "liege_08052025",
    ]
    run(
        regions_train=regions_train,
        target_var="t_max",
        nb_patches_train=32 * len(regions_train),
        nb_patches_norm=32,
        regions_val=["westhoek_13052025"],
        start_year=2002,
        end_year=2017,
        nb_patches_val=32,
        nb_epochs=5,
    )
