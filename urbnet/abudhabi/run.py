# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import inspect
import os

from loguru import logger

from urbnet.abudhabi.model import train_abudhabi
from urbnet.abudhabi.settings import (
    CITIES_WITH_URBCLIM_DATA,
    MLFLOW_EXPERIMENT_NAME,
    OUTPUT_FOLDER_ROOT,
)
from urbnet.utils import get_next_version, get_output_folder_for_run, mlflow_run


def run(
    cities_train: list[str],
    nb_patches_train: int,
    cities_val: list[str],
    nb_patches_val: int,
    patch_size: int = 100,
    start_year: int = 2015,
    end_year: int = 2024,
    n_circ_class_clusters: int = 8,
    circ_class_cluster_idx: int = 0,
    use_individual_model_per_cluster: bool = True,
    use_day_night_model: bool = True,
    time_of_the_day: str = "day",
    learning_rate: float = 0.001,
    batch_size: int = 16,
    nb_epochs: int = 15,
    output_var: str = "temperature",
):
    logger.info("Verifying input parameters")
    if circ_class_cluster_idx >= n_circ_class_clusters:
        raise ValueError(
            f"circ_class_cluster_idx {circ_class_cluster_idx} must be less than n_circ_class_clusters {n_circ_class_clusters}"
        )

    # Check that all cities are in the list of available cities
    for city in cities_train + cities_val:
        if city not in CITIES_WITH_URBCLIM_DATA:
            raise ValueError(
                f"No UrbClim data available for {city}; choose from {CITIES_WITH_URBCLIM_DATA}"
            )

    # throw warning if there are cities in cities_val that appear in cities_train
    common_cities = set(cities_train) & set(cities_val)
    if common_cities:
        logger.warning(
            f"Cities {common_cities} are present in both training and validation sets. "
            "This may lead to overfitting."
        )

    # Automatically infer next version
    version = get_next_version(OUTPUT_FOLDER_ROOT)

    # Build dictionary of parameters from function arguments
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Failed to obtain current frame for argument inspection.")
    args, _, _, values = inspect.getargvalues(frame)
    params = {arg: values[arg] for arg in args}

    with mlflow_run(version, MLFLOW_EXPERIMENT_NAME, params):
        output_folder_run = get_output_folder_for_run(OUTPUT_FOLDER_ROOT, version)
        # Ensure group write permissions (chmod g+w)
        os.chmod(output_folder_run, os.stat(output_folder_run).st_mode | 0o20)

        train_abudhabi(
            cities_train=cities_train,
            nb_patches_train=nb_patches_train,
            cities_val=cities_val,
            nb_patches_val=nb_patches_val,
            patch_size=patch_size,
            start_year=start_year,
            end_year=end_year,
            n_circ_class_clusters=n_circ_class_clusters,
            circ_class_cluster_idx=circ_class_cluster_idx,
            use_individual_model_per_cluster=use_individual_model_per_cluster,
            use_day_night_model=use_day_night_model,
            time_of_the_day=time_of_the_day,
            nb_epochs=nb_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            version=version,
            output_folder_run=output_folder_run,
            output_var=output_var,
        )

    logger.info("Run completed successfully")
    return version


if __name__ == "__main__":
    cities_train = ["Sila", "AbuDhabil", "Sweihan", "AlAin", "LiwaOasis"]
    run(
        cities_train=cities_train,
        nb_patches_train=160 * len(cities_train),
        cities_val=["Ghayathi"],
        nb_patches_val=32,
        nb_epochs=15,
    )
