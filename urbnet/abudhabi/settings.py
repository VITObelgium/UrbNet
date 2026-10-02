# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

from pathlib import Path
import os

OVERALL_FOLDER_PREFIX = Path(os.getenv("URBNET_ABUDHABI_DATA_ROOT", "/projects/"))
OUTPUT_FOLDER_ROOT = Path(os.getenv("URBNET_OUTPUT_ROOT", "/projects/neural_net/"))
NUMBER_OF_PREDICTOR_CHANNELS_TEMPERATURE = 22
NUMBER_OF_PREDICTOR_CHANNELS_WBGT = 23
NUMBER_OF_PREDICTOR_CHANNELS_HUMIDITY = 23
CITIES_WITH_URBCLIM_DATA = [
    "AbuDhabi",
    "AbuQrayn",
    "AlAin",
    "AlDhannah",
    "AlJazeerah",
    "AlKhaznah",
    "AlQuoa",
    "Dalma",
    "Doha",
    "Dubai",
    "Fujairah",
    "Ghayathi",
    "LiwaOasis",
    "MedinatZayed",
    "RasAlKhaimah",
    "Sila",
    "Sweihan",
    "UmAlZumoul",
]
UAE_CITIES = CITIES_WITH_URBCLIM_DATA
AD_CITIES = [
    "AbuDhabi",
    "AbuQrayn",
    "AlAin",
    "AlDhannah",
    "AlJazeerah",
    "AlKhaznah",
    "AlQuoa",
    "Dalma",
    "Ghayathi",
    "LiwaOasis",
    "MedinatZayed",
    "Sila",
    "Sweihan",
    "UmAlZumoul",
]
UAE_INPUT_FOLDER_PREFIX = OVERALL_FOLDER_PREFIX / "input_data" / "UAE"
AD_INPUT_FOLDER_PREFIX = OVERALL_FOLDER_PREFIX / "input_data" / "AD"
MLFLOW_EXPERIMENT_NAME = "HHEATMAP"
# Indices follow the input order in dataset.get_patch and produce; temperature removes Q2M at index 4.
NORMALIZATION_CHANNEL_SPEC_TEMPERATURE = {
    "x_channels": {
        0: "distance_to_coast",
        1: "digital_elevation",
        3: "temperature",
        4: "wind_speed",
    },
    "y_channels": {0: "temperature"},
}
NORMALIZATION_CHANNEL_SPEC_WBGT = {
    "x_channels": {
        0: "distance_to_coast",
        1: "digital_elevation",
        3: "temperature",
        4: "humidity",
        5: "wind_speed",
    },
    "y_channels": {0: "wbgt"},
}
NORMALIZATION_CHANNEL_SPEC_HUMIDITY = {
    "x_channels": {
        0: "distance_to_coast",
        1: "digital_elevation",
        3: "temperature",
        4: "humidity",
        5: "wind_speed",
    },
    "y_channels": {0: "humidity"},
}
