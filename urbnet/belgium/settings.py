# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

from pathlib import Path
import os

OVERALL_FOLDER_PREFIX = Path(os.getenv("URBNET_BELGIUM_DATA_ROOT", "/projects/"))
OUTPUT_FOLDER_ROOT = Path(os.getenv("URBNET_OUTPUT_ROOT", "/projects/neural_net/"))
NUMBER_OF_PREDICTOR_CHANNELS = 24
NUMBER_OF_TARGET_CHANNELS = 1
REGIONS_WITH_URBCLIM_DATA = [
    "antwerp_25042025",
    "gent_08052025",
    "liege_08052025",
    "westhoek_13052025",
    "brussels_17072025",
    "Kempen_18072025",
    "Polders_18072025",
    "ardennen_13052025",
    "Charleroi_18072025",
    "Haspengouw_18072025",
    "brussels_04062025",
    "Fagnes_18072025",
    "Henegouwen_18072025",
    "Lotharingen_18072025",
]
INFERENCE_INPUT_FOLDER_PREFIX = OVERALL_FOLDER_PREFIX / "input_data" / "BELGIUM"


def get_experiment_name(target_var):
    return {
        "t_min": "VECTORBORNE_TMIN",
        "t_max": "VECTORBORNE_TMAX",
        "t_mean": "VECTORBORNE_TMEAN",
    }[target_var]


def get_output_folder(target_var):
    return {
        "t_min": OUTPUT_FOLDER_ROOT / "t_min",
        "t_max": OUTPUT_FOLDER_ROOT / "t_max",
        "t_mean": OUTPUT_FOLDER_ROOT / "t_mean",
    }[target_var]


# 10 meteo var names
METEO_VAR_NAMES = {
    "t_min": [
        "T2M_at_Tmin",
        # 'T2M_at_Tmax',
        "WIND_SPEED_at_Tmin",
        # 'WIND_SPEED_at_Tmax',
        "WIND_DIR_SIN_at_Tmin",
        # 'WIND_DIR_SIN_at_Tmax',
        "WIND_DIR_COS_at_Tmin",
        # 'WIND_DIR_COS_at_Tmax',
        # 'Rs_mean',
        "Rs_mean_prev_day",
    ],
    "t_max": [
        "T2M_at_Tmax",
        # 'T2M_at_Tmax',
        "WIND_SPEED_at_Tmax",
        # 'WIND_SPEED_at_Tmax',
        "WIND_DIR_SIN_at_Tmax",
        # 'WIND_DIR_SIN_at_Tmax',
        "WIND_DIR_COS_at_Tmax",
        # 'WIND_DIR_COS_at_Tmax',
        # 'Rs_mean',
        "Rs_mean",
    ],
    "t_mean": [
        "T2M_mean",
        # 'T2M_at_Tmax',
        "WIND_SPEED_mean",
        # 'WIND_SPEED_at_Tmax',
        "WIND_DIR_SIN_mean",
        # 'WIND_DIR_SIN_at_Tmax',
        "WIND_DIR_COS_mean",
        # 'WIND_DIR_COS_at_Tmax',
        # 'Rs_mean',
        "Rs_mean",
    ],
}

NUMBER_OF_PREDICTOR_CHANNELS_SHARED = 19


def get_number_of_predictor_channels(target_var):
    return len(METEO_VAR_NAMES[target_var]) + NUMBER_OF_PREDICTOR_CHANNELS_SHARED
