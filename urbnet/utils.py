# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import os
import pickle
import tempfile
from contextlib import contextmanager, nullcontext
from functools import lru_cache

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "1000")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "6")
os.environ.setdefault("MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT", "1000")

import mlflow
import requests
import torch
from dotenv import find_dotenv, load_dotenv
from loguru import logger
from mlflow import MlflowClient
from mlflow.artifacts import download_artifacts, list_artifacts

load_dotenv(find_dotenv(), override=True)


def get_next_version(output_folder_root):
    version = 0
    while True:
        version += 1
        output_folder = output_folder_root / f"v{version:03d}"
        if not output_folder.exists():
            logger.info(f"Determined version for this run: v{version:03d}")
            return f"v{version:03d}"


def format_duration(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)

    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if seconds or not parts:
        parts.append(f"{seconds}s")

    return " ".join(parts)


@lru_cache(maxsize=None)
def mlflow_available():
    if os.getenv("USE_MLFLOW", "1") == "0":
        logger.warning("MLflow is disabled by environment variable USE_MLFLOW=0.")
        return False
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
    MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if (
        MLFLOW_TRACKING_URI is None
        or MLFLOW_TRACKING_USERNAME is None
        or MLFLOW_TRACKING_PASSWORD is None
    ):
        logger.warning("MLflow related environment variables are not set, so not using MLflow.")
        return False

    try:
        response = requests.get(f"{MLFLOW_TRACKING_URI}/health", timeout=2)
        if response.status_code == 200:
            logger.info("MLFlow server available and healthy, using MLFlow.")
            return True
        else:
            logger.warning("MLFlow server is not healthy, so not using MLFlow.")
            return False
    except Exception:
        logger.warning("MLFlow server is not reachable, so not using MLFlow.")
        return False


@contextmanager
def mlflow_run(
    version: str,
    exp_name: str,
    params,
):
    if exp_name not in {
        "TESTING",
        "HHEATMAP",
        "VECTORBORNE_TMIN",
        "VECTORBORNE_TMAX",
        "VECTORBORNE_TMEAN",
        "VECTORBORNE_RHMEAN",
    }:
        logger.warning(f"Unknown experiment name: {exp_name}. Not using MLflow.")
        with nullcontext() as ctx:
            yield ctx
        return

    if mlflow_available():
        mlflow.set_experiment(exp_name)
        logger.info(f"Starting MLflow run {version}")
        with mlflow.start_run(run_name=version) as run:
            mlflow.log_param("version", version)
            for param_name, param_val in params.items():
                mlflow.log_param(param_name, param_val)
            yield run
    else:
        with nullcontext() as ctx:
            yield ctx


def mlflow_subrun(
    version: int,
    cities: list[str],
    learning_rate: float,
    batch_size: int,
):
    if mlflow_available():
        return mlflow.start_run(
            run_name=f"v{version:03d}_{'_'.join(cities)}_lr_{learning_rate}_bs_{batch_size}",
            nested=True,
        )
    else:
        return nullcontext()


def fetch_mlflow_run(experiment_name: str, run_name: str):
    assert mlflow_available()
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"attributes.run_name='{run_name}'",
    )
    if not runs:
        raise ValueError(f"No runs found with name {run_name} in experiment 'HHEATMAP'.")
    if len(runs) > 1:
        logger.warning(f"Multiple runs found with name {run_name}. Returning the first one.")
    return runs[0]


def load_model_weights(run_output_folder, version, device, suffix=".pt"):
    best_model_epoch_path = run_output_folder / f"{version}_T2M_best_epoch.pkl"
    logger.info(f"Loading best epoch info from {best_model_epoch_path}")
    with open(best_model_epoch_path, "rb") as f:
        best_epoch = pickle.load(f)
    model_path = run_output_folder / f"{version}_T2M_model_state_dict_epoch_{best_epoch}{suffix}"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    logger.info(f"Loading model weights from {model_path}")
    return torch.load(model_path, weights_only=True, map_location=device)


def load_mlflow_model_weights(
    mlflow_run,
    device,
    suffix=".pt",
):
    artifacts = list_artifacts(run_id=mlflow_run.info.run_id)
    filtered_artifacts = [a for a in artifacts if a.path.endswith(suffix)]
    if not filtered_artifacts:
        raise ValueError(
            f"No model artifacts found with suffix {suffix} in MLflow run {mlflow_run.info.run_name}."
        )
    if len(filtered_artifacts) > 1:
        logger.warning(
            f"Multiple model artifacts found with suffix {suffix}. Returning the best one."
        )
    best_epoch = mlflow_run.data.metrics.get("best_epoch")
    if best_epoch is None:
        raise ValueError(
            f"MLflow run {mlflow_run.info.run_name} has no 'best_epoch' metric."
        )
    best_epoch = int(best_epoch)

    matching_artifacts = [
        artifact
        for artifact in filtered_artifacts
        if artifact.path.endswith(f"epoch_{best_epoch}{suffix}")
    ]
    if not matching_artifacts:
        raise ValueError(
            f"No checkpoint for best epoch {best_epoch} found in MLflow run "
            f"{mlflow_run.info.run_name}."
        )
    model_artifact = matching_artifacts[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        download_artifacts(
            run_id=mlflow_run.info.run_id, dst_path=tmpdir, artifact_path=model_artifact.path
        )
        model_path = os.path.join(tmpdir, model_artifact.path)
        return torch.load(model_path, map_location=torch.device(device), weights_only=True)


@lru_cache(maxsize=None)
def get_output_folder_for_run(root, version):
    output_folder_run = root / version
    if not output_folder_run.exists():
        logger.info(f"Creating run output folder: {output_folder_run}")
        output_folder_run.mkdir(parents=True, exist_ok=False)
    return output_folder_run


def write_normalization_data(norm_data, output_folder_run):
    norm_data_path = output_folder_run / "norm_data.pkl"
    logger.info(f"Writing normalization data to {norm_data_path}")
    with norm_data_path.open("wb") as f:
        pickle.dump(norm_data, f)
    if mlflow_available():
        mlflow.log_artifact(norm_data_path)


def read_norm_data(run_output_folder):
    norm_data_path = run_output_folder / "norm_data.pkl"
    if not norm_data_path.exists():
        raise FileNotFoundError(f"Normalization data file not found: {norm_data_path}")
    logger.info(f"Reading normalization data from {norm_data_path}")
    with norm_data_path.open("rb") as f:
        return pickle.load(f)
