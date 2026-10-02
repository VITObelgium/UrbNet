# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import pickle
import time
from pathlib import Path

import mlflow
import torch
from loguru import logger
from torch.utils.data import DataLoader

from urbnet.abudhabi.dataset import JITDataset, load_normalization
from urbnet.abudhabi.settings import (
    NORMALIZATION_CHANNEL_SPEC_HUMIDITY,
    NORMALIZATION_CHANNEL_SPEC_TEMPERATURE,
    NORMALIZATION_CHANNEL_SPEC_WBGT,
    NUMBER_OF_PREDICTOR_CHANNELS_HUMIDITY,
    NUMBER_OF_PREDICTOR_CHANNELS_TEMPERATURE,
    NUMBER_OF_PREDICTOR_CHANNELS_WBGT,
)
from urbnet.utils import format_duration, mlflow_available
from urbnet.visualize import run_example_predictions


class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels):
        super(SimpleCNN, self).__init__()
        # We're setting padding to 0 to avoid any padding artifacts
        # This reduces the output size by 2 pixels on each side on every convolution
        self.num_conv_layers = 5
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 1, kernel_size=3, padding=0),
        )

    def forward(self, x):
        return self.net(x)

    def trim_output(self, y):
        # Utility function to trim output because of padding=0
        trim = self.num_conv_layers
        return y[:, :, trim:-trim, trim:-trim]


def train_abudhabi(
    cities_train: list[str],
    nb_patches_train: int,
    cities_val: list[str],
    nb_patches_val: int,
    patch_size: int,
    start_year: int,
    end_year: int,
    n_circ_class_clusters: int,
    circ_class_cluster_idx: int,
    use_individual_model_per_cluster: bool,
    use_day_night_model: bool,
    time_of_the_day: str,
    nb_epochs: int,
    learning_rate: float,
    batch_size: int,
    version: str,
    output_folder_run: Path,
    output_var: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    if output_var == "temperature":
        norm_data = load_normalization(
            x_channels=NORMALIZATION_CHANNEL_SPEC_TEMPERATURE["x_channels"],
            y_channels=NORMALIZATION_CHANNEL_SPEC_TEMPERATURE["y_channels"],
        )
    elif output_var == "wbgt":
        norm_data = load_normalization(
            x_channels=NORMALIZATION_CHANNEL_SPEC_WBGT["x_channels"],
            y_channels=NORMALIZATION_CHANNEL_SPEC_WBGT["y_channels"],
        )
    elif output_var == "humidity":
        norm_data = load_normalization(
            x_channels=NORMALIZATION_CHANNEL_SPEC_HUMIDITY["x_channels"],
            y_channels=NORMALIZATION_CHANNEL_SPEC_HUMIDITY["y_channels"],
        )
    else:
        raise ValueError(f"Unsupported output_var: {output_var}")

    # Create often used kwargs
    dataset_kwargs = dict(
        patch_size=patch_size,
        start_year=start_year,
        end_year=end_year,
        use_day_night_model=use_day_night_model,
        n_circ_class_clusters=n_circ_class_clusters,
        circ_class_cluster_idx=circ_class_cluster_idx,
        use_individual_model_per_cluster=use_individual_model_per_cluster,
        time_of_the_day=time_of_the_day,
        output_var=output_var,
    )

    logger.info("Configuring training and validation dataset")
    train_dataset = JITDataset(
        cities=cities_train,
        nb_patches=nb_patches_train,
        norm_data=norm_data,
        **dataset_kwargs,  # type: ignore
    )

    val_dataset = JITDataset(
        cities=cities_val,
        nb_patches=nb_patches_val,
        norm_data=norm_data,
        **dataset_kwargs,  # type: ignore
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda wid: logger.info(f"Train worker {wid} started"),
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,  # Get variety in the examples
        num_workers=0,
        worker_init_fn=lambda wid: logger.info(f"Validation worker {wid} started"),
        pin_memory=True,
    )

    ###########
    # BUILD CNN model
    ###########

    logger.info("Defining and compiling neural network model (PyTorch)")

    # TODO: remove hardcoded number of predictor channels
    if output_var == "temperature":
        model = SimpleCNN(NUMBER_OF_PREDICTOR_CHANNELS_TEMPERATURE).to(device)
    elif output_var == "wbgt":
        model = SimpleCNN(NUMBER_OF_PREDICTOR_CHANNELS_WBGT).to(device)
    elif output_var == "humidity":
        model = SimpleCNN(NUMBER_OF_PREDICTOR_CHANNELS_HUMIDITY).to(device)
    logger.info(model)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def compute_mae(pred, target):
        return torch.mean(torch.abs(pred - target)).item()

    early_stopping_patience = 3
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_path = None
    patience_counter = 0

    for epoch in range(nb_epochs):
        epoch_desc = f"Epoch {epoch + 1}/{nb_epochs}"
        logger.info(f"{epoch_desc}: Training on training set")
        start = time.perf_counter()

        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for xb, yb in train_dataloader:
            xb = xb.to(device)
            yb = model.trim_output(yb.to(device))
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            running_mae += compute_mae(outputs, yb) * xb.size(0)
        epoch_train_loss = running_loss / len(train_dataloader.dataset)  # type: ignore
        epoch_train_mae = running_mae / len(train_dataloader.dataset)  # type: ignore
        train_time = time.perf_counter()
        logger.info(f"{epoch_desc}: Training completed in {format_duration(train_time - start)}")

        logger.info(f"{epoch_desc}: Evaluating on validation set")
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in val_dataloader:
                xb = xb.to(device)
                yb = model.trim_output(yb.to(device))
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                val_mae += compute_mae(outputs, yb) * xb.size(0)
        epoch_val_loss = val_loss / len(val_dataloader.dataset)  # type: ignore
        epoch_val_mae = val_mae / len(val_dataloader.dataset)  # type: ignore
        val_time = time.perf_counter()  # This is a counter

        logger.info(
            f"{epoch_desc}: Evaluation completed in {format_duration(val_time - train_time)}"
        )
        logger.info(
            f"{epoch_desc}: took {format_duration(val_time - start)} - "
            f"train_loss: {epoch_train_loss:.4f} - train_mae: {epoch_train_mae:.4f} - "
            f"val_loss: {epoch_val_loss:.4f} - val_mae: {epoch_val_mae:.4f}"
        )

        model_path = output_folder_run / f"{version}_T2M_model_state_dict_epoch_{epoch}.pt"
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)

        # Log training and validation loss over nb_epochs to mlflow
        if mlflow_available():
            mlflow.log_metric("epoch_training_loss", epoch_train_loss, step=epoch)
            mlflow.log_metric("epoch_validation_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("epoch_training_mae", epoch_train_mae, step=epoch)
            mlflow.log_metric("epoch_validation_mae", epoch_val_mae, step=epoch)
            mlflow.log_artifact(str(model_path))

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_model_path = model_path
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(
                f"Validation loss did not improve, incrementing patience counter to {patience_counter}/{early_stopping_patience}"
            )
            if patience_counter >= early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {patience_counter} epochs without improvement"
                )
                break

    if mlflow_available():
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.log_metric("best_validation_loss", best_val_loss)
    logger.info(
        f"Training completed. Best validation loss: {best_val_loss:.4f} at epoch {best_epoch + 1}"
    )
    assert best_model_path is not None, "No best model path found"
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # save the best model epoch in a separate file
    best_model_epoch_path = output_folder_run / f"{version}_T2M_best_epoch.pkl"
    logger.info(f"Saving best model epoch to {best_model_epoch_path}")
    with open(best_model_epoch_path, "wb") as f:
        pickle.dump(best_epoch, f)

    run_example_predictions(
        model=model,
        dataloader=train_dataloader,
        nb_batches=4,
        norm_data=norm_data,
        version=version,
        output_folder_run=output_folder_run,
        device=device,
        type="train",
        output_var=output_var,
    )

    run_example_predictions(
        model=model,
        dataloader=val_dataloader,
        nb_batches=4,
        norm_data=norm_data,
        version=version,
        output_folder_run=output_folder_run,
        device=device,
        type="val",
        output_var=output_var,
    )
