# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

import pickle
from pathlib import Path

import mlflow
import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from urbnet.belgium.dataset import JITDataset, compute_normalization
from urbnet.belgium.settings import (
    NUMBER_OF_PREDICTOR_CHANNELS,
    NUMBER_OF_TARGET_CHANNELS,
)
from urbnet.utils import mlflow_available, write_normalization_data
from urbnet.visualize import run_example_predictions


class SimpleCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleCNN, self).__init__()
        # We're setting padding to 0 to avoid any padding artifacts
        # This reduces the output size by 2 pixels on each side on every convolution
        self.num_conv_layers = 5
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 64, kernel_size=3, padding=0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, out_channels, kernel_size=3, padding=0),
            # No BN or activation on final layer; assumed output is raw
        )

    def forward(self, x):
        return self.net(x)

    def trim_output(self, y):
        # Utility function to trim output because of padding=0
        trim = self.num_conv_layers
        return y[:, :, trim:-trim, trim:-trim]  # N, C , H, W


def train_vectorborne(
    regions_train: list[str],
    target_var: str,
    nb_patches_train: int,
    nb_patches_norm: int,
    regions_val: list[str],
    nb_patches_val: int,
    patch_size: int,
    start_year: int,
    end_year: int,
    learning_rate: float,
    batch_size: int,
    nb_epochs: int,
    version: str,
    output_folder_run: Path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    dataset_args = dict(
        patch_size=patch_size,
        start_year=start_year,
        end_year=end_year,
    )

    logger.info(f"Computing normalization on {nb_patches_norm} patches")
    norm_data = compute_normalization(
        regions=regions_train,
        target_var=target_var,
        nb_patches=nb_patches_norm,
        **dataset_args,  # type: ignore
        x_channels=[
            0,
            3,
            4,
            5,
            6,
            7,
        ],
        y_channels=[0],
        rng_seed=42,
    )
    write_normalization_data(norm_data, output_folder_run)

    logger.info("Configuring training and validation dataset")
    g = torch.Generator()
    g.manual_seed(42)

    train_dataset = JITDataset(
        regions=regions_train,
        target_var=target_var,
        nb_patches=nb_patches_train,
        **dataset_args,  # type: ignore
        norm_data=norm_data,
        rng_seed=42,
        type="train",
    )

    val_dataset = JITDataset(
        regions=regions_val,
        target_var=target_var,
        nb_patches=nb_patches_val,
        norm_data=norm_data,
        **dataset_args,  # type: ignore
        rng_seed=43,
        type="val",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        worker_init_fn=lambda wid: logger.info(f"Train worker {wid} started"),
        generator=g,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        worker_init_fn=lambda wid: logger.info(f"Validation worker {wid} started"),
        generator=g,
        pin_memory=True,
    )

    ###########
    # BUILD CNN model
    ###########

    logger.info("Defining and compiling neural network model (PyTorch)")

    model = SimpleCNN(NUMBER_OF_PREDICTOR_CHANNELS, NUMBER_OF_TARGET_CHANNELS).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_params}")
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
        model.train()
        running_loss = 0.0
        running_mae = 0.0
        for xb, yb in tqdm(
            train_dataloader,
            desc=f"Training epoch {epoch + 1}/{nb_epochs}",
            total=len(train_dataloader),
        ):
            xb = xb.to(device)
            yb = model.trim_output(yb.to(device))
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
            running_mae += compute_mae(outputs, yb) * xb.size(0)
        epoch_loss = running_loss / len(train_dataset)
        epoch_mae = running_mae / len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        with torch.no_grad():
            for xb, yb in tqdm(
                val_dataloader,
                desc=f"Validation epoch {epoch + 1}/{nb_epochs}",
                total=len(val_dataloader),
            ):
                xb = xb.to(device)
                yb = model.trim_output(yb.to(device))
                outputs = model(xb)
                loss = criterion(outputs, yb)
                val_loss += loss.item() * xb.size(0)
                val_mae += compute_mae(outputs, yb) * xb.size(0)
        epoch_val_loss = val_loss / len(val_dataset)
        epoch_val_mae = val_mae / len(val_dataset)

        logger.info(
            f"Epoch {epoch + 1}/{nb_epochs} - "
            f"loss: {epoch_loss:.4f} - mae: {epoch_mae:.4f} - "
            f"val_loss: {epoch_val_loss:.4f} - val_mae: {epoch_val_mae:.4f}"
        )

        model_path = output_folder_run / f"{version}_T2M_model_state_dict_epoch_{epoch}.pt"
        logger.info(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)

        # Log training and validation loss over nb_epochs to mlflow
        if mlflow_available():
            mlflow.log_metric("epoch_training_loss", epoch_loss, step=epoch)
            mlflow.log_metric("epoch_validation_loss", epoch_val_loss, step=epoch)
            mlflow.log_metric("epoch_training_mae", epoch_mae, step=epoch)
            mlflow.log_metric("epoch_validation_mae", epoch_val_mae, step=epoch)

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
        nb_batches=2,
        norm_data=norm_data,
        version=version,
        output_folder_run=output_folder_run,
        device=device,
        type="train",
    )

    run_example_predictions(
        model=model,
        dataloader=val_dataloader,
        nb_batches=2,
        norm_data=norm_data,
        version=version,
        output_folder_run=output_folder_run,
        device=device,
        type="val",
    )
