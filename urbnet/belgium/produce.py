# Copyright (c) 2026 (VITO 2026)
#
# This project is licensed under the European Union Public Licence,
# version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.

from datetime import datetime, timezone

import numpy as np
import torch
from loguru import logger

from urbnet.belgium.dataset import get_inference_input
from urbnet.belgium.model import SimpleCNN
from urbnet.belgium.settings import get_output_folder
from urbnet.utils import (
    load_model_weights,
    read_norm_data,
)
from urbnet.visualize import visualize_bands, visualize_prediction


def produce(target_var: str, version: str, time: datetime):
    run_output_folder = get_output_folder(target_var) / version
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # NOTE: Model architecture assumed fixed and known (we're only loading model weights)
    logger.info("Constructing model")
    model = SimpleCNN(in_channels=24, out_channels=1).to(device)
    logger.info("Downloading model weights and loading in")
    state_dict = load_model_weights(
        run_output_folder=run_output_folder,
        version=version,
        device=device,
        suffix=".pt",
    )
    model.load_state_dict(state_dict)

    # Put the model in eval mode
    model.eval()

    X_np = get_inference_input(time=time, target_var=target_var)

    visualize_bands(
        X_np,
        filepath=run_output_folder / f"inference_input_{time.strftime('%Y%m%d')}.png",
    )
    trim = model.num_conv_layers
    H, W = X_np.shape[1], X_np.shape[2]
    y_np = np.full((H, W), np.nan)

    norm_data = read_norm_data(run_output_folder)

    for ch, mean in norm_data["x_means"].items():
        X_np[ch, :, :] = (X_np[ch, :, :] - mean) / norm_data["x_stds"][ch]

    X_t = torch.from_numpy(X_np).float().to(device).unsqueeze(0)
    with torch.no_grad():
        ypred = model(X_t)
        ypred_denorm = ypred.cpu().numpy() * norm_data["y_stds"][0] + norm_data["y_means"][0]
        # Check that the resulting tile is trimmed as expected
        assert ypred_denorm.shape == (
            1,
            1,
            X_t.shape[2] - 2 * trim,
            X_t.shape[3] - 2 * trim,
        )
        y_np[trim:-trim, trim:-trim] = ypred_denorm[0, 0, :, :]

    logger.info("Production ready.")
    visualize_prediction(
        y_np,
        output_file=str(run_output_folder / f"inference_{time.strftime('%Y%m%d')}.png"),
        time=time,
    )


if __name__ == "__main__":
    time = datetime(2024, 6, 15, tzinfo=timezone.utc)
    produce(target_var="t_min", version="v005", time=time)
