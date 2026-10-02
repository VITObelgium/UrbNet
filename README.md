# UrbNet

UrbNet contains research scripts for training and running convolutional neural
networks that estimate urban climate fields from meteorological and land-surface
data. The repository currently has two workflows:

- **Belgium** predicts daily minimum, maximum, or mean temperature fields.
- **Abu Dhabi** predicts temperature, wet-bulb globe temperature (WBGT), or
	specific humidity, using circulation and day/night model options.

The code and configuration are provided here; the input datasets and trained
weights are not. Input and output roots can be configured independently with
environment variables; the expected dataset layouts are described below.

## Scientific publication

This repository accompanies a scientific publication that is currently under 
review. A link will be provided later.

## Environment

Python 3.12 or later is required. Pixi is used to manage the project and its
scientific Python dependencies, and is the supported way to install and run
UrbNet. The CPU environment is:

```bash
pixi install -e cpu
pixi run -e cpu python -c "import torch; print(torch.__version__)"
```

A plain `pip install .` is not supported: the standard Python package metadata
does not declare the full runtime dependency set. Pixi installs the project
with its scientific dependencies and selects the appropriate CPU or GPU
PyTorch package.

The default Pixi environment uses the GPU feature and is configured for CUDA
11.8. The training and inference code selects CUDA when PyTorch reports it is
available and otherwise runs on the CPU. On systems without CUDA, use the `cpu`
environment.

MLflow logging during training is optional. Set `USE_MLFLOW=0` to disable it;
otherwise, logging is enabled only when `MLFLOW_TRACKING_URI`,
`MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` are set and the
tracking server responds to its health check. A `.env` file may be used for
these values. Abu Dhabi production inference loads models from MLflow and
requires a reachable, configured tracking server.

## Data and output paths

Set these variables before starting Python to use paths on your machine:

```bash
export URBNET_BELGIUM_DATA_ROOT=/path/to/belgium-data
export URBNET_ABUDHABI_DATA_ROOT=/path/to/abudhabi-data
export URBNET_OUTPUT_ROOT=/path/to/urbnet-output
```

The Belgium data root should contain `urbsim_runs/`, `input_data/BELGIUM/`,
and the Belgium static inputs under `input_data/urbsim-config-belgium/`. The
Abu Dhabi data root should contain `urbsim/` and `input_data/` with the UAE and
AD datasets. Outputs are written under the output root. Abu Dhabi's reusable
normalization JSON and generated per-city distance-to-coast rasters are also
stored there, rather than in the input datasets.

If unset, `URBNET_BELGIUM_DATA_ROOT` and `URBNET_ABUDHABI_DATA_ROOT` default to
the existing `/projects/` data layout, while `URBNET_OUTPUT_ROOT` defaults to
`/projects/neural_net/`. Set all three variables when running outside that
layout. The roots are read when the regional settings modules are imported, so
configure them before starting the Python process.

## Workflows

Training is exposed through Python functions, not a general command-line
interface. Each call reads the regional datasets, creates the next available
version directory, trains a model, and writes normalization data, model
checkpoints, and diagnostic plots. Training can take substantial time and
requires write access to the configured output root.

### Belgium

The `urbnet.belgium.run` function accepts temperature targets `t_min`, `t_max`,
and `t_mean`. Region identifiers must be present in
`urbnet/belgium/settings.py`.

```python
from urbnet.belgium import run

version = run(
		regions_train=["antwerp_25042025", "gent_08052025"],
		target_var="t_max",
		nb_patches_train=64,
		nb_patches_norm=32,
		regions_val=["westhoek_13052025"],
		nb_patches_val=32,
		start_year=2002,
		end_year=2017,
		nb_epochs=5,
)
print(version)
```

The main training data is expected below
`<URBNET_BELGIUM_DATA_ROOT>/urbsim_runs/<region>/`, including yearly
meteorology and UrbClim NetCDF files and terrain rasters. Inference uses
additional files below `<URBNET_BELGIUM_DATA_ROOT>/input_data/BELGIUM/` and
`<URBNET_BELGIUM_DATA_ROOT>/input_data/urbsim-config-belgium/terrain/log/`.
The exact filenames and variables are defined in `urbnet/belgium/dataset.py`.

Run inference with the version returned by training and a timezone-aware
timestamp:

```python
from datetime import datetime, timezone

from urbnet.belgium import produce

produce(
		target_var="t_max",
		version=version,
		time=datetime(2024, 6, 15, tzinfo=timezone.utc),
)
```

This writes input-band and prediction plots to the version's output directory.
Inference requires the matching checkpoint and normalization file from training,
as well as the date-specific ERA5 samples and static raster inputs.

### Abu Dhabi

The `urbnet.abudhabi.run` function accepts `output_var="temperature"`,
`"wbgt"`, or `"humidity"`. City identifiers are listed in
`urbnet/abudhabi/settings.py`.

```python
from urbnet.abudhabi import run

version = run(
		cities_train=["AbuDhabi", "AlAin"],
		nb_patches_train=128,
		cities_val=["Ghayathi"],
		nb_patches_val=32,
		output_var="temperature",
)
print(version)
```

Training data is read from
`<URBNET_ABUDHABI_DATA_ROOT>/urbsim/<city>/`, along with circulation data under
`<URBNET_ABUDHABI_DATA_ROOT>/input_data/`. Inputs include meteorology and
UrbClim NetCDF files, terrain and land-surface rasters, and circulation
classifications. The loaders in `urbnet/abudhabi/dataset.py` define the
required filenames and variables.

Abu Dhabi production uses `urbnet.abudhabi.produce` and accepts the output
variable names `T2M`, `WBGT`, or `QV2M`, a target domain (`UAE` or `AD`), model
run names from MLflow, timestamps, and an operational forecast folder. It
produces a NetCDF file. This workflow requires MLflow and forecast/static inputs;
see the function signature and loaders in `urbnet/abudhabi/produce.py` and
`urbnet/abudhabi/dataset.py` before running it.

## Outputs and implementation notes

Training assigns versions such as `v001` by checking the existing output
directories. Belgium outputs are grouped under
`<URBNET_OUTPUT_ROOT>/<target_var>/`; Abu Dhabi outputs are directly under
`<URBNET_OUTPUT_ROOT>/`. The output contains normalization data, per-epoch
model weights, best-epoch metadata, and example prediction plots. Abu Dhabi
production writes its NetCDF to the operational forecast folder when its path
contains `ECMWF` or `ICON`; otherwise it writes under the model output root.

The `python -m urbnet...` examples in the modules use fixed sample arguments and
data paths; use the Python functions above for controlled runs. In particular,
the Abu Dhabi training example currently contains `AbuDhabil`, which is not in
the configured city list.

## License

Copyright (c) 2026 (VITO 2026)

This project is licensed under the European Union Public Licence,
version 1.2 (EUPL-1.2). See the LICENSE file for the full licence text.
