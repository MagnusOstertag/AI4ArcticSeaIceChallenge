# AI4ArcticSeaIceChallenge

Following packages and versions were used to develop and test the code along with the dependancies installed with pip:
- python==3.9.11
- jupyterlab==3.4.5
- xarray==2022.10.0
- netCDF4==1.6.1
- numpy==1.23.2
- matplotlib==3.6.1
- torch==1.12.1+cu116
- tqdm==4.64.1
- sklearn==0.0
- ipywidgets==8.0.2

In the `venv` the above are installed, for the parameters:
- `AI4ARCTIC_ENV` = '.'
- use the global path for `AI4ARCTIC_DATA`

See the launcher for the `quickstart mlflow`. I have added the `mlflow` code to our quickstart script. 

## Runs

1. Pizza Marinara - 4-lvl u-net: `'unet_conv_filters': [16, 32, 32, 32]` (without mlflow)
2. Pizza Margaritha - 8-lvl u-net: `'unet_conv_filters': [16, 32, 32, 32, 32, 32, 32, 32],` (did only train till epoch 40 because of mlflow error)

## Problems

- test upload would not run because the kernel dies. You have to comment out the plotting code as there is a strange memory leak in matplotlib `cbook`.
- mlflow error `OperationError`

## Ideas

- statistics of the train and the test datasets based on their file-names
- early stopping

### TODO

- handle as a regression problem what is a regression problem: SIC
- mlflow error
- understand under which conditions the predictions are the worst
- upload this as an artifact
- transfer learning
- based on the different train and test distributions, featureize and reweight


