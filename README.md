# AI4ArcticSeaIceChallenge

With `pip` the required python packages can be installed with `requirements.txt`.

In the `.env` the above are installed, for the parameters:
- `AI4ARCTIC_ENV` = '.'
- set and use the global path for `AI4ARCTIC_DATA`

## Runs

1. `Pizza Marinara` - 4-lvl u-net: `'unet_conv_filters': [16, 32, 32, 32]` (without mlflow)
2. `Pizza Margaritha` - 8-lvl u-net: `'unet_conv_filters': [16, 32, 32, 32, 32, 32, 32, 32]` (did only train till epoch 40 because of mlflow error)
3. `Pizza Basilico` - 8-lvl u-net: `'unet_conv_filters': [16, 32, 32, 32, 32, 32, 32, 32]` but with a lot more epochs (it has to be better than Marinara!)
   1. `_bruciato` is it after 100 epochs, the other one is the best model on the test score
4. `Pizza Quattro Formaggi` - 8-lvl u-net with transfer learning from `Pizza Marinara`, regression loss, diagnostics output and reweighting sampler: `75 epochs`, `'unet_conv_filters': [16, 32, 32, 32, 32, 32, 32, 32]`
5. `Pizza Quattro Stagioni` - 6-lvl u-net with transfer learning from `Pizza Marinara` (half the learning rate), regression loss, diagnostics output and reweighting sampler: `40 epochs`, `'unet_conv_filters': [16, 32, 32, 32, 32, 32]`
   * simplified version of `Pizza Quattro Formaggi`
6. TODO: `Pizza Tre Stagioni` - 6-lvl u-net with transfer learning from `Pizza Marinara` (half the learning rate), classification loss, diagnostics output and reweighting sampler: `40 epochs`, `'unet_conv_filters': [16, 32, 32, 32, 32, 32]`
   * see the effect of classification vs regression loss

## Cluster

See the launcher for the `quickstart mlflow`. I have added the `mlflow` code to our quickstart script.

To connect to the `github` repository on the `EOxHub` it is necessary to run the following two commands every time a new session is started:

- `eval "$(ssh-agent -s)"`
- `ssh-add ~/.ssh/id_ed25519_autoice`

## Problems

- test upload would not run because the kernel dies. You have to comment out the plotting code as there is a strange memory leak in matplotlib `cbook`.
- mlflow error `OperationError`, maybe if two browsers access the cluster
- the shape of the output is super strange: `(batch_size, 12, patch_size, patch_size)`

## Ideas

- statistics of the train and the test datasets based on their file-names
- early stopping
- use a regression loss (not only as a metric) for the sea ice concentration (SIC)
- transfer learning
- understand under which conditions the predictions are the worst
- upload error per condition as an artifact
- based on the different train and test distributions, featureize and resample
- attention u-net, multi-head attention

### Timetable

* -> 22.02.: all ToDos done, most training done, mail @Lombardi
* approx. 28.02.: presentation
