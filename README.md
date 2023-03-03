# AI4ArcticSeaIceChallenge

With `pip` the required python packages can be installed with `requirements.txt`.

In the `.env` the above are installed, for the parameters:
- `AI4ARCTIC_ENV` = '.'
- set and use the global path for `AI4ARCTIC_DATA`

## Hot to run

### Data aquisation and preprocessing
The data is not included in the repository. To download the data, run the following command in the data directory of the repository:

```
# TESTSET
$ wget 'https://data.dtu.dk/ndownloader/articles/21762830/versions/1' -O temp.zip
$ unzip temp.zip
$ rm temp.zip

# TRAINSET
$ wget 'https://data.dtu.dk/ndownloader/articles/21316608/versions/3' -O temp.zip
$ unzip temp.zip
$ rm temp.zip
```

### Data split
The data is split into training and test set, where the training set is split into training and validation set. The validatoin split is randomly chosen (with seed). The split can be configured in the train.py script. Test set containes no label information and the predicions of the model are submitted to the challenge.

### Model selection
The model can be selected by specifying the model name in the training_options. So far, the following models are implemented:
- unet
- unet_improved
- unet_transfer
- unet_transformers
- unet_attention

If further models are implemented, they can be added to the train_options dictionary in the train.py script.

### Data loaders
The customized datasets and dataloader can be found in loaders.py. It randomly crops the data with a fixed patch size for training set and uses whole scenes for the validation and test set. Since loading the whole scene allocated a lot of memory, especially for more complex models like the unet_attention.py, we imlemented a method where 

### Training
The training can be started by executing the train.py script. It can be configured within the script by specifying arguments in training_options. The training options are well explained within the script, with a special focus on specific options for slow hardware.
The progess is logged to the mlflow server and best model, according to the combined score, is saved to the models directory. The model is saved as a pytorch state dict. The models directory also contains the training ooptions as pickle file and other relevat information.
Our logs are also available.

Next to the train.py file, we also have a a jupyter notebook with identlical use.


### Evaluation
We evaluate after every epoch. If additional evaluation is needed, we can load an arbitrary model and evaluate it in the eval.py file.

### MLFlow
Nice tool, also works on local machine.

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
