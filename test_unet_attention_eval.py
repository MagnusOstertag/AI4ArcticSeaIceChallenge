# -- Built-in modules -- #
import gc
import os
import sys

# -- Environmental variables -- #
os.environ['AI4ARCTIC_DATA'] = 'data'  # Fill in directory for data location.
# os.environ['AI4ARCTIC_ENV'] = ''  # Fill in directory for environment with Ai4Arctic get-started package.

# -- Third-part modules -- #
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm  # Progress bar
# import mlflow
import pickle
# import mlflow.pytorch

# --Proprietary modules -- #
from functions import chart_cbar, r2_metric, f1_metric, \
    compute_metrics  # Functions to calculate metrics and show the relevant chart colorbar.
from loaders import AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset, \
    get_variable_options  # Custom dataloaders for regular training and validation.
from unet import UNet  # Convolutional Neural Network model
from utils import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES, colour_str


# pickle.dump(train_options, open('models/unet_attention/version_64/train_options_pickle.txt', 'wb'))
train_options = pickle.load(open('models/unet_attention/version_64/train_options_pickle.txt', 'rb'))
train_options['num_val_scenes'] = 10


# train_options = 
# Get options for variables, amsrenv grid, cropping and upsampling.
get_variable_options = get_variable_options(train_options)
# To be used in test_upload.
# %store train_options

# Load training list.
with open(train_options['path_to_env'] + 'datalists/dataset.json') as file:
    train_options['train_list'] = json.loads(file.read())
# Convert the original scene names to the preprocessed names.
train_options['train_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in
                               train_options['train_list']]
# Select a random number of validation scenes with the same seed. Feel free to change the seed.et
np.random.seed(2)
train_options['validate_list'] = np.random.choice(np.array(train_options['train_list']),
                                                  size=train_options['num_val_scenes'],
                                                  replace=False)
# Remove the validation scenes from the train list.
train_options['train_list'] = [scene for scene in train_options['train_list'] if
                               scene not in train_options['validate_list']]
print('Options initialised')

print(colour_str('GPU not available.', 'red'))
device = torch.device('cpu')

# # Get GPU resources.
# if torch.cuda.is_available():
#     print(colour_str('GPU available!', 'green'))
#     print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
#     device = torch.device(f"cuda:{train_options['gpu_id']}")

# else:
#     # print(colour_str('GPU not available.', 'red'))
#     # device = torch.device('mps')
#     # os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  #Disable memory limit
    
# else:
#     print(colour_str('GPU not available.', 'red'))
#     device = torch.device('cpu')

# Custom dataset and dataloader.
# dataset = AI4ArcticChallengeDataset(files=train_options['train_list'], options=train_options)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=True,
#                                          num_workers=train_options['num_workers'], pin_memory=True)
# - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.
dataset_val = AI4ArcticChallengeTestDataset(options=train_options,
                                            files=train_options['validate_list'])
print(sys.getsizeof(dataset_val))
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None,
                                             num_workers=train_options['num_workers_val'],
                                             shuffle=False)

print('GPU and data setup complete.')

# Example Model

from unet_attention import UNetAttention

# Setup U-Net model, adam optimizer, loss function and dataloader.
net = UNetAttention(options=train_options).to(device)
# net.state_dict = torch.load('models/unet_attention/unet_attention.pt')
net.load_state_dict(torch.load('models/unet_attention/version_64/best_model.pt')['model_state_dict'])

net.eval()  # Set network to evaluation mode.
# gc.collect()  # Collect garbage to free memory.

outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

# - Loops though scenes in queue.
for inf_x, inf_y, masks, name in tqdm(iterable=dataloader_val,
                                      total=len(train_options['validate_list']), colour='green',
                                      position=0):
    torch.cuda.empty_cache()
    # gc.collect()

    # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
    with torch.no_grad(), torch.cuda.amp.autocast():
        inf_x = inf_x.to(device, non_blocking=True)
        output = net(inf_x)

    # - Final output layer, and storing of non masked pixels.
    for chart in train_options['charts']:
        output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
        outputs_flat[chart] = np.append(outputs_flat[chart], output[chart][~masks[chart]])
        inf_ys_flat[chart] = np.append(inf_ys_flat[chart], inf_y[chart][~masks[chart]].numpy())

    # torch.cuda.empty_cache()

    del inf_x, inf_y, masks, output  # Free memory.
    # torch.cuda.empty_cache()
    # gc.collect()

# - Compute the relevant scores.
combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat,
                                         charts=train_options['charts'],
                                         metrics=train_options['chart_metric'])

print("")
for chart in train_options['charts']:
    print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")
print(f"Combined score: {combined_score}%")

del inf_ys_flat, outputs_flat  # Free memory.

# Run it headless
# nohup /home/leonie/anani/AI4ArcticSeaIceChallenge/venv/bin/python /home/leonie/anani/AI4ArcticSeaIceChallenge/test_unet_attention.py > /home/leonie/anani/AI4ArcticSeaIceChallenge/logs/test_unet_attention.log &v