# -- Built-in modules -- #
import gc
import os
import sys

# -- Environmental variables -- #
os.environ['AI4ARCTIC_DATA'] = 'data'  # Fill in directory for data location.
# os.environ['AI4ARCTIC_ENV'] = ''  # Fill in directory for environment with Ai4Arctic get-started package.

# -- Third-part modules -- #
import json
import numpy as np
import torch
from tqdm import tqdm  # Progress bar
import pickle

# --Proprietary modules -- #
from functions import chart_cbar, r2_metric, f1_metric, \
    compute_metrics  # Functions to calculate metrics and show the relevant chart colorbar.
from loaders import AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset, \
    get_variable_options  # Custom dataloaders for regular training and validation.
from unet import UNet  # Convolutional Neural Network model
from utils import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES, colour_str


# --- Load train options --- #
model_architecture = 'unet_attention'
model_version = 'version_64_4'

try:
    train_options = pickle.load(open('models/'+model_architecture+'/'+model_version+'/train_options.pkl', 'rb'))
except:
    print('No train options found. Please check if model description is correct.')
    
train_options['model_architecture'] = model_architecture
train_options['model_version'] = model_version
train_options['num_val_scenes'] = None
train_options['val_batch_size'] = 1
train_options['val_patch_size'] = None  # None for full image, smaller if RAM issues

# Get options for variables, amsrenv grid, cropping and upsampling.
get_variable_options = get_variable_options(train_options)


# --- Load GPU --- #
if torch.cuda.is_available():
    print(colour_str('GPU available!', 'green'))
    print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
    device = torch.device(f"cuda:{train_options['gpu_id']}")
    
elif torch.backends.mps.is_available():
    print(colour_str('M1 GPU available!.', 'green'))
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' 
    
else:
    print(colour_str('GPU not available.', 'red'))
    device = torch.device('cpu')
    
    
# -- Load dataset -- #
dataset_val = AI4ArcticChallengeTestDataset(options=train_options,
                                            files=train_options['validate_list'])
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None,
                                             num_workers=train_options['num_workers_val'],
                                             shuffle=False)
print('Number of validation scenes: ', colour_str(len(dataset_val), 'orange'))


# -- Initialize model --
if train_options['model_architecture'] == 'unet':
    from unet import UNet
    net = UNet(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_transformers':
    from unet_transfomers import TransformerUNet
    net = TransformerUNet(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_attention':
    print('Using attention U-Net')
    from unet_attention import UNetAttention
    net = UNetAttention(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_improvements':
    from unet_improvements import UNet
    net = UNet(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_transfer':
    from unet_transfer import TransferUNet
    net = TransferUNet(options=train_options).to(device)
    
# -- Load model --
model_path = 'models/'+model_architecture+'/'+model_version+'/best_model.pt'

if torch.cuda.is_available():
    net.load_state_dict(torch.load(model_path)['model_state_dict'])
    print('Model loaded from GPU.')
else:
    net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))['model_state_dict'])
    
net.eval()  # Set network to evaluation mode.
# gc.collect()  # Collect garbage to free memory.

outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}

# - Loops though scenes in queue.
for inf_x, inf_y, masks, name, _ in tqdm(iterable=dataloader_val,
                                      total=len(train_options['validate_list']), colour='green',
                                      position=0):
    torch.cuda.empty_cache()
    gc.collect()

    # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
    with torch.no_grad(), torch.cuda.amp.autocast():
        inf_x = inf_x.to(device, non_blocking=True)
        output = net(inf_x)
        del inf_x

    # - Final output layer, and storing of non masked pixels.
    for chart in train_options['charts']:
        output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
        outputs_flat[chart] = np.append(outputs_flat[chart], output[chart][~masks[chart]])
        inf_ys_flat[chart] = np.append(inf_ys_flat[chart], inf_y[chart][~masks[chart]].numpy())

    # torch.cuda.empty_cache()

    del inf_y, masks, output  # Free memory.
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

with open('models/'+model_architecture+'/'+model_version+'/eval_scores.txt', 'w') as f:
    print("", file=f)
    for chart in train_options['charts']:
        print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%", file=f)
    print(f"Combined score: {combined_score}%", file=f)

del inf_ys_flat, outputs_flat  # Free memory.