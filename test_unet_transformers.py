# -- Built-in modules -- #
import gc
import os
import sys

# -- Third-part modules -- #
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from tqdm import tqdm  # Progress bar
import mlflow
import pickle
# import mlflow.pytorch

# --Proprietary modules -- #
from functions import chart_cbar, r2_metric, f1_metric, compute_metrics  # Functions to calculate metrics and show the relevant chart colorbar.
from loaders import AI4ArcticChallengeDataset, AI4ArcticChallengeTestDataset, get_variable_options  # Custom dataloaders for regular training and validation.
from unet import UNet  # Convolutional Neural Network model
from utils import CHARTS, SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP, SCENE_VARIABLES, colour_str


# -- Environmental variables -- #
local_computation = True

if local_computation:
    os.environ['AI4ARCTIC_DATA'] = 'data'  # Fill in directory for data location.
    #os.environ['AI4ARCTIC_ENV'] = ''  # Fill in directory for environment with Ai4Arctic get-started package. 


train_options = {
    # -- General options -- #
    'model_name': 'unet_transformers',
    'model_version': 'version_32_0',
    'model_codename': 'pizza_some',
    'test_call': False,
    'eval': True,
    'mlflow': True,
    'reproducable': False,

    # -- Model options -- #
    'model_architecture': 'unet_atention',
    'optimizer': 'adam',

    # -- Training options -- #
    'early_stopping': True,
    'path_to_processed_data': os.environ['AI4ARCTIC_DATA'],  # Replace with data directory path.
    'path_to_env': '',  # Replace with environmment directory path.
    'lr': 0.0001,  # Optimizer learning rate.
    'epochs': 20,  # Number of epochs before training stop.
    'epoch_len': 500,  # Number of batches for each epoch.
    'patch_size': 128,  # Size of patches sampled. Used for both Width and Height.
    'batch_size': 4,  # Number of patches for each batch.
    'val_patch_size': 200,  # Size of patches sampled for validation. Used for both Width and Height.
    'loader_upsampling': 'nearest',  # How to upscale low resolution variables to high resolution.
    'loss_sic': 'classification', # Loss function for SIC. 'classification' or 'regression'.
    
    # -- Data prepraration lookups and metrics.
    'train_variables': SCENE_VARIABLES,  # Contains the relevant variables in the scenes.
    'charts': CHARTS,  # Charts to train on.
    'n_classes': {  # number of total classes in the reference charts, including the mask.
        'SIC': SIC_LOOKUP['n_classes'],
        'SOD': SOD_LOOKUP['n_classes'],
        'FLOE': FLOE_LOOKUP['n_classes']
    },
    'pixel_spacing': 80,  # SAR pixel spacing. 80 for the ready-to-train AI4Arctic Challenge dataset.
    'train_fill_value': 0,  # Mask value for SAR training data.
    'class_fill_values': {  # Mask value for class/reference data.
        'SIC': SIC_LOOKUP['mask'],
        'SOD': SOD_LOOKUP['mask'],
        'FLOE': FLOE_LOOKUP['mask'],
    },

    # -- Data augmentation options -- #

    # -- Metadata options -- #
    'get_metadata': True,  # Flag used in the dataloaders to get metadata.
    'resampling': True,  # Whether to resample the training data. Otherwise, the scene is randomly sampled.
    'path_to_data': 'misc/',  # Path to the data directory.
    'visualize_distribution': True,  # Whether to visualize the difference in distribution between the training, validation and test data.
    # 'difficult_locations':['CentralEast', 'NorthAndCentralEast',],  #  'CentralWest', 'SGRDIEA', 'SGRDIMID', 'SGRDIHA'],  # Locations to oversample as they are more difficult to learn.

    # -- Validation options -- #
    'chart_metric': {  # Metric functions for each ice parameter and the associated weight.
        'SIC': {
            'func': r2_metric,
            'weight': 2,
        },
        'SOD': {
            'func': f1_metric,
            'weight': 2,
        },
        'FLOE': {
            'func': f1_metric,
            'weight': 1,
        },
    },
    'num_val_scenes': 20,  # Number of scenes randomly sampled from train_list to use in validation.
    'validation_seed': 0,  # Seed used to sample validation scenes.
    'dataloader_seed': 0,  # Seed used to sample patches from the scenes.

    # -- GPU/cuda options -- #
    'gpu_id': 0,  # Index of GPU. In case of multiple GPUs.
    'num_workers': 12,  # Number of parallel processes to fetch data.
    'num_workers_val': 0,  # Number of parallel processes during validation.
    # Num worker val needs to be 0, __getitem__ is not thread safe.

    # -- U-Net Options -- # 
    # ! For Transfoemrs, first filter must correspond to number of classes ! #
    'unet_conv_filters': [24, 8, 16, 32],     # Number of filters in the U-Net.
    'conv_kernel_size': (3, 3),  # Size of convolutional kernels.
    'conv_stride_rate': (1, 1),  # Stride rate of convolutional kernels.
    'conv_dilation_rate': (1, 1),  # Dilation rate of convolutional kernels.
    'conv_padding': (1, 1),  # Number of padded pixels in convolutional layers.
    'conv_padding_style': 'zeros',  # Style of padding.
    
    # --Transformer Options -- #
    'is_residual': True,
    'num_heads': 2,
    'bias': False,
    'dtype': torch.float32,

    # -- Transfer learning options -- #
    'transfer_learning': True, # TOCHANGE # Whether to use transfer learning.
    'transfer_model_architecture': {'unet_conv_filters': [16, 32, 32, 32],}, # Dict of the differences in the U-Net options of the model architecture.
    'transfer_model_path': 'archive/pizza_marinara',  # Path to the model to transfer from.
}

# -- Test call -- #
if train_options['test_call']:
    train_options['epochs'] = 1
    train_options['epoch_len'] = 10
    
# --Set seed for reproducibility -- #
np.random.seed(train_options['validation_seed']) # Seed used to sample validation scenes.

if train_options['reproducable']: # Set seed for reproducibility in dataloader and model.
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataloader_shuffle = False
else:
    dataloader_shuffle = True
    
# Get options for variables, amsrenv grid, cropping and upsampling.
get_variable_options = get_variable_options(train_options)


# -- Initialize data -- #
# Load training list.
with open(train_options['path_to_env'] + 'datalists/dataset.json') as file:
    train_options['train_list'] = json.loads(file.read())
# Convert the original scene names to the preprocessed names.
train_options['train_list'] = [file[17:32] + '_' + file[77:80] + '_prep.nc' for file in train_options['train_list']]
# Select a random number of validation scenes with the same seed. Feel free to change the seed.et
train_options['validate_list'] = np.random.choice(np.array(train_options['train_list']), size=train_options['num_val_scenes'], replace=False)
print('Validation scenes:', train_options['validate_list'])
# Remove the validation scenes from the train list.
train_options['train_list'] = [scene for scene in train_options['train_list'] if scene not in train_options['validate_list']]
print('Options initialised')


# -- Initialize GPU -- # 
if torch.cuda.is_available(): # Cuda
    print(colour_str('GPU available!', 'green'))
    print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
    device = torch.device(f"cuda:{train_options['gpu_id']}")

elif torch.backends.mps.is_available(): # Metal MacOS Silicon
    print(colour_str('M1 GPU available!.', 'green'))
    os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0' # Set high watermark to 0 to avoid memory issues.
    
else: # CPU
    print(colour_str('GPU not available.', 'red'))
    device = torch.device('cpu')
    
train_options['device'] = device # Add device to train options.

# -- Initialize dataset and dataloader -- #
# Training.
dataset = AI4ArcticChallengeDataset(files=train_options['train_list'], options=train_options)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=None, shuffle=dataloader_shuffle, num_workers=train_options['num_workers'], pin_memory=True, )

# Validation.
dataset_val = AI4ArcticChallengeTestDataset(options=train_options, files=train_options['validate_list'])
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=None, num_workers=train_options['num_workers_val'], shuffle=False)

print('GPU and data setup complete.')


# -- Initialize model --
if train_options['model_architecture'] == 'unet':
    from unet import UNet
    net = UNet(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_transformers':
    from unet_transfomers import TransformerUNet
    net = TransformerUNet(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_attention':
    from unet_attention import AttentionUNet
    net = AttentionUNet(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_improvements':
    from unet_improvements import ImprovementsUNet
    net = ImprovementsUNet(options=train_options).to(device)
elif train_options['model_architecture'] == 'unet_transfer':
    from unet_transfer import TransferUNet
    net = TransferUNet(options=train_options).to(device)

# -- Initialize optimizer -- #
if train_options['optimizer'] == 'adam':
    optimizer = torch.optim.Adam(list(net.parameters()), lr=train_options['lr'])
torch.backends.cudnn.benchmark = True  # Selects the kernel with the best performance for the GPU and given input size.

# -- Initialize loss functions -- #
# Loss functions to use for each sea ice parameter.
# The ignore_index argument discounts the masked values, ensuring that the model is not using these pixels to train on.
# It is equivalent to multiplying the loss of the relevant masked pixel with 0.
loss_functions = {chart: torch.nn.CrossEntropyLoss(ignore_index=train_options['class_fill_values'][chart]) \
                                                   for chart in train_options['charts']}

# -- Initialize metrics -- #
if train_options['get_metadata']:
    import pandas as pd
    metadata = pd.DataFrame(columns=['sentinel_mission_identifier',
                                    'image_acquisition_start_date',
                                    'image_acquisition_start_date_year', 'image_acquisition_start_date_month', 'image_acquisition_start_date_hour',
                                    'row_rand', 'col_rand', 'sample_n',
                                    'icechart_provider', 'location',
                                    'epoch_no', 'type', 'score_combined',
                                    'loss_SIC', 'loss_SOD', 'loss_FLOE', 'loss_combined',
                                    'score_SIC', 'ice_characteristcs_SIC',
                                    'score_SOD', 'ice_characteristcs_SOD',
                                    'score_FLOE', 'ice_characteristcs_FLOE',])
    metadata_path = os.path.join(train_options['path_to_env'], f"metadata_runs/{train_options['model_codename']}_metadata.csv")
    metadata.to_csv(metadata_path, index=False)


# -- Print -- #
print('Model',train_options['model_name'],'setup complete.')
print('Model version',train_options['model_version'],'initialised.')


# -- MLFlow -- #
if not local_computation:
    ## setting up the sqlite database for tracking of experiments in MLflow
    mlflow.set_tracking_uri('sqlite:///' + os.path.expanduser(os.environ["MLFLOW_BACKEND_STORE_PATH"]))
    os.path.expanduser(os.environ["MLFLOW_BACKEND_STORE_PATH"])

experiment = mlflow.set_experiment(train_options['model_name'] + '_' + train_options['model_version'])
experiment.experiment_id


# -- Training -- #
best_combined_score = 0  # Best weighted model score.

if train_options['early_stopping']:
    from utils.early_stopper import EarlyStopper
    early_stopper = EarlyStopper(patience = 4, min_delta=0.3)

with mlflow.start_run() as run: # Start MLFlow run.
    mlflow.log_params(train_options['chart_metric']) # Log the chart metric.
    for epoch in tqdm(iterable=range(train_options['epochs']), position=0): # Loops through epochs.
        gc.collect()  # Collect garbage to free memory.
        loss_sum = torch.tensor([0.])  # To sum the batch losses during the epoch.
        net.train()  # Set network to evaluation mode.

        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader, total=train_options['epoch_len'], colour='red', position=0)):
            torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            loss_batch = 0  # Reset from previous batch.

            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():
                # - Forward pass. 
                output = net(batch_x)

                # - Calculate loss.
                for chart in train_options['charts']:
                    loss_batch += loss_functions[chart](input=output[chart], target=batch_y[chart].to(device))
                    mlflow.log_metric(key="chart_loss", value=loss_batch)
                    
                
                    
            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            # - Backward pass.
            loss_batch.backward()

            # - Optimizer step
            optimizer.step()

            # - Add batch loss.
            loss_sum += loss_batch.detach().item()

            # - Average loss for displaying
            loss_epoch = torch.true_divide(loss_sum, i + 1).detach().item()
            # print('\rMean training loss: ' + f'{loss_epoch:.3f}', end='\r')
            mlflow.log_metric(key="mean_loss", value=loss_epoch)

            # - Free memory.
            del output, batch_x, batch_y 
        del loss_sum

        # -- Validation Loop -- #
        loss_batch = loss_batch.detach().item()  # For printing after the validation loop.

        # - Stores the output and the reference pixels to calculate the scores after inference on all the scenes.
        outputs_flat = {chart: np.array([]) for chart in train_options['charts']}
        inf_ys_flat = {chart: np.array([]) for chart in train_options['charts']}
        
        # -- Store training options -- #         
        if not os.path.exists('models/'+train_options['model_name']+'/'+train_options['model_version']):
                os.makedirs('models/'+train_options['model_name']+'/'+train_options['model_version'])
        else:
            print('Model version already exists. Overwriting.')
        
        pickle.dump(obj=train_options, file=open('models/'+train_options['model_name']+'/'+train_options['model_version']+'/train_options.pkl', 'wb'))
            
        
        net.eval()  # Set network to evaluation mode.
        # gc.collect()  # Collect garbage to free memory.
        loss_val = 0  # To sum the batch losses during the epoch.
        # - Loops though scenes in queue.
        for inf_x, inf_y, masks, name in tqdm(iterable=dataloader_val, total=len(train_options['validate_list']), colour='green', position=0):
            torch.cuda.empty_cache()
            gc.collect()

            loss_val_chart = 0  # Reset from previous batch.
            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                inf_x = inf_x.to(device, non_blocking=True)
                output = net(inf_x)

            
            # - Final output layer, and storing of non masked pixels.
            for chart in train_options['charts']:
                output[chart] = torch.argmax(output[chart], dim=1).squeeze().cpu().numpy()
                outputs_flat[chart] = np.append(outputs_flat[chart], output[chart][~masks[chart]])
                inf_ys_flat[chart] = np.append(inf_ys_flat[chart], inf_y[chart][~masks[chart]].numpy())
                # for potential loss function
                # loss_val_chart += loss_functions[chart](input=torch.Tensor(outputs_flat[chart]), target=torch.Tensor(inf_ys_flat[chart]))
                # mlflow.log_metric(key="chart_loss_val", value=loss_val_chart)
                
            # loss_val += loss_val_chart.detach().item()
            
            
            # torch.cuda.empty_cache()
            
            del inf_x, inf_y, masks, output  # Free memory.
            # torch.cuda.empty_cache()
            # gc.collect()

        # loss_val_mean = torch.true_divide(loss_val, len(train_options['validate_list'])).detach().item()
        # mlflow.log_metric(key="mean_loss_val", value=loss_val_mean)
        # - Compute the relevant scores.
        combined_score, scores = compute_metrics(true=inf_ys_flat, pred=outputs_flat, charts=train_options['charts'],
                                                    metrics=train_options['chart_metric'])

        print("")
        print(f"Final batch loss: {loss_batch:.3f}")
        print(f"Epoch {epoch} score:")
        for chart in train_options['charts']:
            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")
        print(f"Combined score: {combined_score}%")
        
        mlflow.log_metric(key="final_batch_loss", value=loss_batch)
        mlflow.log_metric(key="combined_score", value=combined_score)

        # If the scores is better than the previous epoch, then save the model and rename the image to best_validation.
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            train_options['best_epoch'] = epoch
            train_options['best_score'] = best_combined_score
            # Check if the directory exists, if not create it.
            
            torch.save(obj={'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch},
                            f='models/'+train_options['model_name']+'/'+train_options['model_version']+'/best_model.pt')
        # elif early_stopper(combined_score, best_combined_score):
        #     break
        
        pickle.dump(obj=train_options, file=open('models/'+train_options['model_name']+'/'+train_options['model_version']+'/train_options.pkl', 'wb'))
            
        del inf_ys_flat, outputs_flat  # Free memory.

    print("Model saved in run %s" % mlflow.active_run().info.run_uuid)

# Run it headless
# nohup /home/leonie/anani/AI4ArcticSeaIceChallenge/venv/bin/python /home/leonie/anani/AI4ArcticSeaIceChallenge/test_unet_attention.py > /home/leonie/anani/AI4ArcticSeaIceChallenge/logs/test_unet_attention.log &