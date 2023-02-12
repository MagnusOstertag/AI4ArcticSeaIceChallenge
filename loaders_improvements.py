#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pytorch Dataset class for training. Function used in train.py.
   Extends the class to return also metadata of each scene on which training is performed."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '1.0.0'
__date__ = '2022-10-17'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import copy
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import pandas as pd

# -- Proprietary modules -- #


class AI4ArcticChallengeDataset(Dataset):
    """Pytorch dataset for loading batches of patches of scenes from the ASID V2 data set."""

    def __init__(self, options, files, get_metadata=False):
        self.options = options
        self.files = files
        self.metadata_flag = get_metadata

        # Channel numbers in patches, includes reference channel.
        self.patch_c = len(self.options['train_variables']) + len(self.options['charts'])

    def __len__(self):
        """
        Provide number of iterations per epoch. Function required by Pytorch dataset.

        Returns
        -------
        Number of iterations per epoch.
        """
        return self.options['epoch_len']
    
    
    def random_crop(self, scene):
        """
        Perform random cropping in scene.

        Parameters
        ----------
        scene :
            Xarray dataset; a scene from ASID3 ready-to-train challenge dataset.

        Returns
        -------
        patch :
            Numpy array with shape (len(train_variables), patch_height, patch_width). None if empty patch.
        """
        patch = np.zeros((len(self.options['full_variables']) + len(self.options['amsrenv_variables']),
                          self.options['patch_size'], self.options['patch_size']))
        
        # Get random index to crop from.
        row_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[0] - self.options['patch_size'])
        col_rand = np.random.randint(low=0, high=scene['SIC'].values.shape[1] - self.options['patch_size'])
        # Equivalent in amsr and env variable grid.
        amsrenv_row = row_rand / self.options['amsrenv_delta']
        amsrenv_row_dec = int(amsrenv_row - int(amsrenv_row))  # Used in determining the location of the crop in between pixels.
        amsrenv_row_index_crop = amsrenv_row_dec * self.options['amsrenv_delta'] * amsrenv_row_dec
        amsrenv_col = col_rand / self.options['amsrenv_delta']
        amsrenv_col_dec = int(amsrenv_col - int(amsrenv_col))
        amsrenv_col_index_crop = amsrenv_col_dec * self.options['amsrenv_delta'] * amsrenv_col_dec
        
        # - Discard patches with too many meaningless pixels (optional).
        if np.sum(scene['SIC'].values[row_rand: row_rand + self.options['patch_size'], 
                                      col_rand: col_rand + self.options['patch_size']] != self.options['class_fill_values']['SIC']) > 1:
            
            # Crop full resolution variables.
            patch[0:len(self.options['full_variables']), :, :] = scene[self.options['full_variables']].isel(
                sar_lines=range(row_rand, row_rand + self.options['patch_size']),
                sar_samples=range(col_rand, col_rand + self.options['patch_size'])).to_array().values
            # Crop and upsample low resolution variables.
            patch[len(self.options['full_variables']):, :, :] = torch.nn.functional.interpolate(
                input=torch.from_numpy(scene[self.options['amsrenv_variables']].to_array().values[
                    :, 
                    int(amsrenv_row): int(amsrenv_row + np.ceil(self.options['amsrenv_patch'])),
                    int(amsrenv_col): int(amsrenv_col + np.ceil(self.options['amsrenv_patch']))]
                ).unsqueeze(0),
                size=self.options['amsrenv_upsample_shape'],
                mode=self.options['loader_upsampling']).squeeze(0)[
                :,
                int(np.around(amsrenv_row_index_crop)): int(np.around(amsrenv_row_index_crop + self.options['patch_size'])),
                int(np.around(amsrenv_col_index_crop)): int(np.around(amsrenv_col_index_crop + self.options['patch_size']))].numpy()

        # In case patch does not contain any valid pixels - return None.
        else:
            patch = None
        if self.metadata_flag:
            return patch, row_rand, col_rand
        return patch


    def prep_dataset(self, patches):
        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W].

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        # Convert training data to tensor.
        x = torch.from_numpy(patches[:, len(self.options['charts']):]).type(torch.float)

        # Store charts in y dictionary.
        y = {}
        for idx, chart in enumerate(self.options['charts']):
            y[chart] = torch.from_numpy(patches[:, idx]).type(torch.long)

        return x, y


    def __getitem__(self, idx):
        """
        Get batch. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        mask : Dict
            Dictionary with 3D torch tensors for each chart; mask for training data x.
        metadata : pd.DataFrame
            Pandas df with metadata for each patch. Only if self.metadata_flag is True
        """
        # Placeholder to fill with data.
        patches = np.zeros((self.options['batch_size'], self.patch_c,
                            self.options['patch_size'], self.options['patch_size']))
        sample_n = 0
        metadata_batch = pd.DataFrame(columns=['sentinel_mission_identifier',
                                        'image_acquisition_start_date',
                                        'image_acquisition_start_date_year', 'image_acquisition_start_date_month', 'image_acquisition_start_date_hour',
                                        'row_rand', 'col_rand', 'sample_n',
                                        'icechart_provider', 'location'])

        # Continue until batch is full.
        while sample_n < self.options['batch_size']:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()

            # - Load scene
            scene = xr.open_dataset(os.path.join(self.options['path_to_processed_data'], self.files[scene_id]))
            # - Extract patches
            try:
                if self.metadata_flag:
                    scene_patch, row_rand, col_rand = self.random_crop(scene)
                else:
                    scene_patch = self.random_crop(scene)
            except:
                print(f"Cropping in {self.files[scene_id]} failed.")
                print(f"Scene size: {scene['SIC'].values.shape} for crop shape: ({self.options['patch_size']}, {self.options['patch_size']})")
                print('Skipping scene.')
                continue
            
            if scene_patch is not None:
                # -- Stack the scene patches in patches
                patches[sample_n, :, :, :] = scene_patch
                sample_n += 1  # Update the index.

                if self.metadata_flag:
                    # -- Add metadata to metadata_batch
                    file_name = scene.attrs['original_id']
                    file_name_split = file_name.split('_')


                    sentinel_mission_identifier = file_name_split[0]
                    image_acquisition_start_date = file_name_split[4]
                    image_acquisition_start_date = pd.to_datetime(image_acquisition_start_date, format='%Y%m%dT%H%M%S')
                    image_acquisition_start_date_year = image_acquisition_start_date.year
                    image_acquisition_start_date_month = image_acquisition_start_date.month
                    image_acquisition_start_date_hour = image_acquisition_start_date.hour
                    icechart_provider = str(scene.attrs['ice_service'])
                    if icechart_provider == 'cis':
                        location = file_name_split[11]
                    elif icechart_provider == 'dmi':
                        location = file_name_split[12]

                    metadata_sample = pd.Series({'sentinel_mission_identifier': sentinel_mission_identifier,
                                                'image_acquisition_start_date': image_acquisition_start_date,
                                                'image_acquisition_start_date_year': image_acquisition_start_date_year,
                                                'image_acquisition_start_date_month': image_acquisition_start_date_month,
                                                'image_acquisition_start_date_hour': image_acquisition_start_date_hour,
                                                'row_rand': row_rand,
                                                'col_rand': col_rand,
                                                'sample_n': sample_n,
                                                'icechart_provider': icechart_provider,
                                                'location': location})
                    metadata_batch = pd.concat([metadata_batch, metadata_sample.to_frame().T], ignore_index=True)

        # Prepare training arrays
        x, y = self.prep_dataset(patches=patches)

        # - calculate mask
        masks = {}
        for chart in self.options['charts']:
            masks[chart] = (y[chart] == self.options['class_fill_values'][chart]).squeeze()

        if self.metadata_flag:
            return x, y, masks, metadata_batch
        else:
            return x, y, masks, None


class AI4ArcticChallengeTestDataset(Dataset):
    """Pytorch dataset for loading full scenes from the ASID ready-to-train challenge dataset for inference."""

    def __init__(self, options, files, test=False, get_metadata=False):
        self.options = options
        self.files = files
        self.test = test
        self.metadata_flag = get_metadata

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)

    def prep_scene(self, scene):
        """
        Upsample low resolution to match charts and SAR resolution. Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        scene :

        Returns
        -------
        x :
            4D torch tensor, ready training data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        """
        x = torch.cat((torch.from_numpy(scene[self.options['sar_variables']].to_array().values).unsqueeze(0),
                      torch.nn.functional.interpolate(
                          input=torch.from_numpy(scene[self.options['amsrenv_variables']].to_array().values).unsqueeze(0),
                          size=scene['nersc_sar_primary'].values.shape,
                          mode=self.options['loader_upsampling'])),
                      axis=1)
        
        if not self.test:
            y = {chart: scene[chart].values for chart in self.options['charts']}

        else:
            y = None
        
        return x, y

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            Dict with 3D torch tensors for each reference chart; reference inference data for x. None if test is true.
        masks :
            Dict with 2D torch tensors; mask for each chart for loss calculation. Contain only SAR mask if test is true.
        name : str
            Name of scene.
        metadata: pd.DataFrame
            Pandas df with metadata for each scene. Only if self.metadata is True.
        """
        scene = xr.open_dataset(os.path.join(self.options['path_to_processed_data'], self.files[idx]))

        metadata_scene = pd.DataFrame(columns=['sentinel_mission_identifier',
                                'image_acquisition_start_date',
                                'image_acquisition_start_date_year', 'image_acquisition_start_date_month', 'image_acquisition_start_date_hour',
                                'row_rand', 'col_rand', 'sample_n',
                                'icechart_provider', 'location'])

        x, y = self.prep_scene(scene)
        name = self.files[idx]
        
        if not self.test:
            masks = {}
            for chart in self.options['charts']:
                masks[chart] = (y[chart] == self.options['class_fill_values'][chart]).squeeze()
                
        else:
            masks = (x.squeeze()[0, :, :] == self.options['train_fill_value']).squeeze()

        # - get metadata
        if self.metadata_flag:
            file_name = scene.attrs['original_id']
            file_name_split = file_name.split('_')


            sentinel_mission_identifier = file_name_split[0]
            image_acquisition_start_date = file_name_split[4]
            image_acquisition_start_date = pd.to_datetime(image_acquisition_start_date, format='%Y%m%dT%H%M%S')
            image_acquisition_start_date_year = image_acquisition_start_date.year
            image_acquisition_start_date_month = image_acquisition_start_date.month
            image_acquisition_start_date_hour = image_acquisition_start_date.hour
            icechart_provider = str(scene.attrs['ice_service'])
            if icechart_provider == 'cis':
                location = file_name_split[11]
            elif icechart_provider == 'dmi':
                location = file_name_split[12]

            # -- non-useful metadata
            row_rand = np.nan
            col_rand = np.nan
            sample_n = np.nan

            metadata_scene = pd.DataFrame({'sentinel_mission_identifier': sentinel_mission_identifier,
                                        'image_acquisition_start_date': image_acquisition_start_date,
                                        'image_acquisition_start_date_year': image_acquisition_start_date_year,
                                        'image_acquisition_start_date_month': image_acquisition_start_date_month,
                                        'image_acquisition_start_date_hour': image_acquisition_start_date_hour,
                                        'row_rand': row_rand,
                                        'col_rand': col_rand,
                                        'sample_n': sample_n,
                                        'icechart_provider': icechart_provider,
                                        'location': location}, index=[0])

        if self.metadata_flag and not self.test:
            return x, y, masks, name, metadata_scene
        elif not self.test:
            return x, y, masks, name, None
        return x, y, masks, name


def get_variable_options(train_options: dict):
    """
    Get amsr and env grid options, crop shape and upsampling shape.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.
    
    Returns
    -------
    train_options: dict
        Updated with amsrenv options.
    """
    train_options['amsrenv_delta'] = 50 / (train_options['pixel_spacing'] // 40)
    train_options['amsrenv_patch'] = train_options['patch_size'] / train_options['amsrenv_delta']
    train_options['amsrenv_patch_dec'] = int(train_options['amsrenv_patch'] - int(train_options['amsrenv_patch']))
    train_options['amsrenv_upsample_shape'] = (int(train_options['patch_size'] + \
                                                   train_options['amsrenv_patch_dec'] * \
                                                   train_options['amsrenv_delta']),
                                               int(train_options['patch_size'] +  \
                                                   train_options['amsrenv_patch_dec'] * \
                                                   train_options['amsrenv_delta']))
    train_options['sar_variables'] = [variable for variable in train_options['train_variables'] \
                                      if 'sar' in variable or 'map' in variable]
    train_options['full_variables'] = np.hstack((train_options['charts'], train_options['sar_variables']))
    train_options['amsrenv_variables'] = [variable for variable in train_options['train_variables'] \
                                          if 'sar' not in variable and 'map' not in variable]
    
    return train_options
                                               
