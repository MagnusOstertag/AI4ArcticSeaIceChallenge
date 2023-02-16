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
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset
import pandas as pd
import json
import gc
import glob
import copy

import matplotlib.pyplot as plt
import seaborn as sns

from utils import SIC_GROUPS, SOD_GROUPS, FLOE_GROUPS
from utils import SIC_LOOKUP, SOD_LOOKUP, FLOE_LOOKUP

# -- Proprietary modules -- #


class AI4ArcticChallengeDataset(Dataset):
    """Pytorch dataset for loading batches of patches of scenes from the ASID V2 data set."""

    def __init__(self, options, files, get_metadata=False):
        self.options = options
        self.files = files
        self.metadata_flag = get_metadata

        # set randomness
        np.random.seed(self.options['seed'])

        # Channel numbers in patches, includes reference channel.
        self.patch_c = len(self.options['train_variables']) + len(self.options['charts'])

        # Create a baseline probablity for sampling each scene.
        # The problem is handled on the scene level, no regard for the randomn cropping.
        # But this may also reduce the chance for overfitting.
        if self.options['resampling']:
            # create a pd dataframe for the files and the seed (like distributions), if it does not exist
            # thus we do not have to generate it every time
            path_to_distribution = os.path.join(self.options['path_to_data'], f"distribution_{self.options['validation_seed']}.csv")

            # get file from other seed if it exists
            files = glob.glob(os.path.join(self.options['path_to_data'], "distribution_*.csv"))
            file = [f for f in files if os.path.isfile(f)]
            file = file[0] if file != [] else None

            if os.path.isfile(path_to_distribution):
                # simply load the distribution
                self.distribution = pd.read_csv(path_to_distribution)
            elif file is not None:
                # if one such dataset exists as we only have to change the dataset column and generate new visualizations
                self.distribution = pd.read_csv(file)

                with open('datalists/testset.json', 'r') as f:
                    testset = json.load(f)
                trainset = options['train_list_extendedNames']
                valset = options['validate_list_extendedNames']

                # change the filename column
                for i in range(0, self.distribution.shape[0]):
                    if self.distribution["filename"][i] in trainset:
                        self.distribution.loc[(i, "dataset")] = 'train'
                    elif self.distribution["filename"][i] in valset:
                        self.distribution.loc[(i, "dataset")] = 'val'
                    elif self.distribution["filename"][i] in testset:
                        self.distribution.loc[(i, "dataset")] = 'test'
            else:
                # if no distribution-file exist, generate it
                self.calculate_distribution()

                # visualize the distribution and save pictures
                if self.options['visualize_distribution']:
                    self.visualize_distribution()

            # I need to do it here, because they might not be calculated already - but now they are calculated twice sometimes
            self.calculate_weights()

            if self.options['visualize_distribution']:
                self.visualize_distribution(weights=True)

            # save the distribution to a csv file
            self.distribution.to_csv(path_to_distribution)

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
            scene_id = self.sampler()

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

    def sampler(self, mode='random'):
        """
        Returns the id of a scene to sample from.
        The resampler takes the following factors into account:
        - The number of samples already from a scene divided by the size of the scene.
        - The difficulity of a scene as judged by human experts.
        - The distribution of the test set regarding geographic location, time of year # , and ice type.
        - The previous performance of the model on a scene.

        Parameters
        ----------
        mode : str
            'random' or 'importance'. Default is 'random'.

        Returns
        -------
        scene_id : int
            Id of the scene to sample from.
        """
        if not self.options['resampling']:
            return np.random.randint(low=0, high=len(self.files), size=1).item()
        elif self.options['resampling']:
            # the id is the index of the scene in self.files
            id = np.random.choice(a=self.distribution[self.distribution['dataset'] == 'train'].index,
                                  p=self.distribution[self.distribution['dataset'] == 'train']['weight'].values,
                                  size=1, replace=True).item()
            filename_of_id = self.distribution.loc[(id, 'filename')]
            filename_of_id = filename_of_id[17:32] + '_' + filename_of_id[77:80] + '_prep.nc'  # get the train_list filename
            return self.files.index(filename_of_id)

    def visualize_distribution(self, weights=False):
        """
        Visualize the distribution of the train, val and test set.

        Parameters
        ----------
        weights : bool
            If True, the right side distribution is the training set distribution weighted by the resampling weights.
            Otherwise, it is weighted by the size of the scenes.
        """
        path_to_visualization = os.path.join(self.options['path_to_data'],
                                             f"distribution_{self.options['validation_seed']}_")

        plot_args_one = dict(hue="dataset", data=self.distribution, stat='percent', common_norm=True)
        postfix = ""
        if not weights:
            plot_args_two = copy.deepcopy(plot_args_one)
            plot_args_two['legend'] = False
            plot_args_two['weights'] = 'size'
            postfix = "_size"
        else:
            total_n_samples_train = self.distribution[self.distribution['dataset'] == 'train'].shape[0]
            total_n_samples_test = self.distribution[self.distribution['dataset'] == 'test'].shape[0]

            plot_args_two = copy.deepcopy(plot_args_one)
            distribution_local = copy.deepcopy(self.distribution[self.distribution['dataset'] != 'val'])
            distribution_local.loc[distribution_local['dataset'] == 'test', 'weight'] = total_n_samples_train / total_n_samples_test
            plot_args_two['common_norm'] = False  # via definition of the above weights
            plot_args_two['data'] = distribution_local
            plot_args_two['legend'] = False
            plot_args_two['weights'] = 'weight'
            postfix = "_weight"

        # plot the distribution of the months, grouped by dataset versus testset
        # norm on the size column
        # show one plot not weighted on the size and one weighted on the size of the scenes
        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 8), dpi=400)
        sns.histplot(y="month", ax=axs[0], **plot_args_one)
        sns.histplot(y="month", ax=axs[1], **plot_args_two)
        plt.savefig(f'{path_to_visualization}month{postfix}.png')
        plt.close()

        # plot the distribution of the years, grouped by dataset versus testset
        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 8), dpi=400)
        sns.histplot(y="year", ax=axs[0], **plot_args_one)
        sns.histplot(y="year", ax=axs[1], **plot_args_two)
        plt.savefig(f'{path_to_visualization}year{postfix}.png')
        plt.close()

        # plot the distribution of the location, grouped by dataset versus testset
        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 8), dpi=400)
        sns.histplot(y="location", ax=axs[0], **plot_args_one)
        sns.histplot(y="location", ax=axs[1], **plot_args_two)
        plt.savefig(f'{path_to_visualization}location{postfix}.png')
        plt.close()

        # plot the distribution of the icechart_provider, grouped by dataset versus testset
        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 8), dpi=400)
        sns.histplot(y="icechart_provider", ax=axs[0], **plot_args_one)
        sns.histplot(y="icechart_provider", ax=axs[1], **plot_args_two)
        plt.savefig(f'{path_to_visualization}icechart_provider{postfix}.png')
        plt.close()

        # plot the distribution of the Sentinel_mission_identifier, grouped by dataset versus testset
        fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 8), dpi=400)
        sns.histplot(y="Sentinel_mission_identifier", ax=axs[0], **plot_args_one)
        sns.histplot(y="Sentinel_mission_identifier", ax=axs[1], **plot_args_two)
        plt.savefig(f'{path_to_visualization}Sentinel_mission_identifier{postfix}.png')
        plt.close()

        # collect as much garbage as possible
        del fig, axs, path_to_visualization
        plt.clf()
        plt.close()
        gc.collect()

    def calculate_weights(self):
        """
        Calculate the rescaling weights for the resampling.
        For every sample, a weight is calculated which follows the distribution of the test set.
        It is necessary because test distribution is (purposefully) altered, as experts have selected the most difficult scenes to be in the test set.

        Only weighting by the location makes the distributions in the relevant variables (month, year, icechart_provider, Sentinel_mission_identifier) already pretty similar.
        # TODO: find a better method!

        Problem: For the discrete multi-class variables, some are not represented in the test set at all. Either I have to
           - remove them from the training set
           - cannot fully represent the test set distribution.
        I opted for the second option. Each class has at least a ratio of 1/(n_classes*3) of the test set.
        """
        # pre-calulate the weights for the training
        total_n_samples_train = self.distribution[self.distribution['dataset'] == 'train'].shape[0]
        total_n_samples_test = self.distribution[self.distribution['dataset'] == 'test'].shape[0]
        self.distribution['weight'] = 1. / total_n_samples_train  # uniform for all training samples
        self.distribution.loc[self.distribution['dataset'] != 'train', 'weight'] = 0.  # no weight for the val, test set
        print(f"total_n_samples_train: {total_n_samples_train}")

        # resampling according to the distribution of the test set
        # weighting by the month
        ratio_month_train = self.distribution[self.distribution['dataset'] == 'train']['month'].value_counts() / total_n_samples_train
        ratio_month_test_temp = self.distribution[self.distribution['dataset'] == 'test']['month'].value_counts() / total_n_samples_test
        # it can be the case that a month is not present in the test set
        months = np.arange(1, 13)
        ratio_month_test = pd.Series({month: ratio_month_test_temp[month] if month in ratio_month_test_temp else (1/(12*3)) for month in months})
        # normalize it to sum = 1
        ratio_month_test = ratio_month_test / ratio_month_test.sum()
        print(f"Sum of the weights for the month: {ratio_month_test.sum()}")
        self.distribution['month_weight'] = 1 + (ratio_month_test[self.distribution['month']].values - ratio_month_train[self.distribution['month']].values) / ratio_month_train[self.distribution['month']].values
        print(f"Sum of the weights for the month: {self.distribution[self.distribution['dataset'] == 'train']['month_weight'].sum()}")

        # weighting by the ice chart provider
        ratio_icechart_provider_train = self.distribution[self.distribution['dataset'] == 'train']['icechart_provider'].value_counts() / total_n_samples_train
        ratio_icechart_provider_test = self.distribution[self.distribution['dataset'] == 'test']['icechart_provider'].value_counts() / total_n_samples_test
        self.distribution['icechart_provider_weight'] = 1 + (ratio_icechart_provider_test[self.distribution['icechart_provider']].values - ratio_icechart_provider_train[self.distribution['icechart_provider']].values) / ratio_icechart_provider_train[self.distribution['icechart_provider']].values
        print(f"Sum of the weights for the icechart_provider: {self.distribution[self.distribution['dataset'] == 'train']['icechart_provider_weight'].sum()}")

        # weighting by the Sentinel mission identifier
        ratio_Sentinel_mission_identifier_train = self.distribution[self.distribution['dataset'] == 'train']['Sentinel_mission_identifier'].value_counts() / total_n_samples_train
        ratio_Sentinel_mission_identifier_test = self.distribution[self.distribution['dataset'] == 'test']['Sentinel_mission_identifier'].value_counts() / total_n_samples_test
        self.distribution['Sentinel_mission_identifier_weight'] = 1 + (ratio_Sentinel_mission_identifier_test[self.distribution['Sentinel_mission_identifier']].values - ratio_Sentinel_mission_identifier_train[self.distribution['Sentinel_mission_identifier']].values) / ratio_Sentinel_mission_identifier_train[self.distribution['Sentinel_mission_identifier']].values
        print(f"Sum of the weights for the Sentinel_mission_identifier: {self.distribution[self.distribution['dataset'] == 'train']['Sentinel_mission_identifier_weight'].sum()}")

        # weighting by the location
        ratio_location_train_temp = self.distribution[self.distribution['dataset'] == 'train']['location'].value_counts() / total_n_samples_train
        ratio_location_test_temp = self.distribution[self.distribution['dataset'] == 'test']['location'].value_counts() / total_n_samples_test
        # it can be the case that a location is not present in the test set
        locations = self.distribution['location'].unique()
        n_locations = len(locations)
        ratio_location_train = pd.Series({location: ratio_location_train_temp[location] if location in ratio_location_train_temp.keys() else (1./(n_locations*3.)) for location in locations})
        ratio_location_test = pd.Series({location: ratio_location_test_temp[location] if location in ratio_location_test_temp.keys() else (1./(n_locations*3.)) for location in locations})
        # normalize it to sum = 1
        ratio_location_train = ratio_location_train / ratio_location_train.sum()
        ratio_location_test = ratio_location_test / ratio_location_test.sum()
        self.distribution['location_weight'] = 1 + (ratio_location_test[self.distribution['location']].values - ratio_location_train[self.distribution['location']].values) / ratio_location_train[self.distribution['location']].values
        print(f"Sum of the weights for the location: {self.distribution[self.distribution['dataset'] == 'train']['location_weight'].sum()}")

        # adding weight for difficult locations
        self.distribution['difficult_location'] = self.distribution['location'].isin(self.options['difficult_locations'])
        negative_weight = self.distribution['difficult_location'].value_counts()[True] / self.distribution['difficult_location'].value_counts()[False]
        self.distribution['difficult_location'] = self.distribution['difficult_location'].map({False: -negative_weight, True: 1})
        additional_factor_difficult_location = (1 / total_n_samples_train)
        self.distribution['difficult_location_weight'] = 1 + (self.distribution['difficult_location'] * additional_factor_difficult_location)
        print(f"Sum of the weights for the difficult_location: {self.distribution[self.distribution['dataset'] == 'train']['difficult_location_weight'].sum()}")

        # # weighting by the size of the scenes
        # TODO?

        # for now only weighting by the location until I found a better solution
        # doing the weighting and normalizing after every step (only really needed for location)
        # self.distribution['weight'] *= self.distribution['month_weight']
        # print(f"weight values after month: {self.distribution['weight'].values.sum()}")
        # self.distribution['weight'] /= self.distribution['weight'].sum()

        # self.distribution['weight'] *= self.distribution['icechart_provider_weight']
        # print(f"weight values after icechart_provider: {self.distribution['weight'].values.sum()}")
        # self.distribution['weight'] /= self.distribution['weight'].sum()

        # self.distribution['weight'] *= self.distribution['Sentinel_mission_identifier_weight']
        # print(f"weight values after Sentinel_mission_identifier: {self.distribution['weight'].values.sum()}")
        # self.distribution['weight'] /= self.distribution['weight'].sum()

        self.distribution['weight'] *= self.distribution['location_weight']
        print(f"weight values after location: {self.distribution['weight'].values.sum()}")
        self.distribution['weight'] /= self.distribution['weight'].sum()

        # self.distribution['weight'] *= self.distribution['difficult_location_weight']
        # print(f"weight values after difficult_location: {self.distribution['weight'].values.sum()}")
        # self.distribution['weight'] /= self.distribution['weight'].sum()
        # print(f"weight values after normalization: {self.distribution['weight'].values.sum()}")

    def calculate_distribution(self):
        # code close to the one from distributions.ipynb
        with open('datalists/testset.json', 'r') as f:
            testset = json.load(f)
        trainset = self.options['train_list_extendedNames']
        valset = self.options['validate_list_extendedNames']

        # create dataframes from the lists
        trainset_df = pd.DataFrame(trainset, columns=['filename'])
        testset_df = pd.DataFrame(testset, columns=['filename'])
        valset_df = pd.DataFrame(valset, columns=['filename'])

        # add a new column indicating the dataset
        trainset_df['dataset'] = 'train'
        testset_df['dataset'] = 'test'
        valset_df['dataset'] = 'val'

        # concatenate the dataframes
        self.distribution = pd.concat([trainset_df, testset_df, valset_df], ignore_index=True)

        # check for duplicates
        if self.distribution.duplicated(subset='filename').any():
            raise ValueError('There are duplicates in the distribution dataframe')

        # split the filename into the appropriate columns
        self.distribution["Sentinel_mission_identifier"] = self.distribution['filename'].str.split("_").str[0]
        self.distribution["misc"] = self.distribution['filename'].str.split("_").str[1:4]
        self.distribution["image_acquisition_start_date"] = self.distribution['filename'].str.split("_").str[4]
        self.distribution["image_acquisition_start_date"] = pd.to_datetime(self.distribution["image_acquisition_start_date"], format='%Y%m%dT%H%M%S')
        self.distribution["image_acquisition_end_date"] = self.distribution['filename'].str.split("_").str[5]
        self.distribution["image_acquisition_end_date"] = pd.to_datetime(self.distribution["image_acquisition_end_date"], format='%Y%m%dT%H%M%S')
        self.distribution["image_type"] = self.distribution['filename'].str.split("_").str[9]
        self.distribution["icechart_provider"] = self.distribution['filename'].str.split("_").str[10]
        self.distribution["location"] = np.nan
        self.distribution["icechart_date"] = np.nan

        for i in range(0, len(self.distribution)):
            if self.distribution["icechart_provider"][i] == 'cis':
                self.distribution.loc[(i, "location")] = self.distribution['filename'][i].split("_")[11]
                try:
                    self.distribution.loc[(i, "icechart_date")] = self.distribution['filename'][i].split("_")[12]
                    self.distribution.loc[(i, "icechart_date")] = pd.to_datetime(self.distribution["icechart_date"][i], format='%Y%m%dT%H%MZ')
                except ValueError:
                    self.distribution.loc[(i, "icechart_date")] = np.nan
            elif self.distribution["icechart_provider"][i] == 'dmi':
                try:
                    self.distribution.loc[(i, "icechart_date")] = self.distribution['filename'][i].split("_")[11]
                    self.distribution.loc[(i, "icechart_date")] = pd.to_datetime(self.distribution["icechart_date"][i], format='%Y%m%d%H%M%S')
                except ValueError:
                    self.distribution.loc[(i, "icechart_date")] = np.nan
                self.distribution.loc[(i, "location")] = self.distribution['filename'][i].split("_")[12]

        # add columns called hour, month, year based on the image_acquisition_start_date
        self.distribution["month"] = self.distribution["image_acquisition_start_date"].dt.month.astype(int)
        self.distribution["hour"] = self.distribution["image_acquisition_start_date"].dt.hour.astype(int)
        self.distribution["year"] = self.distribution["image_acquisition_start_date"].dt.year.astype(int)

        # create empty columns
        classes = {}
        classes['SIC'] = list(SIC_GROUPS.keys()) + [SIC_LOOKUP['mask']]
        classes['SOD'] = list(SOD_GROUPS.keys()) + [SOD_LOOKUP['mask']]
        classes['FLOE'] = list(FLOE_GROUPS.keys()) + [FLOE_LOOKUP['mask']]

        self.distribution["size"] = np.nan
        for key, value in classes.items():
            for i in range(0, len(value)):
                self.distribution[f'{key}_{str(value[i])}'] = np.nan

        # the prepared scenes are uniquely identified by <test_data>/<image_acquisition_start_date>_<icechart_provider>_prep.nc
        for i in range(0, len(self.distribution)):
            gc.collect()

            # create the path and get the size of the file
            path = None
            if self.distribution["dataset"][i] == 'train' or self.distribution["dataset"][i] == 'val':
                try:
                    path = os.path.join(self.options['path_to_processed_data'], self.distribution["image_acquisition_start_date"][i].strftime('%Y%m%dT%H%M%S') + '_' + self.distribution["icechart_provider"][i] + '_prep.nc')
                    self.distribution.loc[(i, "size")] = os.path.getsize(path)
                except FileNotFoundError:
                    print(self.distribution.iloc[i])
            elif self.distribution["dataset"][i] == 'test':
                try:
                    path = os.path.join(self.options['path_to_processed_data'], 'test_data', self.distribution["image_acquisition_start_date"][i].strftime('%Y%m%dT%H%M%S') + '_' + self.distribution["icechart_provider"][i] + '_prep.nc')
                    self.distribution.loc[(i, "size")] = os.path.getsize(path)
                except FileNotFoundError:
                    print(self.distribution.loc[i])

            # open the file and get the distribution of the different chart classes (maybe unnecessary)
            scene = xr.open_dataset(path)

            for chart, value in classes.items():
                for j in range(0, len(value)):
                    if self.distribution["dataset"][i] == 'train' or self.distribution["dataset"][i] == 'val':
                        self.distribution.loc[(i, f'{chart}_{str(value[j])}')] = np.count_nonzero(scene[chart] == value[j])
                    elif self.distribution["dataset"][i] == 'test':
                        continue

            del scene


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
    train_options['amsrenv_upsample_shape'] = (int(train_options['patch_size'] +
                                                   train_options['amsrenv_patch_dec'] *
                                                   train_options['amsrenv_delta']),
                                               int(train_options['patch_size'] +
                                                   train_options['amsrenv_patch_dec'] *
                                                   train_options['amsrenv_delta']))
    train_options['sar_variables'] = [variable for variable in train_options['train_variables']
                                      if 'sar' in variable or 'map' in variable]
    train_options['full_variables'] = np.hstack(
        (train_options['charts'], train_options['sar_variables']))
    train_options['amsrenv_variables'] = [variable for variable in train_options['train_variables']
                                          if 'sar' not in variable and 'map' not in variable]

    return train_options
