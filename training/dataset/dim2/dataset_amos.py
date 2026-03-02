import os
import glob

import numpy as np
import torch
import pytorch_lightning as pl

from monai.data import (
    CacheDataset,
    list_data_collate,
)

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
)


class AMOSDataset(pl.LightningDataModule):
    """Class to define the dataset AMOS for the challenge.
    """   
    organs = [
        'Background', 'Spleen', 'Right Kidney', 'Left Kidney', 'Gallbladder', 
        'Esophagus', 'Liver', 'Stomach', 'Aorta', 'IVC', 'Pancreas', 
        'Right Adrenal', 'Left Adrenal', 'Duodenum', 'Bladder', 'Prostate/Uterus'
    ]

    def __init__(self, args):
        """ Constructor of the class AMOSDataset.

        Args:
            args (argparse.Namespace): Arguments from the command line.
        """        
        super().__init__()

        self.args = args
        self.keys = ["image", "label"]
        self.mode = ("nearest") 
        self.spatial_size = args.roi_size[:2]
        self.scale_range = [-175, 250]

        self.train_files = [] 
        self.val_files = []
        self.test_files = []

        self.preprocess = None
        self.transform = None

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None


    def prepare_data(self):
        """Function to prepare the data for the training, validation and test sets. Splits dataset into train, val and test.

        Raises:
            TypeError: Error if the number of images and labels do not match or are empty.
        """        

        data_images, data_labels = self.__get_data(folders_img_lbl=self.args.folders_img_lbl)

        if (len(data_images) != len(data_labels)) or len(data_images) == 0 or len(data_labels) == 0:
            raise TypeError("Error: Number of images and labels do not match or are empty")

        # Group slices by patient ID to ensure correct splitting (Data Leakage prevention)
        patient_groups = {}
        for img, lbl in zip(data_images, data_labels):
            # Assumes filename format: case_0001_slice_XXX.npy
            pid = os.path.basename(img).split('_')[1] 
            if pid not in patient_groups:
                patient_groups[pid] = []
            patient_groups[pid].append({self.keys[0]: img, self.keys[1]: lbl})

        patient_ids = sorted(list(patient_groups.keys()))
        total_patients = len(patient_ids)
        
        percentage_val = round((1.0 - self.args.percentage_train)/2, 2)
        train_num = int(np.floor(self.args.percentage_train * total_patients))
        val_num = int(np.floor(percentage_val * total_patients))
        test_num = total_patients - train_num - val_num
        
        print("Total Patients {}: train {}, val {} and test {}".format(total_patients, train_num, val_num, test_num))

        # Shuffle and split IDs
        np.random.seed(42) 
        indices = np.random.permutation(total_patients)
        
        indices_train = indices[:train_num]
        indices_val = indices[train_num:train_num+val_num]
        indices_test = indices[train_num+val_num:]

        train_ids = [patient_ids[i] for i in indices_train]
        val_ids = [patient_ids[i] for i in indices_val]
        test_ids = [patient_ids[i] for i in indices_test]

        # Flatten the lists for the dataset
        self.train_files = [item for pid in train_ids for item in patient_groups[pid]]
        self.val_files   = [item for pid in val_ids for item in patient_groups[pid]]
        self.test_files  = [item for pid in test_ids for item in patient_groups[pid]]
        
        print("Total Slices: train {}, val {} and test {}".format(len(self.train_files), len(self.val_files), len(self.test_files)))


    def load_images_prediction(self):
        """Function to load the images for the prediction.

        Returns:
            list: List of dictionaries with the images to predict.
        """       
        data_images = self.__get_data_pred(folders_img_lbl=self.args.folders_img_lbl)
        data_dicts = [
                {self.keys[0]: image_name}
                for image_name in data_images
            ]
        return data_dicts


    def get_preprocessing_transform(self):
        """Function to get the preprocessing transformations for the dataset.

        Returns:
            monai.transforms.compose.Compose: Compose object for preprocessing transformations for the dataset.
        """        
        val_test_transforms = Compose(
            [
            LoadImaged(keys=self.keys, image_only=False),
            EnsureChannelFirstd(keys=self.keys, channel_dim='no_channel'),
            EnsureTyped(keys=self.keys, dtype=torch.float32),
            ]
        )
        return val_test_transforms


    def get_augmentation_transform(self): 
        """Function to get the augmentation transformations for the dataset.

        Returns:
            monai.transforms.compose.Compose: Augmentation transformations for the dataset.
        """        
        train_transforms = Compose([
            LoadImaged(keys=self.keys, image_only=False),
            EnsureChannelFirstd(keys=self.keys, channel_dim='no_channel'),
            EnsureTyped(keys=self.keys, dtype=torch.float32),
            RandRotate90d(keys=self.keys, prob=0.5, max_k=3),
            RandFlipd(keys=self.keys, spatial_axis=[0], prob=0.5),
            RandFlipd(keys=self.keys, spatial_axis=[1], prob=0.5),
            RandShiftIntensityd(keys=[self.keys[0]], offsets=0.10, prob=0.50),
            ])
        return train_transforms


    def get_preprocessing_transform_pred(self):
        """Function to get the preprocessing transformations for the prediction.

        Returns:
            monai.transforms.compose.Compose: Compose object for preprocessing transformations for the prediction.
        """        
        return self.get_preprocessing_transform(), None


    def setup(self, stage=None):
        """Function to setup the dataset for the training, validation and test sets. Load the data and apply the transformations.

        Args:
            stage (str, optional): Different stage (fit, test). Defaults to None.
        """        
        self.preprocess = self.get_preprocessing_transform()
        self.augment = self.get_augmentation_transform()

        if stage == "fit" or stage is None:
            self.train_ds = CacheDataset(
                data=self.train_files,
                transform=self.augment,
                cache_rate=self.args.cache_rate,
                num_workers=self.args.num_workers,
            )

            self.val_ds = CacheDataset(
                data=self.val_files,
                transform=self.preprocess,
                cache_rate=self.args.cache_rate,
                num_workers=self.args.num_workers,
            )
        
        if stage == "test" or stage is None:

            self.test_ds = CacheDataset(
                data=self.test_files,
                transform=self.preprocess,
                cache_rate=self.args.cache_rate,
                num_workers=self.args.num_workers,
            )


    def train_dataloader(self):
        """Function to get the training dataloader.

        Returns:
            torch.utils.data.dataloader.DataLoader: Dataloader for the training set.
        """        
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            collate_fn=list_data_collate,
        )
        return train_loader


    def val_dataloader(self):
        """Function to get the validation dataloader.

        Returns:
            torch.utils.data.dataloader.DataLoader: Dataloader for the validation set.
        """        
        val_loader = torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=self.args.pin_memory, 
            collate_fn=list_data_collate,
        )
        return val_loader


    def test_dataloader(self):
        """Function to get the test dataloader.

        Returns:
            torch.utils.data.dataloader.DataLoader: Dataloader for the test set.
        """        
        test_loader = torch.utils.data.DataLoader(
            self.test_ds, 
            batch_size=1, 
            shuffle=False, 
            num_workers=self.args.num_workers, 
            pin_memory=self.args.pin_memory, 
            collate_fn=list_data_collate,
        )
        return test_loader


    def __get_data(self, folders_img_lbl=True):
        """Function to get the data from the path.

        Args:
            folders_img_lbl (bool, optional): Parameter to set if training and test data are in different folders. Defaults to True.

        Returns:
            list, list: List of images and list of labels.
        """        

        if folders_img_lbl:
            data_images = sorted(
                glob.glob(os.path.join(self.args.data_dir, "imagesTr", "*.npy")))
            data_labels = sorted(
                glob.glob(os.path.join(self.args.data_dir, "labelsTr", "*.npy")))
            
            return data_images, data_labels

        else:
            all_files = sorted(
                glob.glob(os.path.join(self.args.data_dir, "*.npy")))

            data_images = sorted(
                filter(lambda x: "_gt" in x, all_files))
            data_labels = sorted(
                filter(lambda x: "_gt" not in x, all_files))
            
            return data_images, data_labels


    def __get_data_pred(self, folders_img_lbl=True):
        """Function to get the data from the path.

        Args:
            folders_img_lbl (bool, optional): Parameter to set if training and test data are in different folders. Defaults to True.

        Returns:
            list: List of images.
        """ 

        if folders_img_lbl:
            data_images = sorted(
                glob.glob(os.path.join(self.args.data_dir, "imagesTs", "*.npy")))
            
            return data_images

        else:
            all_files = sorted(
                glob.glob(os.path.join(self.args.data_dir, "*.npy")))

            data_images = sorted(
                filter(lambda x: "_gt" in x, all_files))
            
            return data_images