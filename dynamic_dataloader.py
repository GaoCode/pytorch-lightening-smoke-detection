"""
Created by: Anshuman Dewangan
Date: 2021

Description: Loads data from raw image and XML files
"""
# Torch imports
import pytorch_lightning as pl
import pickle
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# Other package imports
import os

# File imports
import util_fns


#####################
## Data Module
#####################

class DynamicDataModule(pl.LightningDataModule):
    def __init__(self, 
                 raw_data_path='./data/raw_data', 
                 labels_path='./data/labels', 
                 metadata_path='./data/metadata.pkl',
                 train_split_path=None,
                 val_split_path=None,
                 test_split_path=None,
                 train_split_size=0.6,
                 test_split_size=0.2,
                 batch_size=1, 
                 num_workers=0, 
                 series_length=5, 
                 time_range=(-2400, 2400), 
                 image_dimensions = (1536, 2016),
                 crop_height = 1344,
                 tile_dimensions = (224, 224),
                 smoke_threshold = 10,
                 create_data = False):
        """
        Args:
            - raw_data_path (str): path to raw data
            - labels_path (str): path to XML labels
            - metadata_path (str): path to metadata.pkl
                - fire_to_images (dict): dictionary with fires as keys and list of corresponding images as values
                - num_fires (int): total number of fires in dataset
                - num_images (int): total number of images in dataset
                - ground_truth_label (dict): dictionary with fires as keys and 1 if fire has "+" in its file name
                - has_xml_label (dict): dictionary with fires as keys and 1 if fire has a .xml file associated with it
                - omit_images_list (list of str): list of images that erroneously do not have XML files for labels
            
            - train_split_path (str): path to existing train split .txt file
            - val_split_path (str): path to existing val split .txt file
            - test_split_path (str): path to existing test split .txt file
            
            - train_split_size (float): % of data to split for train
            - test_split_size (float): % of data to split for test
            - batch_size (int): batch_size for training
            - num_workers (int): number of workers for dataloader
            - series_length (int): how many sequential images should be used for training
            - time_range (int, int): The time range of images to consider for training by time stamp
            
            - image_dimensions (int, int): desired dimensions of image before cropping
            - crop_height (int): height to crop image to
            - tile_dimensions (int, int): desired size of tiles
            - smoke_threshold (int): # of pixels of smoke to consider tile positive
            
            - create_data (bool): should prepare_data be run?
        
        Other Attributes:
            - self.train_split (list): list of image names to be used for train dataloader
            - self.val_split (list): list of image names to be used for val dataloader
            - self.test_split (list): list of image names to be used for test dataloader
            - self.has_setup (bool): if setup has already occurred to prevent from doing twice
        """
        super().__init__()
        
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.metadata = pickle.load(open(metadata_path, 'rb'))
        
        self.train_split_path = train_split_path
        self.val_split_path = val_split_path
        self.test_split_path = test_split_path
        
        self.train_split_size = train_split_size
        self.test_split_size = test_split_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.series_length = series_length
        self.time_range = time_range
        
        self.image_dimensions = image_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.smoke_threshold = smoke_threshold
        
        self.create_data = create_data
        self.has_setup = False
        
    def prepare_data(self):
        """
        Description: Creates metadata.pkl and saved labels for easier loading. Only needs to be run once.
        """
        if self.create_data:
            print("Preparing Data...")
            
            ### Create metadata.pkl ###
            self.metadata = {}

            self.metadata['fire_to_images'] = util_fns.generate_fire_to_images(self.raw_data_path, self.labels_path)
            self.metadata['num_fires'] = 0
            self.metadata['num_images'] = 0

            for fire in self.metadata['fire_to_images']:
                self.metadata['num_fires'] += 1
                for image in self.metadata['fire_to_images']:
                    self.metadata['num_images'] += 1
                    
                    self.metadata['ground_truth_label'] = util_fns.get_ground_truth_label(image)
                    self.metadata['has_xml_label'] = util_fns.get_has_xml_label(image, self.labels_path)

            self.metadata['omit_images_list'] = util_fns.generate_omit_images_list(self.metadata)
        
            with open(f'./data/metadata.pkl', 'wb') as pkl_file:
                pickle.dump(self.metadata, pkl_file)
                
            ### Generate saved labels ###
            util_fns.save_labels(
                raw_data_path='/userdata/kerasData/data/new_data/raw_images',
                labels_path='/userdata/kerasData/data/new_data/drive_clone',
                output_path='/userdata/kerasData/data/new_data/drive_clone_labels')
                
            print("Preparing Data Complete.")
        
        
    def setup(self, stage=None, log_dir=None):
        """
        Args:
            - log_dir (str): logging directory to save train/val/test splits 
        """
        if self.has_setup: return
        print("Setting Up Data...")

        # If any split not provided, randomly create own train/val/test splits
        if self.train_split_path is None or self.val_split_path is None or self.test_split_path is None:
            train_fires, val_fires = train_test_split(list(self.metadata['fire_to_images'].keys()), test_size=(1-self.train_split_size))
            val_fires, test_fires = train_test_split(val_fires, test_size=self.test_split_size/(1-self.train_split_size))
            
            # Shorten fire_to_images to relevant time frame
            self.metadata['fire_to_images'] = util_fns.shorten_time_range(self.metadata['fire_to_images'], self.time_range, train_fires)

            # Save arrays representing series of images
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length) 

            # Create train/val/test split of Images
            self.train_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], train_fires, self.metadata['omit_images_list'])
            self.val_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], val_fires, self.metadata['omit_images_list'])
            self.test_split = util_fns.unpack_fire_images(self.metadata['fire_to_images'], test_fires, self.metadata['omit_images_list'], is_test=True)

            # If logdir is provided, then save train/val/test splits
            if log_dir:
                os.makedirs(log_dir)
                np.savetxt(log_dir+'/train_images.txt', self.train_split, fmt='%s')
                np.savetxt(log_dir+'/val_images.txt', self.val_split, fmt='%s')
                np.savetxt(log_dir+'/test_images.txt', self.test_split, fmt='%s')
        else:
            # Load existing splits
            train_list = np.loadtxt(self.train_split_path, dtype=str)
            val_list = np.loadtxt(self.val_split_path, dtype=str)
            test_list = np.loadtxt(self.test_split_path, dtype=str)
            
            self.train_split = [util_fns.get_image_name(item) for item in train_list]
            self.val_split   = [util_fns.get_image_name(item) for item in val_list]
            self.test_split  = [util_fns.get_image_name(item) for item in test_list]
            
            # Recreate fire_to_images and image_series
            self.metadata['fire_to_images'] = util_fns.generate_fire_to_images([self.train_split, self.val_split, self.test_split])
            self.metadata['image_series'] = util_fns.generate_series(self.metadata['fire_to_images'], self.series_length) 
        
        self.has_setup = True
        print("Setting Up Data Complete.")
            

    def train_dataloader(self):
        train_dataset = DynamicDataloader(self.raw_data_path, 
                                          self.labels_path, 
                                          self.metadata,
                                          self.train_split,
                                          self.image_dimensions,
                                          self.crop_height,
                                          self.tile_dimensions,
                                          self.smoke_threshold)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        return train_loader

    def val_dataloader(self):
        val_dataset = DynamicDataloader(self.raw_data_path, 
                                        self.labels_path,  
                                        self.metadata, 
                                        self.val_split,
                                        self.image_dimensions,
                                        self.crop_height,
                                        self.tile_dimensions,
                                        self.smoke_threshold)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_dataset = DynamicDataloader(self.raw_data_path, 
                                         self.labels_path,   
                                         self.metadata,
                                         self.test_split,
                                         self.image_dimensions,
                                         self.crop_height,
                                         self.tile_dimensions,
                                         self.smoke_threshold)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)
        return test_loader
    
    
#####################
## Dataloader
#####################
    
class DynamicDataloader(Dataset):
    def __init__(self, 
                 raw_data_path, 
                 labels_path, 
                 metadata,
                 data_split, 
                 image_dimensions = (1536, 2016),
                 crop_height = 1344,
                 tile_dimensions = (224,224), 
                 smoke_threshold = 10):
        """
        Args / Attributes:
            - raw_data_path (Path): path to raw data
            - labels_path (Path): path to XML labels
            - metadata (dict): metadata dictionary from DataModule
            - data_split (list): list of images of the current split
            - image_dimensions (int, int): desired dimensions of image before cropping
            - crop_height (int): height to crop image to
            - tile_dimensions (int, int): desired size of tiles
            - smoke_threshold (int): # of pixels of smoke to consider tile positive
        """
        self.raw_data_path = raw_data_path
        self.labels_path = labels_path
        self.metadata = metadata
        self.data_split = data_split
        
        self.image_dimensions = image_dimensions
        self.crop_height = crop_height
        self.tile_dimensions = tile_dimensions
        self.smoke_threshold = smoke_threshold

    def __len__(self):
        return len(self.data_split)

    def __getitem__(self, idx):
        image_name = self.data_split[idx]
        
        # Load all images in the series
        x = []
        series_length = len(self.metadata['image_series'][image_name])
        
        for file_name in self.metadata['image_series'][image_name]:
            img = torchvision.io.read_image(file_name) # img.shape = [num_channels, height, width]
            img = T.Resize(self.image_dimensions)(img)[:,-self.crop_height:] # Resize and crop
            x.append(img)
        
        # x.shape = [series_length, num_channels, height, width]
        # e.g. [5, 3, 1344, 2016]
        x = torch.stack(x) / 255 # Normalize by /255 (good enough normalization)
           
        # Load XML labels        
        label_path = self.labels_path+'/'+image_name+'.pt'
        
        if Path(label_path).exists():
            labels = torch.load(label_path)
            
            # labels.shape = [height, width]
            labels = T.Resize(self.image_dimensions)(labels.unsqueeze(0))[:,-self.crop_height:].squeeze(0)
        else:
            labels = torch.zeros(x.shape[2:])
                
        # WARNING: Tile size must divide perfectly into image height and width
        if self.tile_dimensions:
            # x.shape = [54, 5, 3, 224, 224]
            # labels.shape = [54, 224, 224]
            x = x.view((-1, series_length, 3, self.tile_dimensions[0], self.tile_dimensions[1]))
            labels = labels.view((-1, self.tile_dimensions[0], self.tile_dimensions[1]))

            # tile_labels.shape = [54,]
            labels = (labels.sum(dim=(1,2)) > self.smoke_threshold).float()
        else:
            # Pretend as if tile size = image size
            x = x.unsqueeze(0)
            labels = labels.unsqueeze(0)

        # Load image-level labels for current image
        ground_truth_label = self.metadata['ground_truth_label'][image_name]
        has_xml_label = self.metadata['has_xml_label'][image_name]
        has_positive_tile = util_fns.get_has_positive_tile(labels)
                
        return image_name, x, labels, ground_truth_label, has_xml_label, has_positive_tile