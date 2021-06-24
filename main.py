"""
Created by: Anshuman Dewangan
Date: 2021

Description: Kicks off training and evaluation. Contains many command line arguments for hyperparameters. 
"""

# Torch imports
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Other package imports
from argparse import ArgumentParser
import datetime

# File imports
from lightning_model import LightningModel
from models import ResNet50, ResNet50Focal
from dynamic_dataloader import DynamicDataModule, DynamicDataloader
import util_fns

IS_DEBUG = True

#####################
## Argument Parser
#####################

# All args recorded as hyperparams. Recommended not to use unnecessary args
parser = ArgumentParser(description='Takes raw wildfire images and saves tiled images')

# Experiment args
parser.add_argument('--experiment-name', type=str, default=None,
                    help='(Optional) Name for experiment')
parser.add_argument('--experiment-description', type=str, default=None,
                    help='(Optional) Short description of experiment that will be saved as a hyperparam')

# Path args
parser.add_argument('--raw-data-path', type=str, default='/userdata/kerasData/data/new_data/raw_images',
                    help='Path to raw images.')
parser.add_argument('--labels-path', type=str, default='/userdata/kerasData/data/new_data/drive_clone_labels',
                    help='Path to processed XML labels.')
parser.add_argument('--metadata-path', type=str, default='./data/metadata.pkl',
                    help='Path to metadata.pkl.')
parser.add_argument('--train-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with train image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--val-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with val image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--test-split-path', type=str, default=None,
                    help='(Optional) Path to txt file with test image paths. Only works if train, val, and test paths are provided.')
parser.add_argument('--checkpoint-path', type=str, default=None,
                    help='(Optional) Path to checkpoint to load.')

# Dataloader args
parser.add_argument('--train-split-size', type=int, default=0.6,
                    help='% of data to split for train.')
parser.add_argument('--test-split-size', type=int, default=0.2,
                    help='% of data to split for test.')
parser.add_argument('--batch-size', type=int, default=1,
                    help='Batch size for training.')
parser.add_argument('--num-workers', type=int, default=0,
                    help='Number of workers for dataloader.')
parser.add_argument('--series-length', type=int, default=1,
                    help='Number of sequential video frames to process during training.')
parser.add_argument('--time-range-min', type=int, default=-2400,
                    help='Start time of fire images to consider during training. ')
parser.add_argument('--time-range-max', type=int, default=2400,
                    help='End time of fire images to consider during training (inclusive).')

parser.add_argument('--image-height', type=int, default=1536,
                    help='Desired resize height of image.')
parser.add_argument('--image-width', type=int, default=2016,
                    help='Desired resize width of image.')
parser.add_argument('--crop-height', type=int, default=1344,
                    help='Desired height after cropping.')
parser.add_argument('--tile-size', type=int, default=224,
                    help='Height and width of tile.')
parser.add_argument('--smoke-threshold', type=int, default=10,
                    help='Number of pixels of smoke to consider tile positive.')

parser.add_argument('--flip-augment', action='store_true',
                    help='Enables data augmentation with horizontal flip.')
parser.add_argument('--blur-augment', action='store_true',
                    help='Enables data augmentation with Gaussian blur.')

# Model args
parser.add_argument('--learning-rate', type=float, default=0.001,
                    help='Learning rate for training.')
parser.add_argument('--no-lr-schedule', action='store_true',
                    help='Disables ReduceLROnPlateau learning rate scheduler. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-pretrain-backbone', action='store_true',
                    help='Disables pretraining of backbone.')
parser.add_argument('--no-freeze-backbone', action='store_true',
                    help='Disables freezing of layers on pre-trained backbone.')
parser.add_argument('--focal-alpha', type=float, default=0.25,
                    help='Alpha for focal loss.')
parser.add_argument('--focal-gamma', type=float, default=2,
                    help='Gamma for focal loss.')

# Training args
parser.add_argument('--min-epochs', type=int, default=10,
                    help='Min number of epochs to train for.')
parser.add_argument('--max-epochs', type=int, default=50,
                    help='Max number of epochs to train for.')
parser.add_argument('--no-auto-lr-find', action='store_true',
                    help='Disables auto learning rate finder. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-early-stopping', action='store_true',
                    help='Disables early stopping based on validation loss. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-sixteen-bit', action='store_true',
                    help='Disables use of 16-bit training to reduce memory. See PyTorch Lightning docs for more details.')
parser.add_argument('--no-stochastic-weight-avg', action='store_true',
                    help='Disables stochastic weight averaging. See PyTorch Lightning docs for more details.')
parser.add_argument('--gradient-clip-val', type=float, default=0,
                    help='Clips gradients to prevent vanishing or exploding gradients. See PyTorch Lightning docs for more details.')
parser.add_argument('--accumulate-grad-batches', type=int, default=1,
                    help='Accumulate multiple batches before calling loss.backward() to increase effective batch size. See PyTorch Lightning docs for more details.')

    
#####################
## Main
#####################
    
def main(# Path args
        raw_data_path, 
        labels_path,
        metadata_path=None,
        train_split_path=None, 
        val_split_path=None, 
        test_split_path=None,
        checkpoint_path=None,
    
        # Experiment args
        experiment_name=None,
        experiment_description=None,
        parsed_args=None,

        # Dataloader args
        train_split_size=0.6,
        test_split_size=0.2,
        batch_size=1, 
        num_workers=0, 
        series_length=1, 
        time_range=(-2400,2400), 
        image_dimensions=(1536, 2016),
        crop_height=1344,
        tile_dimensions=(224,224),
        smoke_threshold=10,
        flip_augment=False,
        blur_augment=False,

        # Model args
        learning_rate=0.001,
        lr_schedule=True,
        pretrain_backbone=True,
        freeze_backbone=True,
        focal_alpha=0.25,
        focal_gamma=2,

        # Trainer args 
        min_epochs=10,
        max_epochs=50,
        auto_lr_find=True,
        early_stopping=True,
        sixteen_bit=True,
        stochastic_weight_avg=True,
        gradient_clip_val=0,
        accumulate_grad_batches=1):
        
    try:
        if not IS_DEBUG: util_fns.send_fb_message(f'Experiment {experiment_name} Started...')
        
        ### Initialize data_module ###
        data_module = DynamicDataModule(
            # Path args
            raw_data_path=raw_data_path,
            labels_path=labels_path,
            metadata_path=metadata_path,
            train_split_path=train_split_path,
            val_split_path=val_split_path,
            test_split_path=test_split_path,

            # Dataloader args
            train_split_size=train_split_size,
            test_split_size=test_split_size,
            batch_size=batch_size,
            num_workers=num_workers,
            series_length=series_length,
            time_range=time_range,
        
            image_dimensions=image_dimensions,
            crop_height=crop_height,
            tile_dimensions=tile_dimensions,
        
            flip_augment=flip_augment,
            blur_augment=blur_augment)
        
        ### Initialize model ###
        backbone = ResNet50Focal(series_length, 
                                pretrain_backbone=pretrain_backbone,
                                freeze_backbone=freeze_backbone,
                                focal_alpha=focal_alpha,
                                focal_gamma=focal_gamma)
        model = LightningModel(model=backbone,
                               learning_rate=learning_rate,
                               lr_schedule=lr_schedule,
                               parsed_args=parsed_args)

        ### Implement EarlyStopping ###
        early_stop_callback = EarlyStopping(
           monitor='val/loss',
           min_delta=0.00,
           patience=3,
           verbose=True)

        checkpoint_callback = ModelCheckpoint(monitor='val/loss', save_last=True)

        ### Initialize Trainer ###

        # Initialize logger 
        logger = TensorBoardLogger("lightning_logs/", 
                                   name=experiment_name, 
                                   log_graph=True,
                                   version=None)
        
        # Set up data_module and save train/val/test splits
        data_module.setup(log_dir=logger.log_dir)

        trainer = pl.Trainer(
            # Trainer args
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            auto_lr_find=auto_lr_find,
            callbacks=[early_stop_callback, checkpoint_callback] if early_stopping else [checkpoint_callback],
            precision=16 if sixteen_bit else 32,
            stochastic_weight_avg=stochastic_weight_avg,
            gradient_clip_val=gradient_clip_val,
            accumulate_grad_batches=accumulate_grad_batches,
            
            # Other args
            resume_from_checkpoint=checkpoint_path,
            logger=logger,

            # Dev args
#             fast_dev_run=True, 
            overfit_batches=1,
#             limit_train_batches=0.25,
#             limit_val_batches=0.25,
#             limit_test_batches=0.25,
#             log_every_n_steps=1,
#             checkpoint_callback=False,
#             logger=False,
#             track_grad_norm=2,
#             weights_summary='full',
            profiler="simple", # "advanced" "pytorch"
#             log_gpu_memory=True,
            gpus=1)
        
        ### Training & Evaluation ###

        # Auto find learning rate
        if auto_lr_find:
            trainer.tune(model, datamodule=data_module)

        # Train the model
        trainer.fit(model, datamodule=data_module)

        # Evaluate the best model on the test set
        trainer.test(model, datamodule=data_module)

        if not IS_DEBUG: util_fns.send_fb_message(f'Experiment {experiment_name} Complete')
    except Exception as e:
        if not IS_DEBUG: util_fns.send_fb_message(f'Experiment {args.experiment_name} Failed. Error: ' + str(e))
        raise(e) 
    
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    main(# Path args
        raw_data_path=args.raw_data_path, 
        labels_path=args.labels_path, 
        metadata_path=args.metadata_path,
        train_split_path=args.train_split_path, 
        val_split_path=args.val_split_path, 
        test_split_path=args.test_split_path,
        checkpoint_path=args.checkpoint_path,

        # Experiment args
        experiment_name=args.experiment_name,
        experiment_description=args.experiment_description,
        parsed_args=args,

        # Dataloader args
        train_split_size=args.train_split_size,
        test_split_size=args.test_split_size,
        batch_size=args.batch_size, 
        num_workers=args.num_workers, 
        series_length=args.series_length, 
        time_range=(args.time_range_min,args.time_range_max), 
        
        image_dimensions=(args.image_height, args.image_width),
        crop_height=args.crop_height,
        tile_dimensions=(args.tile_size, args.tile_size),
        smoke_threshold=args.smoke_threshold,
        
        flip_augment=args.flip_augment,
        blur_augment=args.blur_augment,

        # Model args
        learning_rate=args.learning_rate,
        lr_schedule=not args.no_lr_schedule,
        pretrain_backbone=not args.no_pretrain_backbone,
        freeze_backbone=not args.no_freeze_backbone,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,

        # Trainer args
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        auto_lr_find=not args.no_auto_lr_find,
        early_stopping=not args.no_early_stopping,
        sixteen_bit=not args.no_sixteen_bit,
        stochastic_weight_avg=not args.no_stochastic_weight_avg,
        gradient_clip_val=args.gradient_clip_val,
        accumulate_grad_batches=args.accumulate_grad_batches)
 
    