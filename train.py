"""
author: Frauke Albrecht
"""
import argparse
import datetime
import logging
import os

import mlflow.pytorch
import pytorch_lightning as pl
import torch
from nni.utils import merge_parameter
from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import nni
from model import KitchenClassification


class KitchenDataModule(pl.LightningDataModule):
    """
    create data module
    """

    def __init__(self, args):
        super().__init__()
        datadir_train = os.path.join(args.data, "train")
        datadir_val = os.path.join(args.data, "val")
        if args.debug:
            datadir_train = os.path.join(args.data, "debug_train")
            datadir_val = os.path.join(args.data, "debug_val")
        logger.info(f"train data directory: {datadir_train}")
        logger.info(f"validation data directory: {datadir_val}")

        self.args = args
        self.train_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(90),
                transforms.RandomRotation(180),
                transforms.RandomRotation(270),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        self.val_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((args.img_size, args.img_size)),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.train_dataset = datasets.ImageFolder(
            datadir_train, transform=self.train_transforms
        )
        self.val_dataset = datasets.ImageFolder(
            datadir_val, transform=self.val_transforms
        )

    def setup(self, stage: str):
        pass

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=args.num_worker,
            drop_last=True,
        )
        logger.info(f"train_dataloader: {next(iter(train_dataloader))[0].shape}")
        logger.info(f"train_dataloader: {next(iter(train_dataloader))[1].shape}")
        return train_dataloader

    def val_dataloader(self):
        val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=args.num_worker,
        )
        logger.info(f"val_dataloader: {next(iter(val_dataloader))[0].shape}")
        logger.info(f"val_dataloader: {next(iter(val_dataloader))[1].shape}")
        return val_dataloader


class KitchenCallbacks(Callback):
    """custom callbacks"""

    def __init__(self, args):
        self.args = args

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        logger.info(f"\nValidation epoch end [epoch: {trainer.current_epoch}]:")
        for key, item in metrics.items():
            logger.info(f"{key}: {item:.4}")
        # if self.args.nni:
        #    nni.report_intermediate_result(float(metrics['val_loss']))

    def on_train_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        logger.info("\nFinal validation loss:")
        for key, item in metrics.items():
            logger.info(f"{key}: {item:.4}")
        # if self.args.nni:
        #    nni.report_final_result(float(metrics['val_loss']))


def add_nni_params(args):
    """add parameters from nni to argparse arguments and adapt the path for saving the model"""
    args_nni = nni.get_next_parameter()
    assert all((key in args for key in args_nni.keys())), "need only valid parameters"
    args_dict = vars(args)
    # cast params that should be int to int if needed (nni may offer them as float)
    args_nni_casted = {
        key: (int(value) if isinstance(args_dict[key], int) else value)
        for key, value in args_nni.items()
    }
    args_dict.update(args_nni_casted)

    # adjust paths of model and prediction outputs so they get saved together with the other outputs
    nni_output_dir = os.path.expandvars("$NNI_OUTPUT_DIR")
    for param in ["save_model_path"]:
        nni_path = os.path.join(nni_output_dir, os.path.basename(args_dict[param]))
        args_dict[param] = nni_path
    return args


def get_date():
    """get current date and time"""
    x = datetime.datetime.now()
    year = x.year
    month = x.month
    day = x.day
    hour = x.hour
    minute = x.minute
    time = f"{year}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}"
    return time


if __name__ == "__main__":

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--save-model-path", default="saved_models")
    parser.add_argument("--logdir", default="logs")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--num-worker", type=int, default=0)
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--nr-classes", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--backbone", type=str, default="vgg16")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--nni", action="store_true", default=False)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    time = get_date()
    logger = logging.getLogger("training")
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f"{args.logdir}/logfile_{time}.log")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if args.nni:
        args = add_nni_params(args)

    params = vars(args)
    logger.info("argparse arguments:")
    for key, value in params.items():
        logger.info(f"{key}: {value}")
    logger.info("\n")

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=args.save_model_path,
        filename="{args.backbone}-{time}-{epoch}-{val_loss:.2f}",
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.patience,
        verbose=False,
        mode="min",
    )

    model = KitchenClassification(args)
    data_module = KitchenDataModule(args)
    data_module.setup(stage="fit")
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    acc = GPUAccelerator() if torch.cuda.is_available() else CPUAccelerator()
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[checkpoint_callback, early_stop_callback, KitchenCallbacks(args)],
        accelerator=acc,
        max_epochs=args.max_epochs,
    )

    if args.nni:
        tuner_params = nni.get_next_parameter()
        params = vars(merge_parameter(args, tuner_params))
        # get parameters form tuner
        logger.info(nni.get_trial_id())

    mlflow.set_experiment("test")
    with mlflow.start_run():
        # mlflow.log_params(params)
        mlflow.pytorch.autolog()
        if args.nni:
            mlflow.set_tag(key="NNI experiment", value=nni.get_experiment_id())
        trainer.fit(model, train_loader, val_loader)
        mlflow.log_artifact(local_path=checkpoint_callback.best_model_path)

    logger.info(f"best model path: {checkpoint_callback.best_model_path}")
    logger.info(f"best loss: {checkpoint_callback.best_model_score.item()}")
