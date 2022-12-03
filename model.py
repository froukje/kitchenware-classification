"""
author: Frauke Albrecht
"""

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, Precision, Recall
from torchvision import models

SUPPORTED_MODELS = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    # "resnext50_32x4d",
    # "resnext50_32x4d",
    # "wide_resnet50_2",
    "mobilenet_v3_large",
    "mobilenet_v3_small",
    "vgg11",
    # "vgg11_bn",
    "vgg13",
    # "vgg13_bn",
    "vgg16",
    # "vgg16_bn",
    # "vgg19_bn",
    "vgg19",
    "alexnet",
    "squeezenet1_0",
]


class KitchenClassification(pl.LightningModule):
    """
    model for classification
    """

    # pylint: disable=too-many-ancestors
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.train:
            # save hyperparameters only during training
            # ignore this line if model is used for inference
            self.save_hyperparameters(args)
        assert (
            args.backbone in SUPPORTED_MODELS
        ), f"backbone model must be a supported model {SUPPORTED_MODELS}"
        self.args = args
        weights = None
        if "vgg" in args.backbone:
            if args.pretrained:
                weights = f"{args.backbone.upper()}_Weights.IMAGENET1K_V1"
            print(weights)
            self.model = models.__dict__[args.backbone](
                weights=weights
            )  # pretrained=args.pretrained)
            # adapt classifier
            # Freeze model weights
            for param in self.model.parameters():
                param.requires_grad = False

            self.model.classifier = nn.Sequential(
                nn.Linear(25088, 4096, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(4096, 2048, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(2048, args.nr_classes),
            )

        elif "resnet" in args.backbone or "resnext" in args.backbone:
            if args.pretrained:
                nr = args.backbone.split("t")[-1]
                weights = f"ResNet{nr}_Weights.IMAGENET1K_V1"
            print(weights)
            self.model = models.__dict__[args.backbone](weights=weights)

            for param in self.model.parameters():
                param.requires_grad = False

            # replace output features with nr of classes
            self.model.fc.out_features = args.nr_classes

        elif "mobilenet_v3" in args.backbone:
            self.model = models.__dict__[args.backbone](pretrained=args.pretrained)

            for param in self.model.parameters():
                param.requires_grad = False

            # adapt classifier
            in_features = self.model.classifier[0].in_features
            self.model.classifier = nn.Sequential(
                nn.Linear(in_features, args.nr_classes),
            )
        elif "alexnet" in args.backbone:
            self.model = models.__dict__[args.backbone](pretrained=args.pretrained)

            for param in self.model.parameters():
                param.requires_grad = False

            # adapt classifier
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=0.5, inplace=False),
                nn.Linear(9216, 2048, bias=True),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(2048, args.nr_classes),
            )
        # elif "squeezenet" in backbone:
        #    self.model = models.__dict__[backbone](pretrained=pretrained)
        #    self.model.features[0] = nn.Conv2d(input_dim, 96, kernel_size=(7, 7), stride=(2, 2))
        #    self.output_dim = self.model.classifier[1].out_channels

        self.modelname = args.backbone.replace("_", "-")
        print(f"training {self.modelname}")

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.lr)
        return optimizer

    def calc_losses(self, y_hat, y, prefix, calc_metrics=False):
        # pylint: disable=missing-function-docstring
        criterion = nn.CrossEntropyLoss()
        loss = criterion(y_hat, y)

        if calc_metrics:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            r = Recall(average="macro", num_classes=self.args.nr_classes).to(device)
            recall = r(torch.argmax(y_hat, axis=1), y)
            p = Precision(average="macro", num_classes=self.args.nr_classes).to(device)
            precision = p(torch.argmax(y_hat, axis=1), y)
            f1 = F1Score(num_classes=self.args.nr_classes).to(device)
            f1_score = f1(torch.argmax(y_hat, axis=1), y)
            acc = Accuracy()
            accuracy = acc(torch.argmax(y_hat, axis=1), y).to(device)
            for name, val in zip(
                ["recall", "precision", "f1_score", "accuracy"],
                [recall, precision, f1_score, accuracy],
            ):
                self.log(f"{prefix}_{name}", val, on_epoch=True)
        return loss

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.calc_losses(y_hat, y, "train")
        self.log("train_loss", loss)
        y_hat = torch.argmax(y_hat, dim=1)
        return loss

    def validation_step(self, batch, _batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.calc_losses(y_hat, y, "val", True)
        self.log("val_loss", loss)
        y_hat = torch.argmax(y_hat, dim=1)
