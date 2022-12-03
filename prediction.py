"""use the trained model to make predictions and save these as csv"""

import argparse
import glob
import os
from collections import namedtuple

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from model import KitchenClassification
from train import get_date

# from multiprocessing import Pool


def read_data(data_path):
    """read all image locations and save as list"""
    data_list = glob.glob(os.path.join(data_path, "*.jpg"))
    data_list.sort()
    print("read images list:")
    print(f"{data_list[:5]} ...")
    return data_list


def preprocessing(data_list, img_size):
    """preprocess data to apply model"""
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((img_size, img_size)),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    imgs = []
    for img in data_list:
        img = Image.open(img)
        img_transform = transform(img)
        imgs.append(img_transform)

    preprocessed_data = torch.stack(imgs)
    print(preprocessed_data.size())

    return preprocessed_data


def make_predictions(model, data, data_list):
    """apply the model to make predictions and store results in a dataframe"""
    data = data.to(device)
    predictions = {}
    classes = {0: "cup", 1: "fork", 2: "glass", 3: "knife", 4: "plate", 5: "spoon"}
    for i in range(data.size()[0]):
        y_hat = model(data[i][None, :])
        y_pred = torch.argmax(y_hat, axis=1)
        name = data_list[i].split("/")[-1].split(".")[0]
        predictions[name] = classes[y_pred.item()]

    data = {"Id": predictions.keys(), "label": predictions.values()}
    # save as dataframe
    df_preds = pd.DataFrame.from_dict(data)
    print(df_preds.head())
    return df_preds


def write_predictions(df_preds, pred_path):
    """write predictions to csv"""
    print("write predictions ...")
    df_preds.to_csv(pred_path, index=False)
    print(f"predictions saved to {pred_path}")


if __name__ == "__main__":
    print(torch.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="saved_models")
    parser.add_argument(
        "--model-choice", type=str, default="latest", choices=["latest", "custom"]
    )
    parser.add_argument(
        "--model-file", type=str, default="best_model-epoch=4-val_loss=1.79-v1.ckpt"
    )
    parser.add_argument("--data-path", type=str, default="data/test")
    parser.add_argument("--prediction-path", type=str, default="predictions")
    parser.add_argument("--n-processes", type=int, default=8)
    args = parser.parse_args()

    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print()

    # load checkpoint
    if args.model_choice == "latest":
        # load latest model
        list_of_models = glob.glob(f"{args.model_path}/*.ckpt")
        model = max(list_of_models, key=os.path.getctime)
        model_file = model.split("/")[-1]
        print(model)
        print(model_file)
    if args.model_choice == "custom":
        model_file = args.model_file
        model = os.path.join(args.model_path, args.model_file)
    # load checkpoint parameters
    checkpoint = torch.load(
        os.path.join(args.model_path, model_file), map_location=torch.device("cpu")
    )

    # set model_state parameter to inference in order to ignore hyperparameter saving
    checkpoint["hyper_parameters"]["train"] = False

    # convert dictionary to object to access values via dot notation
    checkpoint_args = namedtuple("ObjectName", checkpoint["hyper_parameters"].keys())(
        *checkpoint["hyper_parameters"].values()
    )

    valid_model = KitchenClassification.load_from_checkpoint(
        model, args=checkpoint_args
    ).eval()
    # set device to GPU, if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid_model.to(device)

    data_list = read_data(args.data_path)
    preprocessed_data = preprocessing(data_list, checkpoint_args.img_size)
    df_preds = make_predictions(valid_model, preprocessed_data, data_list)

    time = get_date()
    name = model_file.split(".")[0]
    pred_path = os.path.join(args.prediction_path, f"prediction_{time}_{name}.csv")

    write_predictions(df_preds, pred_path)
    # with Pool(args.n_processes) as p:
    #    print(p.map(write_predictions, df_preds))
