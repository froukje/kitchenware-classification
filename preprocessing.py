#! /usr/bin/env python3
"""read train.csv and sort images by class
create train val split
read test.csv and save in separate folder
create folder structure to use datasets.ImageFolder
train
- cup
- glass
- plate
- spoon
- fork
- knife
val
- cup
- glass
- plate
- spoon
- fork
- knife
test (only for inference)
"""
import glob
import os
import shutil

import numpy as np
import pandas as pd


def read_images(path):
    """read image pathes and save in a list"""
    image_files = []
    for image in glob.glob(os.path.join(path, "*.jpg")):
        image_files.append(image)
        image_files.sort()
    return image_files


def copy_test_images(test_data, img_path):
    """copy test images to test folder"""
    for _, row in test_data.iterrows():
        img = os.path.join(img_path, f'{row["Id"]:04d}.jpg')
        dst = os.path.join("data/test", f'{row["Id"]:04d}.jpg')
        shutil.copyfile(img, dst)


def train_val_split(train_data, img_path):
    """sort images by class and create train and validation data"""
    cups_images = []
    glass_images = []
    plate_images = []
    spoon_images = []
    fork_images = []
    knife_images = []
    class_images = [
        cups_images,
        glass_images,
        plate_images,
        spoon_images,
        fork_images,
        knife_images,
    ]
    class_names = ["cup", "glass", "plate", "spoon", "fork", "knife"]
    # sort images by classes
    for _, row in train_data.iterrows():
        # print(row["Id"], row["label"])
        img = os.path.join(img_path, f'{row["Id"]:04d}.jpg')
        for i, class_name in enumerate(class_names):
            if row["label"] == class_name:
                class_images[i].append(img)

    # random 20% of each class as validation data
    val_percent = 0.2
    for i, class_list in enumerate(class_images):
        print(f"{class_names[i]}: {len(class_list)}")
        val = np.random.choice(
            class_list, size=int(len(class_list) * val_percent), replace=False
        )
        print(f"length validation data: {len(val)}")
        for img_file in val:
            img = img_file.split("/")[-1]
            class_list.remove(img_file)
            dst = os.path.join(f"data/val/{class_names[i]}", img)
            shutil.copyfile(img_file, dst)
        print(f"length training data: {len(class_list)}")
        for img_file in class_list:
            img = img_file.split("/")[-1]
            dst = os.path.join(f"data/train/{class_names[i]}", img)
            shutil.copyfile(img_file, dst)


if __name__ == "__main__":
    IMG_PATH = "data/images"
    META_PATH = "data"
    images = read_images(IMG_PATH)
    print(images[:5])
    df_train = pd.read_csv(os.path.join(META_PATH, "train.csv"))
    df_test = pd.read_csv(os.path.join(META_PATH, "test.csv"))
    print(f"training and validation data length: {len(df_train)}")
    print(f"test data length: {len(df_test)}")

    print(df_train.head())
    # create folders, if they don't exist:
    folders = [
        "data/train",
        "data/val",
        "data/test",
        "data/train/cup",
        "data/train/glass",
        "data/train/plate",
        "data/train/spoon",
        "data/train/fork",
        "data/train/knife",
        "data/val/cup",
        "data/val/glass",
        "data/val/plate",
        "data/val/spoon",
        "data/val/fork",
        "data/val/knife",
    ]
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)

    copy_test_images(df_test, IMG_PATH)
    train_val_split(df_train, IMG_PATH)
