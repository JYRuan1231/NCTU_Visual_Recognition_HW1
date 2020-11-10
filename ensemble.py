#!/usr/bin/env python
# coding: utf-8

import torchvision
from torchvision import transforms, utils, models, datasets
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from auto_augment import ImageNetPolicy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import argparse
from random import sample
import matplotlib.pyplot as plt
import csv
import numpy as np
import time
import os
import copy
import timm


def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        "-m",
        action="append",
        dest="models",
        help=("choose models to apply essamble learning"),
    )

    return parser.parse_args()


def ensemble_learning(candidate=None):
    data_transforms = {
        "test": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        ),
    }
    model_list = [
        "resnet50",
        "densenet201",
        "resnext50_32x4d",
        "resnext101_32x8d",
        "inception_resnet_v2",
        "efficientnet_b4",
    ]
    test_path = "./data/testing_data/testing_data/"
    allFileList = os.listdir(test_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if candidate == None:
        print("No input model")
        return

    print("Models start voting")

    # Final predict
    predicts = []
    for i in range(len(candidate)):
        model_name = candidate[i]
        model_path = "./models/" + model_name
        for x in model_list:
            if model_name.find(x) == 0:
                model_name = x

        if model_name in [
            "resnet50",
            "densenet201",
            "resnext50_32x4d",
            "resnext101_32x8d",
        ]:
            data_transforms = {
                "test": transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                ),
            }
            if model_name == "resnet50":
                model_ft = models.resnet50(pretrained=True)
            if model_name == "densenet201":
                model_ft = models.densenet201(pretrained=True)
            if model_name == "resnext50_32x4d":
                model_ft = models.resnext50_32x4d(pretrained=True)
            if model_name == "resnext101_32x8d":
                model_ft = models.resnext101_32x8d(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 196)
        elif model_name in ["inception_resnet_v2"]:
            data_transforms = {
                "test": transforms.Compose(
                    [
                        transforms.Resize(312),
                        transforms.CenterCrop(299),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                    ]
                ),
            }
            model_ft = timm.create_model(
                model_name, pretrained=True, num_classes=196
            )
        elif model_name in ["efficientnet_b4"]:
            data_transforms = {
                "test": transforms.Compose(
                    [
                        transforms.Resize(412),
                        transforms.CenterCrop(380),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                        ),
                    ]
                ),
            }
            model_ft = timm.create_model(
                "tf_efficientnet_b4_ns", pretrained=True, num_classes=196
            )

        model_ft.load_state_dict(torch.load(model_path))
        model_ft = model_ft.to(device)
        predict = []

        for file in allFileList:
            if os.path.isfile(test_path + file):
                path = test_path + file
                img = Image.open(path).convert("RGB")
                img = data_transforms["test"](img)
                img = img.unsqueeze(0)
                with torch.no_grad():
                    model_ft.eval()
                    img = img.to(device)
                    outputs = model_ft(img)
                    _, preds = torch.max(outputs, 1)
                    predict.append(int(preds.cpu().numpy()))
        predicts.append(predict)

    # obtain label conversion table
    with open("./data/training_labels.csv", newline="") as csvfile:
        rows = csv.DictReader(csvfile)
        label_number = {}
        total_id = 0
        id_list = []
        class_count = 0
        for row in rows:
            if row["label"] not in label_number:
                label_number[row["label"]] = class_count
                class_count = class_count + 1
            id_list.append(row["id"])
            total_id = total_id + 1
        label_conversion = {v: k for k, v in label_number.items()}

    # make final csv file
    csv_name = "./result/" + "voting" + ".csv"
    with open(csv_name, "w", newline="") as csvFile:
        field = ["id", "label"]
        writer = csv.DictWriter(csvFile, field)
        writer.writeheader()
        entry = 0
        for file in allFileList:
            result = []
            if os.path.isfile(test_path + file):
                for y in range(len(predicts)):  # Number of models
                    result.append(predicts[y][entry])
                final_predict = most_frequent(result)
                writer.writerow(
                    {
                        "id": file.split(".jpg")[0],
                        "label": label_conversion[final_predict],
                    }
                )
            entry = entry + 1

    print("Done!")


if __name__ == "__main__":

    args = process_command()

    ensemble_learning(args.models)
