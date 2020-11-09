#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, utils, models
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from cnn_finetune import make_model
import timm
from PIL import Image
import csv
import os
import copy
import time
from auto_augment import ImageNetPolicy
import random


def Image_loader(path):
    return Image.open(path).convert("RGB")


class MyDataset(Dataset):
    def __init__(
        self,
        csv_file,
        file_id,
        transform=None,
        loader=Image_loader,
        label_number=None,
    ):
        imgs = []
        with open(csv_file, newline="") as csvfile:
            rows = csv.DictReader(csvfile)
            for row in rows:
                if row["id"] in file_id:
                    img_path = (
                        "./data/training_data/training_data/"
                        + row["id"]
                        + ".jpg"
                    )
                    label = label_number[row["label"]]
                    imgs.append((img_path, int(label)))
        super(MyDataset, self).__init__()
        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)


def train_model(
    model,
    train_loader,
    valid_loader,
    train_size,
    valid_size,
    criterion,
    optimizer,
    scheduler,
    num_epochs=25,
):

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
                model_loader = train_loader
                dataset_size = train_size
            else:
                model.eval()  # Set model to evaluate mode
                model_loader = valid_loader
                dataset_size = valid_size

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in model_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(
                "{} Loss: {:.4f} Acc: {:.4f}".format(
                    phase, epoch_loss, epoch_acc
                )
            )

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print(
            "Complete one epoch in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )
        print()
    print("Best val Acc: {:4f}".format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def build_model(model_name):
    if model_name in [
        "resnet50",
        "densenet201",
        "resnext50_32x4d",
        "resnext101_32x8d",
    ]:
        data_transforms = {
            "train": transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(p=0.5),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            ),
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
            "train": transforms.Compose(
                [
                    transforms.Resize(312),
                    transforms.RandomResizedCrop(299),
                    transforms.RandomHorizontalFlip(p=0.5),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
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
            "train": transforms.Compose(
                [
                    transforms.Resize(412),
                    transforms.RandomResizedCrop(380),
                    transforms.RandomHorizontalFlip(p=0.5),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    ),
                ]
            ),
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

    return model_ft, data_transforms


def train_test_model(model_name, lr, num_epochs, extra_name):
    # Class Conversion table
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

    #
    train_size = int(0.8 * total_id)
    valid_size = total_id - train_size

    train_id = random.sample(id_list, train_size)

    for i in train_id:
        id_list.remove(i)
    valid_id = id_list

    model_ft, data_transforms = build_model(model_name)

    train_dataset = MyDataset(
        csv_file="./data/training_labels.csv",
        file_id=train_id,
        transform=data_transforms["train"],
        label_number=label_number,
    )
    valid_dataset = MyDataset(
        csv_file="./data/training_labels.csv",
        file_id=valid_id,
        transform=data_transforms["test"],
        label_number=label_number,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=8, shuffle=True, num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=8, shuffle=True, num_workers=4
    )

    # Start training model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(
        model_ft.parameters(), lr=lr, momentum=0.9, weight_decay=1e-5
    )

    # Decay LR by CosineAnnealingWarmRestarts
    exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer_ft, T_0=10, T_mult=2, eta_min=1e-5
    )

    print("Start to training model!")

    model_ft = train_model(
        model_ft,
        train_loader,
        valid_loader,
        train_size,
        valid_size,
        criterion,
        optimizer_ft,
        exp_lr_scheduler,
        num_epochs=num_epochs,
    )

    print("Done!")

    model_path = "./models/" + model_name + extra_name + "_model"

    torch.save(model_ft.state_dict(), model_path)

    # Predict the testing data then export result to a csv file
    csv_name = model_name + extra_name + ".csv"

    print("Start to predict result!")

    with open(csv_name, "w", newline="") as csvFile:

        field = ["id", "label"]
        writer = csv.DictWriter(csvFile, field)
        writer.writeheader()

        test_path = "./data/testing_data/testing_data/"
        allFileList = os.listdir(test_path)

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
                    predict = preds.cpu().numpy()
                writer.writerow(
                    {
                        "id": file.split(".jpg")[0],
                        "label": label_conversion[predict[0]],
                    }
                )

    print("Done!")
