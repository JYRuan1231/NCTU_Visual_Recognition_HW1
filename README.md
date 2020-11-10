# Visual_Recognition_HW1

## Introduction
The project provides an accurate classification model of the car brands for the contest on [Kaggle](https://www.kaggle.com/c/cs-t0828-2020-hw1/).

## Usage
We training and testing with Python 3.6, pytorch 1.4. Need to use and reference [timm](https://github.com/rwightman/pytorch-image-models), [AutoAugment](https://github.com/DeepVoltaire/AutoAugment).

### Traning and Testing model
First choose one of the following models:
* resnet50
* densenet201
* inception_resnet_v2
* resnext50_32x4d
* resnext101_32x8d
* efficientnet_b4

We supply two ways to use our program(training and testing)


Example:

```
python main.py -m resnet50 --lr 0.01 -epochs 50 -e_n _v1
python main.py -m resnet50 --lr 0.01 -epochs 50 -e_n _v2

Required arguments:
--model -m                Choose model:resnet50, densenet201, inception_resnet_v2, resnext50_32x4d, resnext101_32x8d, efficientnet_b4

Not Required arguments:
--lr -l                   Base learning rate                            
--epochs -e			   Number of epochs                                    
----e_name -e_n		   Extra model's name avoid to replace other same name of model

```

Default:

| Argument    | Default value |
| ------------|:-------------:|
| lr          | 0.001         |
| epochs      | 50            |
| e_name      | DL_model      |


OR

```
from model import train_test_model

train_test_model("resnet50", 0.01, 1, "_1")
train_test_model("resnet50", 0.01, 1, "_2")
```
When the program was finished, we will get a model file `/models/` and a csv file `/result/`.
So we have two models and two csv files.

```
./models/resnet50_1_model  
./models/resnet50_2_model
./result/resnet50_1.csv
./result/resnet50_2.csv
```

### Ensemble Learning

We supply two ways to use Ensemble Learning as well. Need to put trained models in folder `/models/`.

Example:

```
python ensemble.py -m resnet50_1_model -m resnet50_2_model 


Required arguments:
--model -m 				Choose model:resnet50, densenet201, inception_resnet_v2, resnext50_32x4d, resnext101_32x8d, efficientnet_b4
```

OR

```
from model import train_test_model

models = ["resnet50_1_model","resnet50_2_model"]
ensemble_learning(models)
```

When the program was finished, we will get a csv file `/result/`.

```
./result/voting.csv
```
## Result

| Model Name                    | Testing Performance (on Kaggle) |
| ------------------------------|:-------------------------------:|
| Resnet50                      | 0.90040                         |
| Densen201                     | 0.91320                         |
| Resnext50_32x4d               | 0.92140                         |
| Inceptionresnet_v2            | 0.92460                         |
| Resnext101_32x8d              | 0.92820                         |
| Efficienct_net_b4             | 0.93680                         |
| Top 3 model (ensmble learning)| 0.94240                         |
| Top 5 model (ensmble learning)| 0.94640                         |



