# Semantik_Segmentation_Adversarial_Tranig_CIRA-PDG

## Introduction
This repository is part of a university research project. Our goal with this system is to provide an effective tool to create robust semantic segmentation models. This project is under development. Please feel free to give us recommendations.

## Project Setup
To start training

First install Python 3. add create conda env and enter in to it.

```
conda create -n robust-segmantation-train python=3.9
source activate robust-segmantation-train
```

We advise you to install Python 3 and use pip to use our project:

```
cd $HOME
git clone --recursive git@github.com:HalmosiL/Semantik_Segmentation_Adversarial_Training_CIRA-PDG.git
cd Semantik_Segmentation_Adversarial_Training_CIRA-PDG
pip install -r requirements.txt
```

## Download Dataset
To install Cityscapes dataset we recomend you to use this repository:

https://github.com/mcordts/cityscapesScripts.git

## Training
To start a traning please use this scripts

Start executor wait until it's setup
```
./start_executor.sh ../configs/config_main.json
```

Start traning
```
python start_traning.py ../configs/config_main.json
```
