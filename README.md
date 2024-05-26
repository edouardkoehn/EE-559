
# EE-559 Project 🌋
Repository contains the code for the project EE559. For this project, we benchmarked hateful meme detection techniques. We tested three different models: FCM, Bert, and Llava. For the FCM and Bert models, this repository contains workflows to train and evaluate those models. For the Llava model, this repository contains only workflows to evaluate the model.

To run the code, please READ the readme!!! This code can be run on a machine that has the CUDA driver. The dataset weighs around 6GB and the Llava model weighs around 8GB, thus be sure you have enough space on your machine.


## Installation and downloading the data
### 1) Set-up the environnement
- Clone the repo
```bash
git clone git@github.com:edouardkoehn/EE-559.git
```

- Create your virtual env
```bash
conda create -n wm python=3.12
conda activate EE559
```
- Install poetry
```bash
pip install poetry
```
- install the modul and set up the precommit
```bash
poetry install
poetry run pre-commit install
poetry env info
```
### 2) Set-up the dataset
- download the dataset from the following adress: [MMHS150K dataset](https://drive.google.com/file/d/1S9mMhZFkntNnYdO-1dZXwF_8XIiFcmlF/view) and save it under ```data/```
- Preprocess the dataset using the following commmand:
```bash
python src/reformat_data.py
```
### 3) Set-up the model
- download the dataset from the following adress: [model weight](https://drive.google.com/drive/folders/178WNg4i2pFYRpRJRPJxdMOGF6n6YnyJA?usp=sharing) and save it under ```data/```

## Running the code

### 1) FCM model
FCM is a multi-modal classifier. This implementation is based on this [paper](https://arxiv.org/pdf/1910.03814).
#### 1.1) Training FCM model
The FCM model can be trained using the following script :
```bash
python src/run_fcm.py
```
#### 1.2) Evaluating FCM model
A saved FCM model can be then be tested using :
```bash
python src/test_fcm.py
```
### 2) Bert model
Bert is a transfomer based classifier([original paper](https://arxiv.org/pdf/1810.04805)).
#### 2.1) Training and evaluating Bert model
To train this model, you can use the following script.
```bash
python src/bert_fine_tune_v2.py
```
### 3) Llava model
#### 3.1) Evaluating Llava model
Prediction with the Llava model can be generated with :
```bash
run_lava -c config_debug.json
```
You need to choose one of the config files that can be found under ```data/config```
To reproduce the analysis of the results, you need to reformat the prediction of the model. In order to reformat the predcitions, you can do:
```bash
python src/reformat_lava_results.py
```
### 4) Reproducing the resutls
To reproduce the results of this project, you just have to run:
```bash
python src/analysis_all.py
```
This script run the analysis for each model and save the figure under the folder ```data/figure```.
## Repository architecure

```bash
├── data/
│   ├── config/ #containing json config files for models
│   ├── pretrained_model/ #containing the model weights
│   ├── MMHS150K #containing the dataset
│   ├── results #containing the prediction of each model in a │json format
│
└── src/ #all the python file for running the model
│
└── scripts/ #all the bash script for running the model on the cluster
│
└── .gitinore
│
└── .pre-commit-config.yaml
│
└── README.md
│
└── pyproject.toml
│
└── LICENSE
```
Authors: Koehn Edouard, Bricq Marin,De Groot Barbara
