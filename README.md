# CadLegalTrain


## Introduction
This program was developed for COMP482 - Natural Language Processing.

The purpose of this program is to identify the laws cited in Canadian legal cases, which can be expanded in the future to identify legal strategies for solving cases. First, datasets of Canadian legislation and regulations are cleaned and highlight the citation each law. Then, texts of legal cases are put into our sentence transformer model to find law citations. Lastly, the extracted citations are compared against citations from the law datsets to determine the laws used, as well as the accuracy of identified laws, and outputs all laws used in the case as well as the accuracy of the law cited being correctly identified to the user.

## Development Environment Configuration

This program was developed and designed to be run on a Windows 10 Home or Windows 11 Home 64 bit based machine. 

## Programming Languages Utilized

For this program, we utilized the Python programming language for all of our code. After running code, a few JSON files will be generated to store text embeddings.

## Project Folder Hierarchy

COMP482-Project\CadLegalTrain
COMP482-Project\CadLegalTrain\canadian_legal_data\train

COMP482-Project\Progress Logs
COMP482-Project\Progress Logs\Sachin Muthayan
COMP482-Project\Progress Logs\Mason Paquette
COMP482-Project\Progress Logs\Jasan Brar
COMP482-Project\Progress Logs\Diego Mackay

COMP482-Project\Images

Please note that this does not include any <u><b>files such as python code or images<b></u> contained within the folders.

## Installation and complation guide

Please follow the steps below to successfully run our program.

### Library Installation

import torch
from tqdm import tqdm

#### re, os, time, and pickle
These are standard libraries built into Python, they do not require installation.

#### numPy
**Command:** py -m pip install numpy **OR** pip install numpy

#### Pandas
**Command:** py -m pip install pandas **OR** pip install pandas

#### gc
**Command:** py -m pip install gc **OR** pip install gc

#### KMeans and pairwise_distances_argmin_min
**Command:** py -m pip install scikit-learn **OR** pip install scikit-learn

#### Datasets
**Command:** py -m pip install datasets **OR** pip install datasets

#### Sentence Transformers
**Command:** py -m pip install -U sentence-transformers **OR** pip install -U sentence-transformers

#### Transformers
**Command:** py -m pip install transformers **OR** pip install transformers

#### PyTorch
**Command:** py -m pip install torch **OR** pip install torch

#### tqdm
**Command:** py -m pip install tqdm **OR** pip install tqdm

### Compilation Commands

Ensure you have navigated to the directory with this code, then follow the steps below and enter the following commands in a CLI terminal:

#### 1. Setup
  
  bash: ./dataset.py

#### 2. Vectorization 
  bash: ./vectorize.py or (split_vectorize_dataset.py)

#### 3. Cluster
  bash: ./cluster_function.py

#### 4. Tokenize

  bash: ./tokenize.py (tokenize_legal.py)

#### 5. Law Identification

  bash: ./main.py

  
#### 6. Train gpt-2 (Diego)

  
  bash: ./train.py


#### 7. Test Model (Diego)


  bash: ./testgpt-2.py

  
### Folder Structure changes

After running the code, there will be no additional <u>folders</u>, although a json file will be generated.

## Tools used

VSCode for writing code

Github Desktop for managing and pushing commits to the repository

## AI Usage
We did NOT use <u>any</u> AI tools for this project.

