import torchvision.models as models
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import pickle
import torch
import math

import socialSigNoDrop
importlib.reload(socialSigNoDrop)
from helpers import *


df = pd.read_csv("./data/migration_matrix.csv")
df = df.sample(20000)
sending = df['sending'].to_list()
print(df.head())
df = df.drop(['Unnamed: 0', 'receiving_GEO2_MX', 'receivingGEO2_MX', 'receiving', 'sending', 'sending_GEO2_MX'], axis = 1)
df = df.fillna(0)
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))



y =  torch.Tensor(df['number_moved'].values)
X = df.loc[:, df.columns != "number_moved"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)


# Prep model with trained weights
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True)
model = socialSigNoDrop.scoialSigNet_NoDrop(X=torch.reshape(torch.tensor(X[0], dtype = torch.float32), (1, X[0].shape[0])), outDim = 1, resnet = resnet50).to(device)
checkpoint = torch.load("./trained_models/socialSig_MEX_50epochs.torch")
model.load_state_dict(checkpoint['model_state_dict'])


print(X.shape)
# print(jkljdklajl)

# Evaluate model and save predictions
eval_df = eval_model(X, y, sending, (1, X[0].shape[0]), model, device)
eval_df.to_csv("./predictions/socialSigNoDrop_MEX_preds.csv", index = False)