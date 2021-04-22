import torchvision.models as models
from sklearn import preprocessing
import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import argparse
import sklearn
import random
import torch
import math
import json

import socialSigNoDrop
importlib.reload(socialSigNoDrop)
from helpers import *


df = pd.read_csv("./us_migration_allvars.csv")
sending = df['sending'].to_list()
df = df.drop(['Unnamed: 0', 'sending'], axis = 1)
df = df.fillna(0)
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

print(df.columns)


with open("./us_vars.txt", "r") as f:
    vars = f.read().splitlines()

df = df[vars]

print(df.shape)

print(df.columns)

y = df['num_persons_to_us'].values
X = df.loc[:, df.columns != "num_persons_to_us"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)



# lr = 1e-3
lr = 1e-4
# lr = 1e-6
# lr = 1e-8
batchSize = 16
epochs = 25
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = socialSigNoDrop.SocialSigNet(X=X, outDim = batchSize).to(device)
resnet50 = models.resnet50(pretrained=True)
model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = batchSize, resnet = resnet50).to(device)
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
checkpoint = torch.load("./trained_models/notransfer_25epoch_normalloss_us.torch")
model.load_state_dict(checkpoint['model_state_dict'])



eval_df = eval_model(X, y, sending, (1, X[0].shape[0]), model, device)
eval_df.to_csv("./predictions/notransfer_25epoch_normalloss_us_preds.csv")

