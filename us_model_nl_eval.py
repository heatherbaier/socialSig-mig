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


df = pd.read_csv("./data/mexico2010.csv")
sending = df['GEO2_MX'].to_list()
df = df.drop(['Unnamed: 0', 'GEO2_MX'], axis = 1)
df = df.fillna(0)
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
with open("./us_vars.txt", "r") as f:
    vars = f.read().splitlines()
vars = [i for i in vars if i in df.columns]
df = df[vars]


y = df['sum_num_intmig'].values
X = df.loc[:, df.columns != "sum_num_intmig"].values

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
checkpoint = torch.load("./new_trained_models/notransfer_50epoch_normalloss_us.torch")
model.load_state_dict(checkpoint['model_state_dict'])



eval_df = eval_model(X, y, sending, (1, X[0].shape[0]), model, device)
eval_df.to_csv("./new_predictions/notransfer_50epoch_normalloss_us_preds.csv")

