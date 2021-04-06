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

import socialSigNoDrop
importlib.reload(socialSigNoDrop)
from helpers import *


df = pd.read_csv("./data/migration_matrix.csv")
df = df.drop(['Unnamed: 0', 'receiving_GEO2_MX', 'receivingGEO2_MX', 'receiving', 'sending', 'sending_GEO2_MX'], axis = 1)
df = df.fillna(0)
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
df.head()


y = df['number_moved'].values
X = df.loc[:, df.columns != "number_moved"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)


lr = 1e-6
batchSize = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = socialSigNoDrop.SocialSigNet(X=X, outDim = batchSize).to(device)
resnet50 = models.resnet50(pretrained=True)
model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = batchSize, resnet = resnet50).to(device)
epochs = 5
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr = lr)


x_train, y_train, x_val, y_val = train_test_split(X, y, .80)


print("x_train: ", len(x_train) )
print("y_train: ", len(y_train) )
print("x_val  : ", len(x_val ) )
print("y_val  : ", len(y_val  ) )



train = [(k,v) for k,v in zip(x_train, y_train)]
val = [(k,v) for k,v in zip(x_val, y_val)]


print(len(train))
print(len(val))


# Prep the training and validation set
train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)



model_wts = train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)



model.load_state_dict(model_wts)
torch.save({
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./trained_models/socialSig_MEX_50epochs.torch")

