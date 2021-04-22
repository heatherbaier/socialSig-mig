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



df = pd.read_csv("mex_migration_allvars.csv")
df = df.drop(['Unnamed: 0', 'receiving', 'sending'], axis = 1)
df = df.fillna(0)
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

with open("./vars.txt", "r") as f:
    vars = f.read().splitlines()

df = df[vars]

print(df.shape)

print(df.columns)

y = df['number_moved'].values
X = df.loc[:, df.columns != "number_moved"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)




lr = 1e-8
# lr = 7e-4
batchSize = 16
epochs = 12
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True)
model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = batchSize, resnet = resnet50).to(device)
criterion = torch.nn.L1Loss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)






# lr = 7e-4
# batchSize = 16
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # model = socialSigNoDrop.SocialSigNet(X=X, outDim = batchSize).to(device)
# inception = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
# model = socialSigNoDrop.socialSigNet_Inception(X=X, outDim = batchSize, inception = inception).to(device)
# epochs = 12
# criterion = torch.nn.MSELoss(reduction = 'mean')
# optimizer = torch.optim.SGD(model.parameters(), lr = lr)


x_train, y_train, x_val, y_val = train_test_split(X, y, .80)


print("x_train: ", len(x_train))
print("y_train: ", len(y_train))
print("x_val  : ", len(x_val))
print("y_val  : ", len(y_val))



train = [(k,v) for k,v in zip(x_train, y_train)]
val = [(k,v) for k,v in zip(x_val, y_val)]


print(len(train))
print(len(val))




# Prep the training and validation set
train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)


print("\nModel training starting...\n")

model_wts, val_losses_plot = train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)




model.load_state_dict(model_wts)
torch.save({
            'epoch': 20,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./trained_models/socialSig_MEX_10epochs_AdamLoss.torch")





print(val_losses_plot)

plt.plot([i for i in range(0, epochs)], val_losses_plot)
plt.savefig(("./losses_plot_full_real.png"))



