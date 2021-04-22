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
df = df.drop(['Unnamed: 0', 'sending'], axis = 1)
df = df.fillna(0)
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

with open("./us_vars.txt", "r") as f:
    vars = f.read().splitlines()
df = df[vars]




print("CLASS 0: ", len(df[(df['num_persons_to_us'] == 0)]))
print("CLASS 1: ", len(df[(df['num_persons_to_us'] > 0) & (df['num_persons_to_us'] < 14)]))
print("CLASS 2: ", len(df[(df['num_persons_to_us'] >= 14) & (df['num_persons_to_us'] < 198)]))
print("CLASS 3: ", len(df[(df['num_persons_to_us'] >= 198) & (df['num_persons_to_us'] < 600)]))
print("CLASS 4: ", len(df[(df['num_persons_to_us'] >= 600) & (df['num_persons_to_us'] < 34582.000000)]))

weights = {"0" : 1/185, "1": 1/384, "2": 1/1177, "3": 1/401, "4": 1/183}


def classify_migration(x):
    if x == 0:
        return weights["0"]
        # return 0
    elif x > 0 and x < 14:
        return weights["1"]
        # return 1
    elif x >= 14 and x < 198:
        return weights["1"]
        # return 2
    elif x >= 198 and x < 600:
        return weights["2"]
        # return 3
    else:
        return weights["3"]
        # return 4



df['weight'] = df['num_persons_to_us'].apply(lambda x: classify_migration(x))

print(df['weight'].value_counts())

w = df['weight'].values


df = df.drop(['weight'], axis = 1)

print(df.columns)


y = df['num_persons_to_us'].values
X = df.loc[:, df.columns != "num_persons_to_us"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)




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
checkpoint = torch.load("./trained_models/socialSig_MEX_10epochs_AdamLoss.torch")
model.load_state_dict(checkpoint['model_state_dict'])




x_train, y_train, x_val, y_val, w_train, w_val = train_test_split(X, y, w, .80)

print("x_train: ", len(x_train))
print("y_train: ", len(y_train))
print("x_val  : ", len(x_val))
print("y_val  : ", len(y_val))


print(w_train)



train = [(k,v,w) for k,v,w in zip(x_train, y_train, w_train)]
val = [(k,v,w) for k,v,w in zip(x_val, y_val, w_val)]


print(len(train))
print(len(val))





# Prep the training and validation set
train = torch.utils.data.DataLoader(train, batch_size = batchSize, shuffle = True)
val = torch.utils.data.DataLoader(val, batch_size = batchSize, shuffle = True)


print('done')



model_wts, val_losses_plot = train_weighted_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)






model.load_state_dict(model_wts)
torch.save({
            'epoch': 20,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./trained_models/transfer_25epoch_weightedloss_us.torch")





print(val_losses_plot)

plt.plot([i for i in range(0, epochs)], val_losses_plot)
plt.savefig(("./transfer_25epoch_weightedloss_us.png"))