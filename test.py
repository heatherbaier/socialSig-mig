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




weights = {'0': 1/51, '1': 1/832, '2': 1/1362, '3': 1/86}

def classify_migration(x):
    if x == 0:
        return weights['0']
    elif (x > 0) & (x < 1000):
        return weights['1']
    elif (x >= 100) & (x < 10000):
        return weights['2']
    else:
        return weights['3']


df = pd.read_csv("./data/mexico2010.csv")
df = df.drop(['Unnamed: 0', 'GEO2_MX'], axis = 1)
df = df.fillna(0)
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
with open("./us_vars.txt", "r") as f:
    vars = f.read().splitlines()
vars = [i for i in vars if i in df.columns]
df = df[vars]


df['weight'] = df['sum_num_intmig'].apply(lambda x: classify_migration(x))
w = df['weight'].values
df = df.drop(['weight'], axis = 1)


y = df['sum_num_intmig'].values
X = df.loc[:, df.columns != "sum_num_intmig"].values

mMScale = preprocessing.MinMaxScaler()
X = mMScale.fit_transform(X)




lr = 1e-4
# lr = 1e-6
# lr = 1e-8
batchSize = 16
epochs = 100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = socialSigNoDrop.SocialSigNet(X=X, outDim = batchSize).to(device)
resnet50 = models.resnet50(pretrained=True)
model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = batchSize, resnet = resnet50).to(device)
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = lr)
# checkpoint = torch.load("./trained_models/socialSig_MEX_10epochs_AdamLoss.torch")
# model.load_state_dict(checkpoint['model_state_dict'])




x_train, y_train, x_val, y_val, w_train, w_val = train_test_split_weighted(X, y, w, .80)

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



best_model_wts_mae, best_model_wts_loss, val_losses_plot = train_weighted_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr)



# daadg


model.load_state_dict(best_model_wts_mae)
torch.save({
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./new_trained_models/best_model_wts_mae_200epochs.torch")


model.load_state_dict(best_model_wts_loss)
torch.save({
            'epoch': 50,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./new_trained_models/best_model_wts_loss_200epochs.torch")





print(val_losses_plot)

plt.plot([i for i in range(0, epochs)], val_losses_plot)
plt.savefig(("./new_loss_plots/best_model_wts_200epoch.png"))