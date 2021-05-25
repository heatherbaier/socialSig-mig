from os import sched_getaffinity
from pandas.core.frame import DataFrame
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


# Model variables
# .005
LR = 5e-3
BATCH_SIZE = 16
EPOCHS = 200


# Data variables
TRAIN_MUNIS = "./train_municipalities.txt"
VAL_MUNIS = "./val_municipalities.txt"


# Clean and prep the data
df = pd.read_csv("./data/mexico2010.csv")
df = df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
munis = df['GEO2_MX'].to_list()
df = df.drop(['Unnamed: 0'], axis = 1).fillna(0)
f = open("./us_vars.txt", "r")
vars = f.read().splitlines()
f.close()
df = df[[i for i in vars if i in df.columns]]
df['weight'] = df['sum_num_intmig'].apply(lambda x: classify_migration(x))
X = df.drop(["sum_num_intmig", "weight", "GEO2_MX"], axis = 1).values


# Save included variables to a text file for records
outfile = open('final_variables.txt', 'w')
[outfile.write(L + "\n") for L in df.columns]
outfile.close()


for col in df.columns:
    if col not in ['weight', "sum_num_intmig", "GEO2_MX"]:
        df[col] = df[col] / max(df[col])


# Train/val split
train_munis = read_muni_list(TRAIN_MUNIS)
val_munis = read_muni_list(VAL_MUNIS)

train_df = df[df['GEO2_MX'].isin(train_munis)]
val_df = df[df['GEO2_MX'].isin(val_munis)]

y_train = train_df['sum_num_intmig'].values
x_train = train_df.drop(['weight', "sum_num_intmig", "GEO2_MX"], axis = 1).values
w_train = train_df['weight'].values


y_val = val_df['sum_num_intmig'].values
x_val = val_df.drop(['weight', "sum_num_intmig", "GEO2_MX"], axis = 1).values
w_val = val_df['weight'].values


# Double check shapes
print(x_train.shape)
print(x_val.shape)

print(y_train.shape)
print(y_val.shape)

print(w_train.shape)
print(w_val.shape)


# Prep the final training and validation set
train = [(k,v,w) for k,v,w in zip(x_train, y_train, w_train)]
val = [(k,v,w) for k,v,w in zip(x_val, y_val, w_val)]



def eval(data, model, device):
    preds, trues = [], []
    for obs in data:
        cur_x = torch.tensor(obs[0], dtype = torch.float32)
        cur_true = obs[1]
        pred = model(cur_x.to(device)).item()
        preds.append(pred)
        trues.append(cur_true.item())
    preds_df = pd.DataFrame()
    preds_df['true'] = trues   
    preds_df['pred'] = preds
    return preds_df




# for iteration in range(0, 10):

train_dl = torch.utils.data.DataLoader(train, batch_size = BATCH_SIZE, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE, shuffle = True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
resnet50 = models.resnet50(pretrained=True)
model = socialSigNoDrop.scoialSigNet_NoDrop(X=X, outDim = BATCH_SIZE, resnet = resnet50).to(device)
criterion = torch.nn.MSELoss(reduction = 'mean')
optimizer = torch.optim.Adam(model.parameters(), lr = LR, betas = (0.5, 0.999))

# Train the model
best_model_wts_mae, best_model_wts_loss, val_losses_plot = train_weighted_model(model, train_dl, val_dl, criterion, optimizer, EPOCHS, BATCH_SIZE, device, LR)

train_dl = torch.utils.data.DataLoader(train, batch_size = 1, shuffle = True)
val_dl = torch.utils.data.DataLoader(val, batch_size = 1, shuffle = True)


"""MAE"""

print("Evaluating MAE.")

model.load_state_dict(best_model_wts_mae)

train_name = "./final_ss/socialSig_trainpreds_200epochs_mae_5e3.csv"
val_name = "./final_ss/socialSig_valpreds_200epochs_mae_5e3.csv"

eval(val_dl, model, device).to_csv(val_name, index = False)
eval(train_dl, model, device).to_csv(train_name, index = False)

torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./final_ss/socialSig_200epochs_mae_5e3.torch")




"""LOSS"""

model.load_state_dict(best_model_wts_loss)

print("Evaluating Loss.")

train_name = "./final_ss/socialSig_trainpreds_200epochs_loss_5e3.csv"
val_name = "./final_ss/socialSig_valpreds_200epochs_loss_5e3.csv"

eval(val_dl, model, device).to_csv(val_name, index = False)
eval(train_dl, model, device).to_csv(train_name, index = False)

torch.save({
            'epoch': EPOCHS,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': criterion,
        }, "./final_ss/socialSig_200epochs_loss_5e3.torch")




# print(val_losses_plot)

# plt.plot([i for i in range(0, EPOCHS)], val_losses_plot)
# plt.savefig(("./new_loss_plots/socialSig_100epochs_loss_v2.png"))