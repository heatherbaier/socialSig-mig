
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from helpers import r2_np, mae_np

## DenseNet Validation Set
dn_val = pd.read_csv("./densenet_valpreds_100epochs_v2.csv")
dn_mae = mae_np(dn_val['true'], dn_val['pred'])
dn_r2 = r2_np(dn_val['true'], dn_val['pred'])
plt.figure(figsize=(7, 6))
plt.scatter(dn_val['true'], dn_val['pred'])
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.title("DenseNet Validation Set\n R2: " + str(round(dn_r2, 2)) + "\n MAE: " + str(round(dn_mae, 2)))
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./final_model_plots/densenet_val.png")
plt.clf()


## DenseNet Training Set 
dn_train = pd.read_csv("./densenet_trainpreds_100epochs_v2.csv")
dn_mae = mae_np(dn_train['true'], dn_train['pred'])
dn_r2 = r2_np(dn_train['true'], dn_train['pred'])
plt.figure(figsize=(7, 6))
plt.scatter(dn_train['true'], dn_train['pred'])
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.title("DenseNet Training Set\n R2: " + str(round(dn_r2, 2)) + "\n MAE: " + str(round(dn_mae, 2)))
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./final_model_plots/densenet_train.png")
plt.clf()


## socialSig Validation Set
ss_val = pd.read_csv("./socialSig_valpreds_100epochs_loss.csv")
ss_mae = mae_np(ss_val['true'], ss_val['pred'])
ss_r2 = r2_np(ss_val['true'], ss_val['pred'])
plt.figure(figsize=(7, 6))
plt.scatter(ss_val['true'], ss_val['pred'])
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.title("DenseNet Validation Set\n R2: " + str(round(ss_r2, 2)) + "\n MAE: " + str(round(ss_mae, 2)))
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./final_model_plots/socialSig_val.png")
plt.clf()


## socialSig Training Set 
ss_train = pd.read_csv("./socialSig_trainpreds_100epochs_loss.csv")
ss_mae = mae_np(ss_train['true'], ss_train['pred'])
ss_r2 = r2_np(ss_train['true'], ss_train['pred'])
plt.figure(figsize=(7, 6))
plt.scatter(ss_train['true'], ss_train['pred'])
plt.xlim([0, 20000])
plt.ylim([0, 20000])
plt.title("DenseNet Training Set\n R2: " + str(round(ss_r2, 2)) + "\n MAE: " + str(round(ss_mae, 2)))
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./final_model_plots/socialSig_train.png")
plt.clf()