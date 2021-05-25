import matplotlib.pyplot as plt
from copy import deepcopy
import pandas as pd
import numpy as np
import importlib
import sklearn
import random
import torch
import math

from helpers import *

import time


def scale(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret.astype(int)


def construct_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    ^^ fix that explanation yo lol
    '''
    print(dim)
    print(length)
    indices = []
    weights = scale(weights.clone().detach().numpy())
    print(weights.size)
    for i in range(0, dim):
        to_add = i * length
        cur_indices = [i + to_add for i in weights]
        indices.append(cur_indices)
    return torch.tensor(indices, dtype = torch.int64)

def scale_noOverlap(x, out_range=(0, 29)):
    '''
    Takes as input the coordinate weights and scales them between 0 and len(weights)
    Dan removed int rounding from this one.
    '''
    domain = np.min(x), np.max(x)
    y = (x - (domain[1] + domain[0]) / 2) / (domain[1] - domain[0])
    to_ret = y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2
    return to_ret


def construct_noOverlap_indices(weights, dim, length):
    '''
    The coordinate weights are between 0-len(weights) but the size of X is len(weights) * batch size so the torch.taken
    function will only take items at indices between  0 & len(weights) meaning only the first item in the batch. This function
    adds len(weights) to each index so taken grabs from every batch
    Dan then modified whatever the above was to ensure the rounding only occurs to available indices, precluding
    drop out. 
    ^^ fix that explanation yo lol
    '''
    indices = []
    weights = scale_noOverlap(weights.clone().cpu().detach().numpy())
    indices = dim*[[x for _,x in sorted(zip(weights,range(0,length)))]]
    for i in range(0,len(indices)):
        indices[i] = [x+(i*length) for x in indices[i]]
    return torch.tensor(indices, dtype = torch.int64).cuda()#.to("cuda:0")


def update_function(param, grad, loss, learning_rate):
    '''
    Calculates the new coordinate weights based on the LR and gradient
    '''
    return param - learning_rate * grad.mean(axis = 0)


def mae(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return torch.abs(real - pred).mean()


def show_image(best_epoch):
    '''
    Takes as input an epoch number and displays the SocialSig from that epoch
    '''
    df = pd.read_csv("./figs/im" + str(best_epoch) + ".csv")
    df["0"] = df["0"].str.split("(").str[1].str.split(",").str[0].astype(float)
    plt.imshow(np.reshape(np.array(df["0"]), (10, 10)))


def train_test_split_weighted(X, y, w, split):

    train_num = int(len(X) * split)
    val_num = int(len(X) - train_num)

    all_indices = list(range(0, len(X)))
    train_indices = random.sample(range(len(X)), train_num)
    val_indices = list(np.setdiff1d(all_indices, train_indices))

    x_train, x_val = X[train_indices], X[val_indices]
    w_train, w_val = w[train_indices], w[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return x_train, y_train, x_val, y_val, w_train, w_val



def train_test_split(X, y, split):

    train_num = int(len(X) * split)
    val_num = int(len(X) - train_num)

    all_indices = list(range(0, len(X)))
    train_indices = random.sample(range(len(X)), train_num)
    val_indices = list(np.setdiff1d(all_indices, train_indices))

    x_train, x_val = X[train_indices], X[val_indices]
    y_train, y_val = y[train_indices], y[val_indices]

    return x_train, y_train, x_val, y_val




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



def read_muni_list(path):
    f = open(path, "r")
    munis = f.read().splitlines()
    f.close()
    munis = [int(i) for i in munis]
    return munis





def train_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr):

    start_time = time.perf_counter()

    best_mae = 9000000000000000000
    best_model_wts_mae = deepcopy(model.state_dict())
    best_model_wts_loss = deepcopy(model.state_dict())

    val_losses_plot = []


    for epoch in range(epochs):

        for phase in ['train','val']:

            if phase == 'train':

                c = 1
                running_train_mae, running_train_loss = 0, 0

                for inputs, output in train:

                    if len(inputs) == batchSize:

                        inputs = inputs.to(device)
                        output = output.to(device)

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = criterion(y_pred, output)  
                        
                        # Zero gradients, perform a backward pass, and update the weights.
                        optimizer.zero_grad()
                        grad = torch.autograd.grad(outputs = loss, inputs = inputs, retain_graph = True)
                        loss.backward()
                        optimizer.step()

                        # Update the coordinate weights
                        # https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/4
                        with torch.no_grad():
                            for name, p in model.named_parameters():
                                if name == 'SocialSig.W':
                                    new_val = update_function(p, grad[0], loss, lr)
                                    p.copy_(new_val)

                        running_train_mae += mae(y_pred, output).item()
                        running_train_loss += loss.item()
                        
                        # print(c)
                        c += 1

            if phase == 'val':

                d = 1
                running_val_mae, running_val_loss,  = 0, 0

                for inputs, output in val:

                    if len(inputs) == batchSize:

                        inputs = inputs.to(device)
                        output = output.to(device)

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = criterion(y_pred, output)  

                        running_val_mae += mae(y_pred, output).item()
                        running_val_loss += loss.item()
                        

                        # print(d)
                        d += 1
                        


                        
                        
        print("Epoch: ", epoch)  
        print("  Train:")
        print("    Loss: ", running_train_loss / c)      
        print("    MAE: ", running_train_mae / c)
        print("  Val:")
        print("    Loss: ", running_val_loss / d)      
        print("    MAE: ", running_val_mae / d)


        val_losses_plot.append(running_val_loss / d)
        

        if (running_val_mae / d) < best_mae:
            best_mae = (running_val_mae / d)
            best_model_wts_mae = deepcopy(model.state_dict())

        if (running_val_loss / d) < best_loss:
            best_loss = (running_val_loss / d)
            best_model_wts_loss = deepcopy(model.state_dict())

            # # Save each best epoch
            # fname = "./epochs/socialSig_MEX_12epochs_AdamLoss_epoch" + str(epoch) + "_real.torch"
            # torch.save({
            #             'epoch': 50,
            #             'model_state_dict': model.state_dict(),
            #             'optimizer_state_dict': optimizer.state_dict(),
            #             'loss': criterion,
            #         }, fname)

            print("  Saving current weights to epochs folder.")
        
        print("\n")

    end_time = time.perf_counter()
    print("Best MAE: ", best_mae)
    print("Training completed in: ", ((end_time - start_time) / 60) / 60, "hours.")
    print("\n")

    return best_model_wts_mae, best_model_wts_loss, val_losses_plot





def weighted_loss(pred, true, weight):
    return torch.sum((abs(pred - true) * weight))



def train_weighted_model(model, train, val, criterion, optimizer, epochs, batchSize, device, lr):

    start_time = time.perf_counter()

    best_mae = 9000000000000000000
    best_loss = 9000000000000000000
    best_model_wts_mae = deepcopy(model.state_dict())
    best_model_wts_loss = deepcopy(model.state_dict())


    val_losses_plot = []

    for epoch in range(epochs):

        for phase in ['train','val']:

            if phase == 'train':

                c = 1
                running_train_mae, running_train_loss, running_train_r2 = 0, 0, 0

                for inputs, output, weights in train:

                    if len(inputs) == batchSize:

                        inputs = inputs.to(device)
                        output = output.to(device)
                        weights = torch.reshape(weights.to(device), (batchSize,1))

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = weighted_loss(y_pred, output, weights)  

                        # Zero gradients, perform a backward pass, and update the weights.
                        optimizer.zero_grad()
                        grad = torch.autograd.grad(outputs = loss, inputs = inputs, retain_graph = True)
                        loss.backward()
                        optimizer.step()

                        # Update the coordinate weights
                        # https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/4
                        with torch.no_grad():
                            for name, p in model.named_parameters():
                                if name == 'SocialSig.W':
                                    new_val = update_function(p, grad[0], loss, lr)
                                    p.copy_(new_val)

                        running_train_mae += mae(y_pred, output).item()
                        running_train_loss += loss.item()
                        running_train_r2 += r2(y_pred, output).item()
                        
                        # print(c)
                        c += 1

            if phase == 'val':

                d = 1
                running_val_mae, running_val_loss, running_val_r2  = 0, 0, 0

                for inputs, output, weights in val:

                    if len(inputs) == batchSize:

                        inputs = inputs.to(device)
                        output = output.to(device)
                        weights = torch.reshape(weights.to(device), (batchSize,1))

                        inputs = torch.tensor(inputs, dtype = torch.float32, requires_grad = True)
                        output = torch.reshape(torch.tensor(output, dtype = torch.float32, requires_grad = True), (batchSize,1))

                        # Forward pass
                        y_pred = model(inputs)
                        loss = weighted_loss(y_pred, output, weights)  

                        running_val_mae += mae(y_pred, output).item()
                        running_val_loss += loss.item()
                        running_val_r2 += r2(y_pred, output).item()

                        # print(d)
                        d += 1
                        
        print("Epoch: ", epoch)  
        print("  Train:")
        print("    Loss: ", running_train_loss / c)      
        print("    MAE: ", running_train_mae / c)
        # print("    R2: ", running_train_r2 / c)
        print("  Val:")
        print("    Loss: ", running_val_loss / d)      
        print("    MAE: ", running_val_mae / d)
        # print("    R2: ", running_val_r2 / d)


        val_losses_plot.append(running_val_loss / d)
        

        if (running_val_mae / d) < best_mae:
            best_mae = (running_val_mae / d)
            best_model_wts_mae = deepcopy(model.state_dict())
            print("  New best MAE.")

        if (running_val_loss / d) < best_loss:
            best_loss = (running_val_loss / d)
            best_model_wts_loss = deepcopy(model.state_dict())
            print("  New best loss.")

            print("  Saving current weights to epochs folder.")
        
        print("\n")

    end_time = time.perf_counter()
    print("Best MAE: ", best_mae)
    print("Training completed in: ", ((end_time - start_time) / 60) / 60, "hours.")
    print("\n")

    return best_model_wts_mae, best_model_wts_loss, val_losses_plot





def eval_model(X, y, sending, size, model, device):

    preds, ids, true_vals = [], [], []

    for i in range(0, len(X)):

        print(i)
        
        input = torch.reshape(torch.tensor(X[i], dtype = torch.float32), size).to(device)
        model.eval()
        pred = model(input).detach().cpu().numpy()[0][0]
        true_val = y[i]#.detach().cpu().numpy()
        cur_id = sending[i]
        true_vals.append(true_val)
        preds.append(pred)
        ids.append(cur_id)

    # Make data frame
    df = pd.DataFrame()
    df['sending_id'] = ids
    df['true'] = true_vals
    df['pred'] = preds
    df['abs_error'] = abs(df['true'] - df['pred'])
    df['error'] = df['true'] - df['pred']

    return df




def r2(true, pred):
    '''
    r2 = 1 - (RSS / TSS)
    R^2	=	coefficient of determination
    RSS	=	sum of squares of residuals
    TSS	=	total sum of squares
    '''
    m = torch.mean(true)
    TSS = sum((true - m) ** 2)
    RSS = sum((true - pred) ** 2)
    r2 = 1 - (RSS / TSS)
    return r2




def mae_np(real, pred):
    '''
    Calculates MAE of an epoch
    '''
    return abs(real - pred).mean()


def r2_np(true, pred):
    '''
    r2 = 1 - (RSS / TSS)
    R^2	=	coefficient of determination
    RSS	=	sum of squares of residuals
    TSS	=	total sum of squares
    '''
    m = np.mean(true)
    TSS = sum((true - m) ** 2)
    RSS = sum((true - pred) ** 2)
    r2 = 1 - (RSS / TSS)
    return r2
