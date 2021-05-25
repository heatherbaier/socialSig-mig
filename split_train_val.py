import pandas as pd
import numpy as np
import random

df = pd.read_csv("./data/mexico2010.csv")

munis = df.GEO2_MX
split = .80

train_num = int(len(munis) * split)
val_num = int(len(munis) - train_num)

all_indices = list(range(0, len(munis)))
train_indices = random.sample(range(len(munis)), train_num)
val_indices = list(np.setdiff1d(all_indices, train_indices))

train_munis, val_munis = np.array(munis)[train_indices], np.array(munis)[val_indices]

outfile = open('train_municipalities.txt', 'w')
[outfile.write(str(L) + "\n") for L in train_munis]
outfile.close()

outfile = open('val_municipalities.txt', 'w')
[outfile.write(str(L) + "\n") for L in val_munis]
outfile.close()