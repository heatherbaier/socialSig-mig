import matplotlib.pyplot as plt
import pandas as pd

## MEX -> US: Transfer & Not Weighted

t_nw = pd.read_csv("./predictions/transfer_25epoch_normalloss_us_preds.csv")
plt.scatter(t_nw['true'], t_nw['pred'])
plt.xlim([0,3000])
plt.title("MEX -> US: Transfer & Not Weighted \n(True cut-off at 3000, real max = 38500)")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./model_plots/t_nw.png")
plt.clf()

## MEX -> US: No Transfer & Not Weighted
nt_nw = pd.read_csv("./predictions/notransfer_25epoch_normalloss_us_preds.csv")
plt.scatter(nt_nw['true'], nt_nw['pred'])
plt.xlim([0,3000])
plt.title("MEX -> US: No Transfer & Not Weighted\n (True cut-off at 3000, real max = 38500)")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./model_plots/nt_nw.png")
plt.clf()

## MEX -> US: Transfer & Weighted
t_w = pd.read_csv("./predictions/transfer_25epoch_weightedloss_us_preds.csv")
plt.scatter(t_w['true'], t_w['pred'])
plt.xlim([0,3000])
plt.title("MEX -> US: Transfer & Weighted \n(True cut-off at 3000, real max = 38500)")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./model_plots/t_w.png")
plt.clf()

## MEX -> US: No Transfer & Weighted
nt_w = pd.read_csv("./predictions/notransfer_25epoch_weightedloss_us_preds2.csv")
plt.scatter(nt_w['true'], nt_w['pred'])
plt.xlim([0,3000])
plt.title("MEX -> US: No Transfer & Weighted \n(True cut-off at 3000, real max = 38500)")
plt.xlabel("True")
plt.ylabel("Predicted")
plt.savefig("./model_plots/nt_w.png")
plt.clf()