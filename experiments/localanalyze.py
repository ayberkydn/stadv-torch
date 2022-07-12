#%%
from itertools import product
from kornia import tensor_to_image
from matplotlib import pyplot as plt
import numpy as np
import torch
import os, pickle


colors = []
means = []
stds = []
for n in range(1000):
    mode = "lab"
    res = True
    runpath = f"./results/{mode}_{res}_{n}"
    dictpath = os.path.join(runpath, "dict.pkl")
    with open(dictpath, "rb") as f:
        res = pickle.load(f)
    # print(res['success'])
    colors.append(res["colorfulness"])
    means.append(res["meanroot"])
    stds.append(res["stdroot"])
colors = np.array(colors)
means = np.array(means)
stds = np.array(stds)
#%%

#%%
bins = 50
plt.hist(colors, color=[0.5, 0.2, 1], bins=bins, cumulative=True)
plt.title("Colorfulness histogram")
plt.show()

#%%
# modes = ["lab", "ycbcr", "rgb"]


def rate_color(mode, restricted, colorlimit):
    """
    returns total, success, fooled
    from the portion of dataset
    whose colorfulness>colorlimit
    """
    success = 0
    fooled = 0
    total = 0
    for n in range(1000):
        runpath = f"./results/{mode}_{restricted}_{n}"
        dictpath = os.path.join(runpath, "dict.pkl")
        with open(dictpath, "rb") as f:
            res = pickle.load(f)
        if res["colorfulness"] >= colorlimit:
            total += 1
            if res["success"] == True:
                success += 1
            if res["fooled"] == True:
                fooled += 1
    return success, fooled, total


mode = "ycbcr"
restricted = True
colorrange = np.linspace(colors.min(), colors.max(), bins)
successrates = []
foolingrates = []
for colorlimit in colorrange:
    success, fooled, total = rate_color(mode, restricted, colorlimit)
    successrates.append(success / total)
    foolingrates.append(fooled / total)

plt.style.use("seaborn-dark")
plt.hist(colors, color=[0.5, 0.2, 1], bins=bins, cumulative=True)
plt.title("Colorfulness CDF of the dataset")
plt.xlabel("Colorfulness")
plt.show()
plt.plot(colorrange, foolingrates)
plt.title("Colorfulness fooling rate")
plt.xlabel("Colorfulness")
plt.show()
