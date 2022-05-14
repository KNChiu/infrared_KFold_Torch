#%%
import numpy as np

Re = [0.7, 0.889, 0.889, 0.778, 0.889, 1, 1, 0.75, 1]
mAp50 = [0.6884, 0.9301, 0.9436, 0.6859, 0.7293, 0.9953, 0.8620, 0.7334, 0.9626]
mAp95 = [0.3315, 0.5193, 0.5158, 0.3310, 0.4710, 0.5106, 0.4267, 0.3889, 0.4828]

ACC = [0.82, 0.82, 0.91, 0.64, 0.9, 0.8, 1.0, 0.8, 0.6, 0.9]
AUC = [0.8, 0.83, 0.93, 0.6, 0.92, 0.75, 1.0, 0.81, 0.65, 0.84]

#%%
mean = std = ACC
print(np.mean(mean))
print(np.std(std))
# %%
