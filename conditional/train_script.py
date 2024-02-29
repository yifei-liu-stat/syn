import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import QuantileTransformer

from tqdm import tqdm
import joblib

from utils_data import TrueSampler
from utils_model import MLPDiffusionContinuous
from ddpm import MyDDPM, training_loop, generate_samples


seed = 2024
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# Generate simulated data #
n_samples = 3000
true_sampler = TrueSampler(sigma=0.2)
X, y = true_sampler.sample(n_samples)
yx = np.concatenate([y[:, None], X], axis=1)

# save the original data
np.save("./data/yx.npy", yx)

# quantile transformation: ddpm is trained on the quantile-transformed data
qt = QuantileTransformer(output_distribution="normal", random_state=seed)
qt.fit(yx)
yx_norm = qt.transform(yx)
yx_norm = torch.tensor(yx_norm, dtype=torch.float32)

# save the quantile transformation for getting back to the original space
joblib.dump(qt, "./ckpt/qt_train.joblib")


# Initialization #

d_in = yx_norm.shape[1]
d_time = 128
hidden_dims = [512, 256, 256, 256, 256, 128]
n_steps = 1000

n_epochs = 1000
lr = 1e-3
device = "cuda:7"


noise_pred_network = MLPDiffusionContinuous(
    d_in=d_in, hidden_dims=hidden_dims, dim_t=d_time
)
tabular_ddpm = MyDDPM(network=noise_pred_network, n_steps=n_steps, device=device)

# # Example #:
# timestamp = torch.tensor([2], dtype=torch.long).to(device)
# yx_noisy = tabular_ddpm(yx_norm, timestamp)
# yx_noisy_denoised = tabular_ddpm.backward(yx_noisy, timestamp)


# Training #

optimizer = optim.Adam(tabular_ddpm.parameters(), lr=lr)
training_loop(
    yx_norm, tabular_ddpm, n_epochs, optimizer, store_path="./ckpt/tabular_ddpm.pt"
)
