import torch
import torch.nn as nn

import sys
import time
from tqdm import tqdm

sys.path.insert(0, "../tab-ddpm/tab_ddpm/")
from utils_pass import perturb_pass_gaussian_laplace, perfect_permute


class MyDDPM(nn.Module):
    def __init__(
        self,
        network,
        n_steps=200,
        min_beta=10**-4,
        max_beta=0.02,
        device=None,
    ):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device
        )  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor(
            [torch.prod(self.alphas[: i + 1]) for i in range(len(self.alphas))]
        ).to(device)

    def forward(self, x0, t, eta=None):
        """
        Predict noisy sample x_t
        - t should be 1-d tensor of indices with the same length as x0
        """
        # add noise
        a_bar = self.alpha_bars[t]
        n = len(a_bar)

        if eta is None:
            eta = torch.randn_like(x0).to(self.device)

        x0_noisy = (
            a_bar.sqrt().reshape(n, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1) * eta
        )
        return x0_noisy

    def backward(self, x, t):
        # predict added noise
        return self.network(x, t)


def training_loop(yx, ddpm, n_epochs, optim, device=None, store_path="tabular_ddpm.pt"):
    mse = nn.MSELoss()
    best_loss = float("inf")
    n_steps = ddpm.n_steps

    if device is None:
        device = ddpm.device

    for epoch in tqdm(range(n_epochs), desc=f"Training progress", leave=False):

        x0 = yx.to(device)
        n = len(x0)

        # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
        eta = torch.randn_like(x0).to(device)
        t = torch.randint(0, n_steps, (n,)).to(device)

        # Computing the noisy image based on x0 and the time-step (forward process)
        noisy_imgs = ddpm(x0, t, eta)

        # Getting model estimation of noise based on the images and the time-step
        eta_theta = ddpm.backward(noisy_imgs, t)

        # Optimizing the MSE between the noise plugged and the predicted noise
        loss = mse(eta_theta, eta)
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Keep track of the loss and store the best model so far
        epoch_loss = loss.item()
        log_string = f"Loss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # torch.save(ddpm.state_dict(), store_path)

        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), store_path)
            log_string += f" --> Best model ever (stored to {store_path})"

        print(log_string)


def generate_samples(
    ddpm,
    n_samples=16,
    device=None,
    tabular_dim=8,
):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, tabular_dim).to(device)

        looper = tqdm(
            enumerate(list(range(ddpm.n_steps))[::-1]), total=ddpm.n_steps, leave=False
        )
        for idx, t in looper:
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Denoising
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            if t > 0:
                z = torch.randn(n_samples, tabular_dim).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            looper.set_description(f"Imputation at step {t}")

    return x


# I realize that the jump step is for the resampling, i.e., how much should x jump back during resampling
def generate_imputation(
    ddpm,
    yx,
    mask_yx,
    resampling_steps=1,
    device=None,
):
    """
    Missing value imputation using unconditional DDPM.
    - yx: input data (even missing values should be specified as values)
    - mask_yx: binary mask indicating missing values (1 for observed, 0 for missing)
    - resampling_steps: number of resampling steps during each denoising step. 10 steps usually give good result (though defaults to 1)
    """

    assert yx.shape == mask_yx.shape, "yx and mask_yx should have the same shape"

    n_samples = yx.shape[0]
    x0, m = yx.clone(), mask_yx.clone()

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        x0, m = x0.to(device), m.to(device)

        # Starting from random noise
        x = torch.randn_like(x0).to(device)

        looper = tqdm(
            enumerate(list(range(ddpm.n_steps))[::-1]), total=ddpm.n_steps, leave=False
        )
        for idx, t in looper:
            time_tensor = (torch.ones(n_samples) * t).to(device).long()
            for u in range(resampling_steps):
                # retrieve the known part from original data
                x_known = ddpm(x0, time_tensor)

                # denoise the unknown part from noisy data
                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                eta_theta = ddpm.backward(x, time_tensor)
                z = torch.randn_like(x0).to(device) if t > 0 else 0
                x_unknown = (1 / alpha_t.sqrt()) * (
                    x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
                ) + sigma_t * z

                # impute the missing values in intermediate steps
                x = m * x_known + (1 - m) * x_unknown

                # resampling
                if u < resampling_steps - 1 and t > 0:
                    x = (1 - beta_t).sqrt() * x + beta_t.sqrt() * torch.randn_like(
                        x0
                    ).to(device)

            looper.set_description(f"Imputation at step {t}")

    return x


def generate_pass(ddpm, yx_norm, perturbation_size=0, device=None):
    """
    Generate a PASS sample based on rank matching with yx on latent space.
    """

    n_samples, tabular_dim = yx_norm.shape

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn_like(yx_norm)

        # Rank matching
        x = perturb_pass_gaussian_laplace(x, perturbation_size)
        print("Begin matching ranks ...")
        start = time.time()
        x, _ = perfect_permute(x, yx_norm)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        print("Rank matching done in {:.2f} seconds.".format(time.time() - start))

        looper = tqdm(
            enumerate(list(range(ddpm.n_steps))[::-1]), total=ddpm.n_steps, leave=False
        )
        for idx, t in looper:
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Denoising
            x = (1 / alpha_t.sqrt()) * (
                x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta
            )

            if t > 0:
                z = torch.randn(n_samples, tabular_dim).to(device)

                # sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            looper.set_description(f"Imputation at step {t}")

    return x
