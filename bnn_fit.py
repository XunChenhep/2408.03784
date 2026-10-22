"""Bayesian neural network regression tutorial. Run this file directly.

Dependencies: torch, numpy, matplotlib (already installed on this machine).
Generate demo data with 1-sigma error bars, or load a CSV with columns x,y,yerr.
Method: Bayes by Backprop with an independent Gaussian variational posterior
for every weight and bias.
Reference: https://proceedings.mlr.press/v37/blundell15.html

Run: python bnn_fit.py
Use your own data: python bnn_fit.py --csv my_data.csv
Save without opening a plot window: python bnn_fit.py --no-show
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


# ===== 1. Start here when experimenting with the settings =====
@dataclass
class Settings:
    hidden: int = 16       # Architecture: 1 -> hidden -> 1, with tanh activation
    prior_std: float = 1.0 # Prior for every weight and bias: Normal(0, prior_std^2)
    steps: int = 6000
    learning_rate: float = 0.01
    train_samples: int = 16 # Weight samples per step to estimate the expected log likelihood
    prediction_samples: int = 2000
    seed: int = 7


def true_function(x):
    """Generate demo data and a reference curve; not used during training."""
    return np.sin(x) + 0.3 * x


def noise_std(x):
    """Known measurement standard deviation for the demo, varying with x."""
    return 0.10 + 0.03 * np.abs(x)


def load_data(csv_path=None, seed=7):
    """Provide three 1D arrays, x, y and yerr, to use your own data.

    yerr contains the 1-sigma standard deviation of each y measurement,
    rather than its variance or the width of a 95% interval.
    This example assumes exact x values and independent Gaussian errors in y.
    """
    if csv_path is None:
        rng = np.random.default_rng(seed)
        x = np.linspace(-2.0, 2.0, 40)
        yerr = noise_std(x)
        y = true_function(x) + rng.normal(0.0, yerr)
    else:
        data = np.genfromtxt(csv_path, delimiter=",", names=True, encoding="utf-8-sig")
        if data.dtype.names != ("x", "y", "yerr"):
            raise ValueError("CSV 表头必须依次为 x,y,yerr。")
        x, y, yerr = (np.atleast_1d(data[key]) for key in ("x", "y", "yerr"))
    if len(x) < 4 or not all(np.isfinite(a).all() for a in (x, y, yerr)):
        raise ValueError("至少需要 4 个数据点，且所有数值必须有限。")
    if np.any(yerr <= 0) or np.ptp(x) == 0:
        raise ValueError("yerr 必须全部大于 0，x 至少应有两个不同取值。")
    order = np.argsort(x)
    return x[order], y[order], yerr[order]


# ===== 2. Bayesian linear layer: represent each weight by a distribution =====
class BayesianLinear(nn.Module):
    def __init__(self, n_in, n_out, prior_std=1.0):
        super().__init__()
        self.prior_std = prior_std
        self.weight_mu = nn.Parameter(torch.randn(n_out, n_in) * 0.15)
        self.weight_rho = nn.Parameter(torch.full((n_out, n_in), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(n_out))
        self.bias_rho = nn.Parameter(torch.full((n_out,), -3.0))

    @staticmethod
    def std(rho):
        return F.softplus(rho) + 1e-6  # Keep standard deviations positive

    def forward(self, x):
        # x shape: [number of samples, number of data points, input dimension].
        # Reparameterize as w = mu + sigma * epsilon, epsilon ~ Normal(0,1).
        count = x.shape[0]
        weight = self.weight_mu + self.std(self.weight_rho) * torch.randn(
            count, *self.weight_mu.shape, dtype=x.dtype, device=x.device)
        bias = self.bias_mu + self.std(self.bias_rho) * torch.randn(
            count, 1, *self.bias_mu.shape, dtype=x.dtype, device=x.device)
        # Use each sampled set of weights for all x to obtain a full function draw.
        return torch.bmm(x, weight.transpose(1, 2)) + bias

    def kl(self):
        """Analytic KL[Normal(mu,sigma^2) || Normal(0,prior_std^2)]."""
        result = self.weight_mu.new_zeros(())
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            std = self.std(rho)
            result = result + (torch.log(self.prior_std / std)
                               + (std**2 + mu**2) / (2 * self.prior_std**2) - 0.5).sum()
        return result


class BayesianNN(nn.Module):
    def __init__(self, hidden=16, prior_std=1.0):
        super().__init__()
        self.layer1 = BayesianLinear(1, hidden, prior_std)
        self.layer2 = BayesianLinear(hidden, 1, prior_std)

    def forward(self, x, n_samples=1):
        # Return [number of samples, number of data points]; resample on each call.
        x = x.reshape(1, -1, 1).expand(n_samples, -1, -1)
        return self.layer2(torch.tanh(self.layer1(x))).squeeze(-1)

    def kl(self):
        return self.layer1.kl() + self.layer2.kl()


# ===== 3. Build a Gaussian likelihood from error bars and fit the posterior =====
def gaussian_nll(predictions, y, yerr):
    # y_i | w ~ Normal(f_w(x_i), yerr_i^2).
    # Sum over data points to obtain the full-data NLL for each weight sample.
    return (0.5 * ((predictions - y) / yerr)**2
            + yerr.log() + 0.5 * math.log(2 * math.pi)).sum(dim=-1)


def train(x, y, yerr, settings):
    model = BayesianNN(settings.hidden, settings.prior_std)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.steps, eta_min=settings.learning_rate * 0.05)
    history = []
    for step in range(1, settings.steps + 1):
        optimizer.zero_grad()
        predictions = model(x, settings.train_samples)
        nll = gaussian_nll(predictions, y, yerr).mean() # Then average over weight samples
        kl = model.kl()
        # Minimize the negative ELBO = E_q[-log p(data|w)] + KL(q||p).
        # Dividing both terms by N preserves their relative weighting.
        loss = (nll + kl) / len(x)
        if not torch.isfinite(loss):
            raise RuntimeError("损失非有限，请检查数据和参数。")
        loss.backward()
        optimizer.step()
        scheduler.step()
        history.append([step, float(loss.detach()), float(nll.detach()) / len(x), float(kl.detach()) / len(x)])
        if step == 1 or step % 1000 == 0:
            print(f"step {step:5d}: negative_ELBO/N={float(loss.detach()):.5f}", flush=True)
    return model, np.asarray(history)


# ===== 4. Posterior prediction: separate function and observation intervals =====
@torch.no_grad()
def posterior_prediction(model, x, yerr, n_samples=2000):
    function_samples = model(x, n_samples)
    observation_samples = function_samples + yerr * torch.randn_like(function_samples)
    quantiles = torch.tensor([0.025, 0.975], dtype=x.dtype)
    return {
        "mean": function_samples.mean(dim=0).numpy(),
        "function_interval": torch.quantile(function_samples, quantiles, dim=0).numpy(),
        "observation_interval": torch.quantile(observation_samples, quantiles, dim=0).numpy(),
        "function_std": function_samples.std(dim=0).numpy(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, help="自己的数据：带表头的三列 CSV，x,y,yerr")
    parser.add_argument("--steps", type=int, default=Settings.steps)
    parser.add_argument("--no-show", action="store_true", help="仅保存图片，不弹出窗口")
    args = parser.parse_args()
    settings = Settings(steps=args.steps)
    if settings.steps < 1:
        parser.error("--steps 必须大于 0")
    # Set the backend before importing pyplot so --no-show works without a display.
    import matplotlib
    if args.no_show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.manual_seed(settings.seed)
    x_raw, y_raw, err_raw = load_data(args.csv, settings.seed)
    # Standardize for stable training; divide error bars by the same y_scale.
    x_center, x_scale = x_raw.mean(), x_raw.std()
    y_center, y_scale = y_raw.mean(), max(y_raw.std(), 1e-8)
    x = torch.tensor((x_raw - x_center) / x_scale)
    y = torch.tensor((y_raw - y_center) / y_scale)
    yerr = torch.tensor(err_raw / y_scale)
    start = time.perf_counter()
    model, history = train(x, y, yerr, settings)

    # Plot the observed range only; these intervals do not establish extrapolation accuracy.
    plot_x = np.linspace(x_raw.min(), x_raw.max(), 401)
    plot_error = noise_std(plot_x) if args.csv is None else np.interp(plot_x, x_raw, err_raw)
    # For user data, interpolate the errors for prediction; the BNN does not learn this noise model.
    torch.manual_seed(settings.seed + 1) # Use a separate, reproducible prediction random stream
    pred = posterior_prediction(model, torch.tensor((plot_x - x_center) / x_scale),
                                torch.tensor(plot_error / y_scale), settings.prediction_samples)
    pred["mean"] = pred["mean"] * y_scale + y_center
    for key in ("function_interval", "observation_interval"):
        pred[key] = pred[key] * y_scale + y_center
    pred["function_std"] *= y_scale
    with torch.no_grad():
        train_mean = model(x, settings.prediction_samples).mean(dim=0).numpy() * y_scale + y_center
    metrics = {
        "n_data": len(x_raw),
        "train_rmse": float(np.sqrt(np.mean((train_mean - y_raw)**2))),
        "mean_squared_standardized_residual": float(np.mean(((train_mean - y_raw) / err_raw)**2)),
        "elapsed_seconds": time.perf_counter() - start,
        "inference": "mean-field Gaussian variational inference (Bayes by Backprop)",
        "intervals": "pointwise 95%, under the variational approximation",
        "seed": settings.seed,
    }
    if args.csv is None:
        inside = (plot_x >= x_raw.min()) & (plot_x <= x_raw.max())
        truth = true_function(plot_x)
        metrics["demo_truth_rmse_in_data_range"] = float(np.sqrt(np.mean((pred["mean"][inside] - truth[inside])**2)))
    out = Path(__file__).resolve().parent / "runs" / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    np.savetxt(out / "data.csv", np.column_stack((x_raw, y_raw, err_raw)), delimiter=",", header="x,y,yerr", comments="")
    np.savetxt(out / "predictions.csv", np.column_stack((plot_x, pred["mean"], pred["function_interval"].T,
               pred["observation_interval"].T, pred["function_std"])), delimiter=",",
               header="x,mean,function_2.5pct,function_97.5pct,observation_2.5pct,observation_97.5pct,function_std", comments="")
    np.savetxt(out / "loss.csv", history, delimiter=",", header="step,negative_ELBO_per_point,NLL_per_point,KL_per_point", comments="")
    torch.save({"state_dict": model.state_dict(), "settings": asdict(settings),
                "normalization": {"x_center": float(x_center), "x_scale": float(x_scale),
                                  "y_center": float(y_center), "y_scale": float(y_scale)}}, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    ax = axes[0]
    ax.fill_between(plot_x, *pred["observation_interval"], color="#f2c681", alpha=0.65, label="95% new-observation interval")
    ax.fill_between(plot_x, *pred["function_interval"], color="#5ca7d1", alpha=0.7, label="95% latent-function interval")
    ax.plot(plot_x, pred["mean"], color="#175481", label="BNN posterior mean")
    ax.errorbar(x_raw, y_raw, yerr=err_raw, fmt="o", color="#242424", ms=3, capsize=2, label="Data with 1-sigma errors")
    if args.csv is None:
        ax.plot(plot_x, true_function(plot_x), "--", color="#65876b", lw=1.5, label="Truth (demo only)")
    ax.set(xlabel="x", ylabel="y", title="Bayesian neural network: direct regression")
    ax.legend(fontsize=8, loc="best")
    window = min(100, len(history))
    smooth = np.convolve(history[:, 1], np.ones(window) / window, mode="valid")
    axes[1].plot(history[:, 0], history[:, 1], alpha=0.2, color="#5ca7d1", label="Monte Carlo objective")
    axes[1].plot(history[window-1:, 0], smooth, color="#175481", label=f"{window}-step moving average")
    axes[1].set(xlabel="Training step", ylabel="Negative ELBO / N", title="Training objective (lower is better)")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(out / "fit.png", dpi=180)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"\n图和数据已保存到：{out}\n拟合图：{out / 'fit.png'}", flush=True)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
