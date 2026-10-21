"""Fit a Cornell potential through the holographic w(r), arXiv:2408.03784.

Only the MLP parameters are optimized. Natural units: r,L in GeV^-1; E in GeV.
Run: python fit_cornell.py --config config.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

HBARC = 0.1973269804  # GeV fm
ADS_I = math.sqrt(math.pi) * math.gamma(0.75) / math.gamma(0.25)


@dataclass
class Config:
    kappa: float = 0.52
    sigma: float = 0.18  # GeV^2
    C: float = 0.0  # GeV; the same fixed energy zero is added to both potentials
    L_min_fm: float = 0.1
    L_max_fm: float = 1.2
    n_train: int = 48
    hidden: tuple[int, ...] = (32, 32)
    r_scale: float = 2.0  # GeV^-1, only input scaling
    n_quad: int = 80
    adam_steps: int = 1000
    learning_rate: float = 0.003
    lbfgs_steps: int = 150
    seed: int = 42
    threads: int = 1

    @property
    def g(self):
        # w(0)=1 fixes the Coulomb coefficient: kappa = 4*g*ADS_I**2.
        return self.kappa / (4 * ADS_I**2)

    def validate(self):
        if self.kappa <= 0 or self.sigma <= 0:
            raise ValueError("This confining example requires kappa > 0 and sigma > 0.")
        if not (0 < self.L_min_fm < self.L_max_fm):
            raise ValueError("Require 0 < L_min_fm < L_max_fm.")
        if self.n_train < 8 or self.n_quad < 16 or self.r_scale <= 0:
            raise ValueError("Require n_train >= 8, n_quad >= 16 and r_scale > 0.")
        if not self.hidden or any(n <= 0 for n in self.hidden):
            raise ValueError("hidden must contain positive layer widths.")
        if self.adam_steps < 0 or self.lbfgs_steps < 0 or self.threads < 1:
            raise ValueError("Invalid optimizer steps or thread count.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not all(math.isfinite(v) for v in asdict(self).values() if isinstance(v, float)):
            raise ValueError("All real-valued configuration entries must be finite.")


def cornell(L, cfg):
    return -cfg.kappa / L + cfg.sigma * L + cfg.C


class WarpMLP(nn.Module):
    """A positive smooth warp factor with exact w(0)=1 and w'(0)=0.

    w(r) = exp[r^2 softplus(N_theta((r/r_scale)^2))].
    The positive exponent is a modeling restriction suitable for this example;
    its r-dependence is learned, not fixed to the AZ quadratic ansatz.
    """
    def __init__(self, cfg):
        super().__init__()
        self.r_scale = cfg.r_scale
        layers = []
        sizes = (1, *cfg.hidden, 1)
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        # AZ-like initialization gives a physical starting branch; no w labels.
        s_initial = cfg.sigma / (math.e * cfg.g)
        nn.init.normal_(self.net[-1].weight, std=0.001)
        nn.init.constant_(self.net[-1].bias, math.log(math.expm1(s_initial)))

    def log_w(self, r):
        x = ((r / self.r_scale)**2).unsqueeze(-1)
        return r**2 * nn.functional.softplus(self.net(x).squeeze(-1))

    def forward(self, r):
        return self.log_w(r).exp()


class HolographicPotential:
    def __init__(self, model, g, n_quad=80):
        self.model, self.g = model, g
        x, weights = np.polynomial.legendre.leggauss(n_quad)
        t = torch.as_tensor((x + 1) / 2)
        self.u = 1 - t**2
        self.weights = torch.as_tensor(weights / 2) * (2 * t)

    def evaluate(self, r0, energy=True):
        """Paper Eqs. (15),(16) at f=1, using u=r/r0=1-t^2.

        expm1 and a rationalized UV subtraction avoid cancellation. Negative
        radicands are errors, never clamped into an artificial real branch.
        """
        r = r0[:, None] * self.u
        logw = self.model.log_w(r)
        logw0 = self.model.log_w(r0)
        logq = 2 * self.u.log() + logw0[:, None] - logw
        radicand = -torch.expm1(2 * logq)
        if not torch.isfinite(radicand).all() or (radicand <= 0).any():
            raise RuntimeError("Invalid U-string branch: w(r)/r^2 must exceed w(r0)/r0^2.")
        s = radicand.sqrt()
        q = logq.exp()
        L = 2 * r0 * ((q / s) * self.weights).sum(dim=1)
        if not energy:
            return L
        # w/s - 1 = (w-1) + w*q^2/[s*(1+s)].
        integrand = (torch.expm1(logw) + logw.exp() * q**2 / (s * (1 + s))) / self.u**2
        E = 2 * self.g / r0 * ((integrand * self.weights).sum(dim=1) - 1)
        return L, E

    def _log_A_derivative(self, r):
        with torch.enable_grad():
            r = r.detach().requires_grad_(True)
            logA = self.model.log_w(r) - 2 * r.log()
            return torch.autograd.grad(logA.sum(), r)[0].detach()

    @torch.no_grad()
    def branch_upper(self):
        """Find the first minimum of A=w/r^2; stay strictly before it."""
        r = torch.logspace(-3, math.log10(30.0), 160)
        derivative = self._log_A_derivative(r)
        hit = torch.where(derivative >= 0)[0]
        if not len(hit):
            return r[-1]
        index = int(hit[0])
        if index == 0:
            raise RuntimeError("No UV-connected decreasing branch was found.")
        low, high = r[index-1].clone(), r[index].clone()
        for _ in range(24):
            mid = (low + high) / 2
            if self._log_A_derivative(mid) < 0:
                low = mid
            else:
                high = mid
        return low * (1 - 1e-4)

    @torch.no_grad()
    def turning_points(self, target_L):
        upper = self.branch_upper()
        upper_L = self.evaluate(upper.reshape(1), energy=False)[0]
        if not torch.isfinite(upper_L) or upper_L < target_L.max():
            raise RuntimeError("The physical branch does not cover the requested L interval.")
        low = torch.zeros_like(target_L)
        high = torch.ones_like(target_L) * upper
        for _ in range(34):
            mid = (low + high) / 2
            below = self.evaluate(mid, energy=False) < target_L
            low = torch.where(below, mid, low)
            high = torch.where(below, high, mid)
        return (low + high) / 2

    def energy_at_fixed_L(self, target_L):
        r0 = self.turning_points(target_L)
        L, E = self.evaluate(r0)
        # Implicit gradient at fixed L, since dE/dr0 = F*dL/dr0.
        # The roots are detached; this correction restores the omitted term.
        force = (self.g * self.model(r0) / r0**2).detach()
        return E - force * (L - target_L)


def ads_check():
    class AdS(nn.Module):
        def log_w(self, r):
            return r * 0
    r0 = torch.tensor([0.15, 0.7, 1.8])
    L, E = HolographicPotential(AdS(), g=0.3, n_quad=120).evaluate(r0)
    error = max(float((L / (2 * ADS_I * r0) - 1).abs().max()),
                float((E / (-0.6 * ADS_I / r0) - 1).abs().max()))
    if error > 1e-7:
        raise RuntimeError(f"Pure AdS analytic check failed: {error}")
    return error


def fit(cfg, out):
    cfg.validate()
    out.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(cfg.threads)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    analytic_error = ads_check()
    model = WarpMLP(cfg)
    potential = HolographicPotential(model, cfg.g, cfg.n_quad)
    L = torch.linspace(cfg.L_min_fm / HBARC, cfg.L_max_fm / HBARC, cfg.n_train)
    target = cornell(L, cfg)
    history = []
    best_loss, best_state = float("inf"), None
    start = time.perf_counter()

    def loss_fn():
        return ((potential.energy_at_fixed_L(L) + cfg.C - target)**2).mean()

    def record(stage, step, loss):
        nonlocal best_loss, best_state
        value = float(loss.detach())
        if not math.isfinite(value):
            raise RuntimeError("Non-finite loss.")
        history.append((len(history), stage, step, value))
        if value < best_loss:
            best_loss = value
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return value

    print(f"g={cfg.g:.10f}; fitting {cfg.L_min_fm}-{cfg.L_max_fm} fm", flush=True)
    record("initial", 0, loss_fn())
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    for step in range(1, cfg.adam_steps + 1):
        optimizer.zero_grad()
        loss = loss_fn()
        value = record("adam", step, loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(f"Adam {step:4d}: RMSE={math.sqrt(value):.6g} GeV", flush=True)
    # Begin refinement at the best actual evaluated model, not an unchecked step.
    record("adam_final", cfg.adam_steps, loss_fn())
    model.load_state_dict(best_state)
    if cfg.lbfgs_steps:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=0.8,
                                     max_iter=cfg.lbfgs_steps, line_search_fn="strong_wolfe",
                                     tolerance_grad=1e-10, tolerance_change=1e-13)
        def closure():
            optimizer.zero_grad()
            loss = loss_fn()
            record("lbfgs", len(history), loss)
            loss.backward()
            return loss
        optimizer.step(closure)
        record("lbfgs_final", len(history), loss_fn())
    model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - start
    metrics = save_results(cfg, out, model, history, elapsed, analytic_error)
    print(json.dumps(metrics, indent=2), flush=True)
    return model, metrics


@torch.no_grad()
def save_results(cfg, out, model, history, elapsed, analytic_error):
    # Twice as dense and shifted relative to training, with 4x quadrature order.
    L_fm = torch.linspace(cfg.L_min_fm, cfg.L_max_fm, 2 * cfg.n_train + 1)
    target_L = L_fm / HBARC
    fine = HolographicPotential(model, cfg.g, 4 * cfg.n_quad)
    r0 = fine.turning_points(target_L)
    actual_L, E_ren = fine.evaluate(r0)
    prediction = E_ren + cfg.C
    target = cornell(target_L, cfg)
    residual = prediction - target
    coarse = HolographicPotential(model, cfg.g, 2 * cfg.n_quad)
    coarse_r0 = coarse.turning_points(target_L)
    _, coarse_E = coarse.evaluate(coarse_r0)
    near_zero = torch.tensor([0.0], requires_grad=True)
    with torch.enable_grad():
        w0 = model(near_zero)
        dw0 = torch.autograd.grad(w0.sum(), near_zero)[0]
    # Independently check a dense branch grid and Hamilton-Jacobi force identity.
    branch_r = torch.linspace(1e-4, float(r0.max()), 2001)
    max_derivative = float(fine._log_A_derivative(branch_r).max())
    turning_grid = torch.linspace(float(r0.min()) / 4, float(r0.max()), 301)
    L_grid = fine.evaluate(turning_grid, energy=False)
    min_dL_dr0 = float(((L_grid[1:] - L_grid[:-1]) /
                       (turning_grid[1:] - turning_grid[:-1])).min())
    eps = 1e-4
    Lp, Ep = fine.evaluate(r0 * (1 + eps))
    Lm, Em = fine.evaluate(r0 * (1 - eps))
    F = cfg.g * model(r0) / r0**2
    force_error = float((((Ep - Em) / (Lp - Lm)) / F - 1).abs().max())
    metrics = {
        "g": cfg.g,
        "dense_rmse_GeV": float(residual.square().mean().sqrt()),
        "dense_max_abs_error_GeV": float(residual.abs().max()),
        "root_max_error_fm": float(((actual_L - target_L) * HBARC).abs().max()),
        "quadrature_2n_vs_4n_max_delta_GeV": float((coarse_E - E_ren).abs().max()),
        "pure_AdS_max_relative_error": analytic_error,
        "force_identity_max_relative_error": force_error,
        "w_at_zero": float(w0.detach()),
        "dw_dr_at_zero": float(dw0),
        "max_d_logA_dr_on_fitted_branch": max_derivative,
        "min_dL_dr0_on_fitted_branch": min_dL_dr0,
        "largest_sampled_r0_GeV_inverse": float(r0.max()),
        "training_seconds": elapsed,
        "torch_version": str(torch.__version__),
        "seed": cfg.seed,
    }
    if metrics["root_max_error_fm"] > 1e-6 or max_derivative >= 0 or min_dL_dr0 <= 0:
        raise RuntimeError(f"Final root/physical branch validation failed: {metrics}")
    if metrics["quadrature_2n_vs_4n_max_delta_GeV"] > 1e-5 or force_error > 1e-4:
        raise RuntimeError(f"Final quadrature/force validation failed: {metrics}")
    torch.save({"config": asdict(cfg), "g": cfg.g, "state_dict": model.state_dict()}, out / "model.pt")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (out / "potential.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["L_fm", "L_GeV_inverse", "V_Cornell_GeV", "V_MLP_GeV", "residual_GeV", "r0_GeV_inverse"])
        writer.writerows(zip(L_fm.tolist(), target_L.tolist(), target.tolist(),
                             prediction.tolist(), residual.tolist(), r0.tolist()))
    r = torch.linspace(0, float(r0.max()), 301)
    with (out / "warp_factor.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["r_GeV_inverse", "w_MLP"])
        writer.writerows(zip(r.tolist(), model(r).tolist()))
    with (out / "loss.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["evaluation", "optimizer", "step", "MSE_GeV_squared"])
        writer.writerows(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    axes[0].plot(L_fm, target, color="black", label="Cornell target")
    axes[0].plot(L_fm, prediction, "--", color="#d35400", label="Holographic MLP")
    axes[0].set(xlabel="L [fm]", ylabel="V [GeV]")
    axes[0].legend()
    axes[1].plot(L_fm, residual * 1000, color="#2874a6")
    axes[1].axhline(0, color="gray", linewidth=0.7)
    axes[1].set(xlabel="L [fm]", ylabel="MLP - Cornell [MeV]")
    axes[2].plot(r, model(r), color="#148f77")
    axes[2].set(xlabel=r"r [GeV$^{-1}$]", ylabel="Learned w(r)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(out / "fit.png", dpi=180)
    plt.close(fig)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON parameters; otherwise uses the example defaults.")
    parser.add_argument("--out", type=Path, help="New output directory (existing directories are preserved).")
    args = parser.parse_args()
    cfg = Config(**json.loads(args.config.read_text(encoding="utf-8"))) if args.config else Config()
    out = args.out or Path(__file__).resolve().parent / "runs" / time.strftime("%Y%m%d_%H%M%S")
    fit(cfg, out)


if __name__ == "__main__":
    main()
