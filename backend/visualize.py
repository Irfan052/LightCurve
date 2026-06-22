import matplotlib
# Use non-interactive Agg backend to avoid GUI threads issues in web servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Tuple
from backend.utils import fig_to_base64, logger
from backend.config import DATA_RESULTS_DIR

# Styling constants
PRIMARY_COLOR = "#3b82f6"     # Modern blue
ACCENT_COLOR = "#ef4444"      # Coral Red
BG_COLOR = "#0f172a"          # Slate dark
GRID_COLOR = "#f1f5f9"
TEXT_COLOR = "#1e293b"
PLOT_BG = "#f8fafc"

# Set plot defaults for clean, modern look
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["text.color"] = TEXT_COLOR
plt.rcParams["axes.labelcolor"] = TEXT_COLOR
plt.rcParams["xtick.color"] = TEXT_COLOR
plt.rcParams["ytick.color"] = TEXT_COLOR
plt.rcParams["grid.color"] = "#e2e8f0"

def plot_raw_lightcurve(
    time: np.ndarray, 
    flux: np.ndarray, 
    target_name: str
) -> str:
    """Generates a plot of the raw, noisy light curve."""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    ax.set_facecolor(PLOT_BG)
    
    ax.scatter(time, flux, s=1, color="#64748b", alpha=0.5, label="Raw Observations")
    
    ax.set_title(f"Raw Light Curve: {target_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Time (BJD - 2457000, days)", fontsize=10)
    ax.set_ylabel("Relative Flux (Normalized)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right", framealpha=0.9)
    
    # Save a physical file in results for archiving
    fig.savefig(DATA_RESULTS_DIR / f"{target_name.replace(' ', '_')}_raw.png", dpi=150, bbox_inches="tight")
    
    return fig_to_base64(fig)

def plot_cleaned_detrended(
    time: np.ndarray,
    raw_flux: np.ndarray,
    trend_flux: np.ndarray,
    flat_flux: np.ndarray,
    target_name: str
) -> str:
    """
    Generates a 2-panel plot showing:
    1. Raw light curve with the low-frequency trend line superimposed.
    2. Detrended/flattened light curve.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor="white")
    
    # Upper Panel - Trend
    ax1.set_facecolor(PLOT_BG)
    ax1.scatter(time, raw_flux, s=1, color="#94a3b8", alpha=0.4, label="Cleaned Flux")
    ax1.plot(time, trend_flux, color=ACCENT_COLOR, linewidth=1.5, label="Stellar Trend (Savitzky-Golay)")
    ax1.set_title(f"Stellar Trend & Flattening: {target_name}", fontsize=12, fontweight="bold", pad=12)
    ax1.set_ylabel("Raw Relative Flux", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend(loc="upper right")
    
    # Lower Panel - Flattened
    ax2.set_facecolor(PLOT_BG)
    ax2.scatter(time, flat_flux, s=1, color=PRIMARY_COLOR, alpha=0.5, label="Flattened Flux")
    ax2.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    ax2.set_ylabel("Flattened Flux", fontsize=10)
    ax2.set_xlabel("Time (BJD - 2457000, days)", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend(loc="upper right")
    
    plt.tight_layout()
    
    fig.savefig(DATA_RESULTS_DIR / f"{target_name.replace(' ', '_')}_detrended.png", dpi=150, bbox_inches="tight")
    return fig_to_base64(fig)

def plot_folded_transit(
    phase: np.ndarray,
    flux: np.ndarray,
    bin_centers: np.ndarray,
    bin_flux: np.ndarray,
    target_name: str,
    period: float
) -> str:
    """
    Generates a phase-folded light curve plot focusing on phase [-0.25, 0.25].
    Shows individual data points and a binned curve to highlight transit shape.
    """
    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor="white")
    ax.set_facecolor(PLOT_BG)
    
    # Plot individual folded points
    ax.scatter(phase, flux, s=2, color="#94a3b8", alpha=0.3, label="Individual Phases")
    
    # Plot binned points
    ax.plot(bin_centers, bin_flux, color=PRIMARY_COLOR, linewidth=2.0, label="Binned Flux (100 bins)")
    ax.scatter(bin_centers, bin_flux, color=ACCENT_COLOR, s=10, zorder=3)
    
    ax.axhline(1.0, color="#64748b", linestyle="--", linewidth=1.0)
    
    # Zoom in on the central part where transits typically occur
    ax.set_xlim(-0.25, 0.25)
    
    ax.set_title(f"Phase Folded Transit (P = {period:.4f} days): {target_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Phase", fontsize=10)
    ax.set_ylabel("Relative Flux", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="lower right")
    
    fig.savefig(DATA_RESULTS_DIR / f"{target_name.replace(' ', '_')}_folded.png", dpi=150, bbox_inches="tight")
    return fig_to_base64(fig)

def plot_bls_periodogram(
    periods: np.ndarray,
    powers: np.ndarray,
    best_period: float,
    target_name: str
) -> str:
    """Generates the Box Least Squares (BLS) periodogram power spectrum plot."""
    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="white")
    ax.set_facecolor(PLOT_BG)
    
    ax.plot(periods, powers, color="#0f172a", linewidth=1.0)
    ax.axvline(best_period, color=ACCENT_COLOR, linestyle="--", linewidth=1.5, 
               label=f"Peak Period = {best_period:.4f} d")
    
    ax.set_title(f"BLS Periodogram Power Spectrum: {target_name}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Period (days)", fontsize=10)
    ax.set_ylabel("BLS Power", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")
    
    fig.savefig(DATA_RESULTS_DIR / f"{target_name.replace(' ', '_')}_bls.png", dpi=150, bbox_inches="tight")
    return fig_to_base64(fig)
