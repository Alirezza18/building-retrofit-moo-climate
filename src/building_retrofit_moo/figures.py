"""Publication-quality figure generation.

Every figure is written to ``results/figures`` in the formats listed in
``Settings.figure_formats`` (PNG + vector PDF by default).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

from . import config, sensitivity
from .io_utils import fs_path, write_csv
from .config import (
    CORR_CMAP_COLORS,
    FIGURE_DIR,
    NICE_NAMES,
    SCENARIO_LABELS,
    TARGET_LABELS,
    TEST_C,
    TRAIN_C,
    Settings,
)
from .models import FULL_NAMES, MODEL_NAMES

_SERIF = ("Times New Roman", "Liberation Serif", "DejaVu Serif")
_SANS = ("DejaVu Sans", "Arial", "Liberation Sans", "sans-serif")


def _save(fig, name: str, settings: Settings) -> Path:
    settings.ensure_dirs()
    last = None
    for fmt in settings.figure_formats:
        path = FIGURE_DIR / f"{name}.{fmt}"
        kwargs = {"dpi": settings.fig_dpi, "bbox_inches": "tight"}
        if fmt == "tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(fs_path(path), **kwargs)
        last = path
    with warnings.catch_warnings():
        # Non-interactive backends (Agg in scripts/CI) warn that show() is a no-op.
        warnings.simplefilter("ignore", UserWarning)
        plt.show()
    plt.close(fig)
    return last  # type: ignore[return-value]


def _set_serif():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": list(_SERIF),
        "mathtext.fontset": "stix", "axes.unicode_minus": False,
    })


def _set_sans():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": list(_SANS),
        "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
    })


# --------------------------------------------------------------------------- #
# Figure 4 - actual vs predicted for all five models
# --------------------------------------------------------------------------- #
def _fmt(v: float) -> str:
    return f"{v:.4f}" if v < 0.01 else f"{v:.3f}"


def _metrics_box(ax, mtr: dict, mte: dict):
    x0, x1, y0, y1 = 0.58, 0.98, 0.02, 0.50
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.02", transform=ax.transAxes,
        fc="white", ec="#333333", lw=1.0, alpha=0.9, zorder=5))
    step = (y1 - y0) / 8.6

    def block(title, m, start):
        lines = [r"$\bf{" + title + "}$", f"$R^2$: {m['R2']:.4f}",
                 f"MAE: {_fmt(m['MAE'])}", f"CV-RMSE: {m['CV_RMSE']:.2f}%"]
        for i, s in enumerate(lines):
            ax.text(x1 - 0.02, y1 - (start + i + 0.6) * step, s, transform=ax.transAxes,
                    ha="right", va="center", fontsize=10.6, zorder=6)

    block("TRAIN", mtr, 0)
    ys = y1 - 4.3 * step
    ax.plot([x0 + 0.02, x1 - 0.02], [ys, ys], transform=ax.transAxes,
            color="#666666", lw=0.8, ls="--", zorder=6)
    block("TEST", mte, 4.6)


def plot_model_comparison(scenario, preds, metrics_df, settings: Settings):
    targets = list(metrics_df["Target"].unique())
    fig, axes = plt.subplots(len(targets), len(MODEL_NAMES), figsize=(21, 13.5))
    for r, t in enumerate(targets):
        allv = np.concatenate([np.concatenate(preds[(m, t)]) for m in MODEL_NAMES])
        pad = 0.06 * (allv.max() - allv.min())
        lo, hi = allv.min() - pad, allv.max() + pad
        for c, name in enumerate(MODEL_NAMES):
            ax = axes[r, c]
            ytr, ptr, yte, pte = preds[(name, t)]
            ax.scatter(ytr, ptr, s=20, c=TRAIN_C, alpha=0.35, linewidths=0, zorder=2)
            ax.scatter(yte, pte, s=22, c=TEST_C, alpha=0.75, linewidths=0, zorder=3)
            ax.plot([lo, hi], [lo, hi], color="#2c3e50", lw=1.8, ls="--", zorder=4)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.grid(True, linestyle=":", alpha=0.4, zorder=1)
            ax.tick_params(axis="both", which="major", labelsize=12.5)
            ax.set_xlabel(r"Actual " + TARGET_LABELS[t], fontsize=13.8, fontweight="bold")
            if c == 0:
                ax.set_ylabel(r"Predicted " + TARGET_LABELS[t], fontsize=13.8, fontweight="bold")
                ax.text(-0.30, 1.03, f"({'abc'[r]})", transform=ax.transAxes,
                        fontsize=17.5, fontweight="bold", va="bottom")
            if r == 0:
                ax.set_title(FULL_NAMES[name], fontsize=16.25, fontweight="bold", pad=10)
            ax.legend(handles=[
                Line2D([], [], marker="o", ls="", color=TRAIN_C, alpha=0.6, markersize=6, label="Train data"),
                Line2D([], [], marker="o", ls="", color=TEST_C, markersize=6, label="Test data"),
                Line2D([], [], color="#2c3e50", ls="--", lw=1.8, label="Ideal fit ($y=x$)")],
                loc="upper left", fontsize=10.6, frameon=True, facecolor="white",
                framealpha=0.85, edgecolor="none", handletextpad=0.2, labelspacing=0.25)
            mt = metrics_df[(metrics_df.Target == t) & (metrics_df.Model == name)]
            _metrics_box(ax, mt[mt.Split == "Train"].iloc[0], mt[mt.Split == "Test"].iloc[0])

    fig.suptitle(f"{scenario.replace('_', '-')} scenario", fontsize=20, fontweight="bold", y=1.01)
    plt.tight_layout(h_pad=2.2, w_pad=1.5)
    return _save(fig, f"Figure4_model_comparison_{scenario}", settings)


# --------------------------------------------------------------------------- #
# Figure 5 - Actual vs Predicted time series
# --------------------------------------------------------------------------- #
def plot_actual_vs_predicted(predictions_test, targets, settings: Settings):
    _set_sans()
    plt.rcParams["font.size"] = 11
    # predictions_test is keyed by (scenario, target); recover scenario order.
    scenarios = [s for s in config.SCENARIOS
                 if any(key[0] == s for key in predictions_test)]
    fig, axes = plt.subplots(len(scenarios), len(targets), figsize=(16, 7.5), sharex=True)
    axes = np.atleast_2d(axes)
    for r, scen in enumerate(scenarios):
        for c, t in enumerate(targets):
            ax = axes[r, c]
            y_true, y_pred = predictions_test[(scen, t)]
            rmse = float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
            ax.plot(y_true, color="#2b5c8f", label="Actual", lw=1.2)
            ax.plot(y_pred, color="#d97724", ls="--", label="Predicted", lw=1.2, alpha=0.85)
            if r == 0:
                ax.set_title(f"Prediction results of {t}", fontweight="bold", fontsize=13, pad=10)
            ax.set_ylabel(TARGET_LABELS[t], fontweight="bold")
            if r == len(scenarios) - 1:
                ax.set_xlabel("Prediction samples", fontweight="bold")
            ax.text(0.95, 0.05, f"RMSE = {rmse:.4f}", transform=ax.transAxes,
                    va="bottom", ha="right",
                    bbox=dict(boxstyle="square,pad=0.3", fc="white", ec="black", lw=0.8))
            ax.legend(loc="upper left", frameon=True, framealpha=0.9, edgecolor="none")
            ax.grid(True, ls=":", alpha=0.6)
            ax.tick_params(direction="in", top=True, right=True)
    plt.tight_layout(rect=[0.02, 0, 1, 1])
    return _save(fig, "Figure_5_Actual_vs_Predicted", settings)


# --------------------------------------------------------------------------- #
# Figure 6 - scatter with metric boxes
# --------------------------------------------------------------------------- #
def plot_performance_metrics(models, X_train, X_test, Y, idx_train, idx_test,
                             targets, settings: Settings, primary_model="CatBoost"):
    _set_sans()
    plt.rcParams["font.size"] = 13
    scenarios = list(Y.keys())
    fig, axes = plt.subplots(nrows=len(scenarios), ncols=len(targets), figsize=(16, 12))
    axes = np.atleast_2d(axes)
    for row, scen in enumerate(scenarios):
        for col, t in enumerate(targets):
            ax = axes[row, col]
            model = models[(scen, t)]
            y_tr_true = Y[scen].loc[idx_train, t].values
            y_te_true = Y[scen].loc[idx_test, t].values
            y_tr_pred = model.predict(X_train)
            y_te_pred = model.predict(X_test)
            m_tr = sensitivity_stats(y_tr_true, y_tr_pred)
            m_te = sensitivity_stats(y_te_true, y_te_pred)

            ax.scatter(y_tr_true, y_tr_pred, color="#d8b2d8", alpha=0.5, s=25,
                       label="Train data", edgecolors="none")
            ax.scatter(y_te_true, y_te_pred, color="#58ca93", alpha=0.7, s=25,
                       label="Test data", edgecolors="none")
            allv = np.concatenate([y_tr_true, y_te_true, y_tr_pred, y_te_pred])
            pad = (allv.max() - allv.min()) * 0.05
            lims = [allv.min() - pad, allv.max() + pad]
            ax.plot(lims, lims, color="black", linestyle=":", linewidth=1.8, label="Ideal fit (y=x)")
            ax.set_xlim(lims); ax.set_ylim(lims)
            if row == 0:
                ax.set_title(f"Prediction results of {t}", fontweight="bold", fontsize=17, pad=10)
            ax.set_xlabel("Actual " + TARGET_LABELS[t], fontweight="bold", fontsize=15)
            ax.set_ylabel("Predicted " + TARGET_LABELS[t], fontweight="bold", fontsize=15)
            txt = ("TRAIN\n"
                   f"$R^2$ : {m_tr['R2']:.3f}\nMAE: {m_tr['MAE']:.3f}\nCV-RMSE: {m_tr['CV_RMSE']:.1f}%\n"
                   "---------------------\n"
                   "TEST\n"
                   f"$R^2$ : {m_te['R2']:.3f}\nMAE: {m_te['MAE']:.3f}\nCV-RMSE: {m_te['CV_RMSE']:.1f}%")
            ax.text(0.95, 0.05, txt, transform=ax.transAxes, fontsize=14,
                    verticalalignment="bottom", horizontalalignment="right", multialignment="center",
                    bbox=dict(boxstyle="square,pad=0.5", facecolor="white", edgecolor="black", linewidth=0.8))
            ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9,
                      edgecolor="none", fontsize=14)
            ax.grid(True, linestyle=":", alpha=0.5, color="gray")
            ax.tick_params(direction="in", top=True, right=True)
    plt.tight_layout(rect=[0.02, 0, 1, 1])
    return _save(fig, "Figure_6_Performance_Metrics_Comparison", settings)


def sensitivity_stats(y_true, y_pred) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"R2": float(r2_score(y_true, y_pred)),
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "CV_RMSE": rmse / float(np.mean(y_true)) * 100.0}


# --------------------------------------------------------------------------- #
# SHAP summary figure
# --------------------------------------------------------------------------- #
def _jitter(vals, row_height=0.9, nbins=100):
    q = np.round(nbins * (vals - vals.min()) / (vals.max() - vals.min() + 1e-12))
    ys = np.zeros(len(vals))
    layer, last = 0, -1
    for i in np.argsort(vals):
        if q[i] != last:
            layer = 0
            last = q[i]
        ys[i] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
        layer += 1
    return ys * (row_height / (np.max(np.abs(ys)) + 1)) * 0.5 if ys.any() else ys


def _beeswarm(ax, sv, Xn, cols):
    imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)
    for rank, f in enumerate(order):
        ax.scatter(sv[:, f], rank + _jitter(sv[:, f]), c=Xn.iloc[:, f], cmap="coolwarm",
                   vmin=0, vmax=1, s=4, alpha=0.9, linewidths=0, rasterized=True)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NICE_NAMES.get(cols[f], cols[f]) for f in order], fontsize=10)
    ax.axvline(0, color="grey", lw=0.6, alpha=0.6, zorder=0)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("SHAP value (Impact on prediction)", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", length=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    return imp, order


def _add_importance_bars(ax, imp, order):
    ax2 = ax.twiny()
    ax2.barh(range(len(order)), imp[order], height=0.7, color="#b9cde5", zorder=1)
    ax2.set_xlim(0, imp.max() * 1.12)
    ax2.set_ylim(ax.get_ylim())
    ax2.xaxis.set_ticks_position("top")
    ax2.xaxis.set_label_position("top")
    ax2.set_xlabel("Mean |SHAP| (Feature Importance)", fontsize=10, fontweight="bold",
                   labelpad=6, color="#4a7bb0")
    ax2.tick_params(axis="x", labelsize=9, length=3, colors="#4a7bb0")
    ax2.tick_params(axis="y", left=False, labelleft=False)
    for s in ["left", "right", "bottom"]:
        ax2.spines[s].set_visible(False)
    ax2.spines["top"].set_color("#4a7bb0")
    ax2.set_zorder(1)
    ax.set_zorder(2)
    ax.patch.set_visible(False)


def plot_shap_summary(models, X_plot, targets, scenarios, settings: Settings):
    _set_serif()
    plt.rcParams["font.serif"] = list(_SERIF)
    X_norm = (X_plot - X_plot.min()) / (X_plot.max() - X_plot.min())
    cols = list(X_plot.columns)
    col_titles = [f"{SCENARIO_LABELS[s].split()[0]} scenario" for s in scenarios]

    fig, axes = plt.subplots(len(targets), len(scenarios), figsize=(11.5, 13.5))
    for r, t in enumerate(targets):
        for c, s in enumerate(scenarios):
            ax = axes[r, c]
            sv = sensitivity.shap_values(models[(s, t)], X_plot)
            imp, order = _beeswarm(ax, sv, X_norm, cols)
            cb = fig.colorbar(ScalarMappable(Normalize(0, 1), "coolwarm"), ax=ax,
                              fraction=0.035, pad=0.02)
            cb.set_label("Feature value", fontsize=10, fontweight="bold")
            cb.ax.tick_params(labelsize=8)
            cb.outline.set_visible(False)
            _add_importance_bars(ax, imp, order)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=13, fontweight="bold", pad=38)
        axes[r, 0].text(-0.42, 1.02, f"{'abc'[r]})", transform=axes[r, 0].transAxes,
                        fontsize=13, fontweight="bold", va="bottom")
    plt.tight_layout(h_pad=3, w_pad=2)
    return _save(fig, "SHAP_summary_figure", settings)


# --------------------------------------------------------------------------- #
# Figure 8 - SHAP dependence
# --------------------------------------------------------------------------- #
def plot_shap_dependence(models, X, settings: Settings):
    _set_sans()
    plt.rcParams["font.size"] = 12
    for scen_key in ["Baseline", "Mid_future"]:
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(19, 5.5), dpi=300)
        for ax, cfg in zip(axes, settings.shap_dependence):
            t_code = cfg["target"]
            model = models[(scen_key, t_code)]
            sv = sensitivity.shap_values(model, X)
            feat_x, feat_color = cfg["feature_x"], cfg["interaction_feature"]
            feat_idx = list(X.columns).index(feat_x)
            y_shap = sv[:, feat_idx]

            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, zorder=1)
            sc = ax.scatter(X[feat_x].values, y_shap, c=X[feat_color].values, cmap="plasma",
                            s=25, alpha=0.85, edgecolors="k", linewidths=0.2, zorder=2)
            ax.set_title(t_code, fontweight="bold", fontsize=15, pad=12)
            ax.set_xlabel(cfg["label_x"], fontweight="bold", fontsize=13)
            ax.set_ylabel(f"SHAP value for {t_code}", fontweight="bold", fontsize=13)
            ax.tick_params(axis="both", which="major", labelsize=11, direction="in",
                           top=True, right=True, length=5, width=0.8)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
            cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label(cfg["label_color"], fontweight="bold", fontsize=12)
            cbar.ax.tick_params(labelsize=10, direction="in")
            cbar.outline.set_linewidth(0.8)
            ax.grid(False)
        fig.suptitle(SCENARIO_LABELS[scen_key], fontsize=16, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        _save(fig, f"Figure_8_SHAP_Dependence_{scen_key}", settings)


# --------------------------------------------------------------------------- #
# Figure 9 - violin distributions of the whole dataset
# --------------------------------------------------------------------------- #
def plot_violin_distributions(X, y_base, y_future, targets, settings: Settings):
    import seaborn as sns
    _set_sans()
    df_base = pd.concat([X, y_base], axis=1); df_base["Scenario_Label"] = "Baseline\nscenario"
    df_fut = pd.concat([X, y_future], axis=1); df_fut["Scenario_Label"] = "Mid-future\nscenario"
    df_all = pd.concat([df_base, df_fut], ignore_index=True)

    fig, axes = plt.subplots(1, len(targets), figsize=(15, 5.5), sharex=True, dpi=300)
    colors = ["#2b93b8", "#e06138"]
    scen_labels = ["Baseline\nscenario", "Mid-future\nscenario"]
    for ax, col in zip(axes, targets):
        sns.violinplot(data=df_all, x="Scenario_Label", y=col, hue="Scenario_Label",
                       legend=False, ax=ax, palette=colors, inner=None, cut=0,
                       linewidth=0.9, saturation=0.95)
        for i, scen_label in enumerate(scen_labels):
            vals = df_all[df_all["Scenario_Label"] == scen_label][col].dropna().values
            q25, med, q75 = np.percentile(vals, [25, 50, 75])
            iqr = q75 - q25
            low_c = vals[vals >= q25 - 1.5 * iqr]; high_c = vals[vals <= q75 + 1.5 * iqr]
            low = np.min(low_c) if len(low_c) else q25
            high = np.max(high_c) if len(high_c) else q75
            ax.plot([i, i], [low, high], color="#2c3e50", linewidth=1.2, zorder=3)
            ax.plot([i - 0.04, i + 0.04], [low, low], color="#2c3e50", linewidth=1.2, zorder=3)
            ax.plot([i - 0.04, i + 0.04], [high, high], color="#2c3e50", linewidth=1.2, zorder=3)
            ax.add_patch(plt.Rectangle((i - 0.065, q25), 0.13, q75 - q25,
                                       facecolor="#34495e", edgecolor="#2c3e50",
                                       linewidth=0.8, alpha=0.9, zorder=4))
            ax.plot([i - 0.065, i + 0.065], [med, med], color="white", linewidth=1.2, zorder=5)
        ax.set_title(f"{col} distribution", fontweight="bold", fontsize=14, pad=12)
        ax.set_ylabel(TARGET_LABELS[col], fontweight="bold", fontsize=12)
        ax.set_xlabel("", labelpad=0)
        ax.tick_params(axis="x", labelsize=11.5)
        ax.tick_params(axis="y", labelsize=11, direction="in", top=True, right=True)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5, color="gray")
        ax.set_axisbelow(True)
    plt.tight_layout()
    return _save(fig, "Figure_9_Violin_Distributions_All_Data", settings)


# --------------------------------------------------------------------------- #
# Pareto fronts (top-N TOPSIS solutions)
# --------------------------------------------------------------------------- #
def plot_pareto_fronts(pareto_by_scenario, best_by_scenario, targets, settings: Settings):
    _set_sans()
    plt.rcParams.update({"axes.edgecolor": "#333333", "axes.linewidth": 0.8,
                         "grid.color": "#e0e0e0", "grid.linestyle": "--", "grid.linewidth": 0.5})
    pairs = [("EUI", "IDD"), ("EUI", "OCI"), ("IDD", "OCI")]
    scenarios = list(pareto_by_scenario.keys())
    fig, axes = plt.subplots(len(scenarios), 3, figsize=(16, 10))
    sc_mappable = None
    for row, scen in enumerate(scenarios):
        p_df = pareto_by_scenario[scen].head(settings.n_display_pareto)
        b_df = best_by_scenario[scen]
        for k, (a, b) in enumerate(pairs):
            ax = axes[row, k]
            sc = ax.scatter(p_df[a], p_df[b], c=p_df["C_star"], cmap="viridis", s=45,
                            edgecolors="black", linewidths=0.4, alpha=0.9, zorder=3)
            ax.scatter(b_df[a], b_df[b], c="#D62728", s=220, marker="*", edgecolors="black",
                       linewidths=0.6, zorder=5,
                       label="Top TOPSIS design" if (row == 0 and k == 0) else "")
            ax.set_title(f"{scen.replace('_', '-')}: {a} vs {b}", fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel(TARGET_LABELS[a], fontsize=10)
            ax.set_ylabel(TARGET_LABELS[b], fontsize=10)
            ax.grid(True)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if row == 0 and k == 0:
                ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, fontsize=9)
            sc_mappable = sc
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.02])
    cbar = fig.colorbar(sc_mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(r"TOPSIS closeness coefficient ($C^*$), top "
                   f"{settings.n_display_pareto} solutions per scenario",
                   fontsize=11, fontweight="bold")
    plt.subplots_adjust(top=0.93, bottom=0.12, hspace=0.35, wspace=0.28)
    return _save(fig, "Pareto_fronts_top50", settings)


# --------------------------------------------------------------------------- #
# Figure 10 - parallel coordinates
# --------------------------------------------------------------------------- #
def build_parallel_coordinates_data(X, features, targets, initial_scheme,
                                    pareto_by_scenario, best_by_scenario, models,
                                    settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assemble the Figure-10 data table and normalization bounds (also saved as CSV)."""
    x_init = pd.DataFrame([initial_scheme])[features]
    initial = {}
    for scen in pareto_by_scenario:
        d = dict(initial_scheme)
        d.update({t: float(np.ravel(models[(scen, t)].predict(x_init))[0]) for t in targets})
        initial[scen] = d

    all_cols = list(features) + list(targets)
    bounds_min = {c: float(X[c].min()) for c in features}
    bounds_max = {c: float(X[c].max()) for c in features}
    for t in targets:
        vals = np.concatenate([
            pareto_by_scenario[s][t].head(settings.n_display_pareto).values
            for s in pareto_by_scenario
        ] + [[initial[s][t] for s in pareto_by_scenario]])
        bounds_min[t], bounds_max[t] = float(vals.min()), float(vals.max())

    def normalize_row(row):
        out = []
        for c in all_cols:
            mn, mx = bounds_min[c], bounds_max[c]
            out.append((row[c] - mn) / (mx - mn) if mx > mn else 0.5)
        return np.array(out)

    rows = []
    for scen, df in pareto_by_scenario.items():
        d = df.head(settings.n_display_pareto)[all_cols + ["C_star"]].astype(float).copy()
        d.insert(0, "Rank", range(1, len(d) + 1))
        d.insert(0, "Type", "Top-50 Pareto")
        d.insert(0, "Scenario", scen)
        rows.append(d)
    for scen, best in best_by_scenario.items():
        d = pd.DataFrame([best[all_cols + ["C_star"]].astype(float)])
        d.insert(0, "Rank", 1)
        d.insert(0, "Type", "Selected TOPSIS design")
        d.insert(0, "Scenario", scen)
        rows.append(d)
    for scen, d in initial.items():
        r = {"Scenario": scen, "Type": "Initial scheme", "Rank": np.nan, **d, "C_star": np.nan}
        rows.append(pd.DataFrame([r]))
    data = pd.concat(rows, ignore_index=True)[["Scenario", "Type", "Rank"] + all_cols + ["C_star"]]

    bounds = pd.DataFrame({"Min": bounds_min, "Max": bounds_max}).loc[all_cols]
    return data, bounds, normalize_row, initial


def plot_parallel_coordinates(X, features, targets, pareto_by_scenario, best_by_scenario,
                              models, settings: Settings, initial_scheme: dict):
    from scipy.interpolate import PchipInterpolator
    _set_sans()
    data, bounds, normalize_row, initial = build_parallel_coordinates_data(
        X, features, targets, initial_scheme, pareto_by_scenario, best_by_scenario, models, settings)
    write_csv(data, FIGURE_DIR.parent / "tables" / "Figure_10_data_inputs_outputs.csv")
    write_csv(bounds, FIGURE_DIR.parent / "tables" / "Figure_10_axis_bounds.csv", index=True)

    all_cols = list(features) + list(targets)
    n_vars = len(all_cols)
    x_idx = np.arange(n_vars)
    x_smooth = np.linspace(0, n_vars - 1, 300)

    def smooth(y):
        return np.clip(PchipInterpolator(x_idx, y)(x_smooth), 0.0, 1.0)

    scens = list(pareto_by_scenario.keys())
    base_s, mid_s = scens[0], scens[1]
    norm_base = np.array([normalize_row(r) for _, r in
                          pareto_by_scenario[base_s].head(settings.n_display_pareto).iterrows()])
    norm_mid = np.array([normalize_row(r) for _, r in
                         pareto_by_scenario[mid_s].head(settings.n_display_pareto).iterrows()])
    nb_base, nb_mid = normalize_row(best_by_scenario[base_s]), normalize_row(best_by_scenario[mid_s])
    ni_base, ni_mid = normalize_row(initial[base_s]), normalize_row(initial[mid_s])

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.fill_between(x_smooth, smooth(norm_base.min(0)), smooth(norm_base.max(0)),
                    color="#b0e0e6", alpha=0.45, label=f"Top 50 Pareto solutions ({base_s})")
    ax.fill_between(x_smooth, smooth(norm_mid.min(0)), smooth(norm_mid.max(0)),
                    color="#ffc0cb", alpha=0.45, label=f"Top 50 Pareto solutions ({mid_s})")
    ax.plot(x_smooth, smooth(nb_base), color="#008080", lw=2.5, label=f"Ideal solution ({base_s})")
    ax.plot(x_smooth, smooth(nb_mid), color="#c71585", lw=2.5, label=f"Ideal solution ({mid_s})")
    ax.plot(x_smooth, smooth(ni_base), color="black", ls="--", lw=2.0, label=f"Initial scheme ({base_s})")
    ax.plot(x_smooth, smooth(ni_mid), color="dimgray", ls=":", lw=2.2, label=f"Initial scheme ({mid_s})")
    ax.scatter(x_idx, nb_base, color="#008080", s=40, zorder=5)
    ax.scatter(x_idx, nb_mid, color="#c71585", s=40, zorder=5)
    ax.scatter(x_idx, ni_base, color="black", s=30, zorder=5)
    ax.scatter(x_idx, ni_mid, color="dimgray", s=30, zorder=5)
    for i in range(n_vars):
        ax.axvline(i, color="lightgray", lw=0.8, zorder=1)
    n_features = len(features)
    ax.axvline(n_features - 0.5, color="gray", ls="--", lw=1.5, zorder=2)
    labels = [NICE_NAMES.get(c, c) for c in all_cols[:n_features]] + [TARGET_LABELS[t] for t in targets]
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontweight="bold", fontsize=10)
    for i, col in enumerate(all_cols):
        mn, mx = bounds.loc[col, "Min"], bounds.loc[col, "Max"]
        ax.text(i, 1.03, f"{mx:.2f}" if mx < 10 else f"{mx:.1f}", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold")
        ax.text(i, -0.05, f"{mn:.2f}" if mn < 10 else f"{mn:.1f}", ha="center", va="top",
                fontsize=9.5, fontweight="bold")
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontweight="bold")
    ax.set_ylabel("Normalization", fontweight="bold", fontsize=12)
    ax.set_ylim(-0.08, 1.08)
    ax.text((n_features - 1) / 2.0, 1.12, "Retrofit variables", ha="center", va="center",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="none"))
    ax.text(n_features + (len(targets) - 1) / 2.0, 1.12, "Performance results", ha="center",
            va="center", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="none"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=True,
              facecolor="white", edgecolor="gray", fontsize=10.5)
    ax.grid(False)
    plt.tight_layout()
    return _save(fig, "Figure_10_Parallel_Coordinates_top50", settings)


# --------------------------------------------------------------------------- #
# Figure 11 - correlation heatmaps (Spearman / Pearson)
# --------------------------------------------------------------------------- #
def plot_correlation_heatmap(pareto_by_scenario, features, settings: Settings,
                             method: str = "spearman"):
    _set_serif()
    cmap = LinearSegmentedColormap.from_list("corr", CORR_CMAP_COLORS)
    norm = Normalize(-1, 1)
    cols = list(features)
    n = len(cols)
    scens = list(pareto_by_scenario.keys())
    data = {s: pareto_by_scenario[s].head(settings.n_display_pareto) for s in scens}

    stars = lambda p: "**" if p <= 0.01 else ("*" if p <= 0.05 else "")
    fmt_r = lambda r: f"{r:.2f}" if abs(r) >= 0.01 else f"{r:.3f}"
    radius = lambda r: 0.08 + 0.34 * abs(r)

    fig, axes = plt.subplots(1, 2, figsize=(16, 9.2))
    fig.subplots_adjust(left=0.07, right=0.985, top=0.91, bottom=0.30, wspace=0.12)
    matrices = {}
    for ax, scen in zip(axes, scens):
        rho, pval = sensitivity.correlation_matrix(data[scen], method)
        matrices[scen] = (rho, pval)
        R, P = rho.values, pval.values
        for i in range(n):
            for j in range(n):
                if np.isnan(R[i, j]):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=13, color="black")
                    continue
                if i == j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor="#f0f0f0", lw=0, zorder=1))
                    ax.text(j, i, "1", ha="center", va="center", fontsize=10, fontweight="bold")
                elif i > j:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, facecolor=cmap(norm(R[i, j])), lw=0, zorder=1))
                    tc = "white" if abs(R[i, j]) > 0.55 else "black"
                    ax.text(j, i - 0.07, fmt_r(R[i, j]), ha="center", va="center", fontsize=10.5,
                            fontweight="bold", color=tc, zorder=3)
                    s = stars(P[i, j])
                    if s:
                        ax.text(j, i + 0.24, s, ha="center", va="center", fontsize=10,
                                fontweight="bold", color=tc, zorder=3)
                else:
                    ax.add_patch(plt.Circle((j, i), radius(R[i, j]), facecolor=cmap(norm(R[i, j])),
                                            edgecolor="black", lw=0.8, zorder=3))
        for k in np.arange(-.5, n, 1):
            ax.axhline(k, color="#cccccc", lw=0.8, zorder=2)
            ax.axvline(k, color="#cccccc", lw=0.8, zorder=2)
        ax.set_xlim(-.5, n - .5); ax.set_ylim(n - .5, -.5); ax.set_aspect("equal")
        labels = [NICE_NAMES.get(c, c) for c in cols]
        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=40, ha="right", rotation_mode="anchor",
                           fontsize=10.5, fontweight="bold")
        ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
        ax.tick_params(length=0, pad=4)
        for sp in ax.spines.values():
            sp.set_linewidth(1.2); sp.set_color("black")
        ax.set_title(f"{SCENARIO_LABELS[scen]} (top {settings.n_display_pareto} Pareto solutions)",
                     fontsize=13.5, fontweight="bold", pad=8)

    LEG_Y = 0.08
    cax = fig.add_axes([0.07, LEG_Y + 0.02, 0.22, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal",
                      ticks=[-1, -0.5, 0, 0.5, 1])
    cb.ax.set_xticklabels(["−1", "−0.5", "0", "0.5", "1"], fontsize=10)
    cb.outline.set_edgecolor("black")
    cb.set_label(f"{'Spearman rho' if method == 'spearman' else 'Pearson r'}", fontsize=11, fontweight="bold")

    fig.text(0.43, LEG_Y + 0.075, "Circle size (upper triangle)", fontsize=10, fontweight="bold", ha="center")
    lax = fig.add_axes([0.33, LEG_Y - 0.02, 0.20, 0.08]); lax.axis("off")
    lax.set_xlim(0, 3); lax.set_ylim(0, 1); lax.set_aspect("equal")
    for k, v in enumerate([0.1, 0.5, 0.9]):
        lax.add_patch(plt.Circle((0.5 + k * 1.0, 0.5), (0.05 + 0.18 * abs(v)) * 1.5,
                                 facecolor="#bdbdbd", edgecolor="black", lw=0.8))
        lax.text(0.5 + k * 1.0, -0.25, f"|r| = {v}", ha="center", va="center", fontsize=9.5)
    fig.text(0.64, LEG_Y + 0.055, "* p ≤ 0.05     ** p ≤ 0.01", fontsize=10, style="italic")
    fig.text(0.64, LEG_Y + 0.015, "-  correlation undefined (constant variable)", fontsize=10, style="italic")

    name = f"Figure11_{'Spearman' if method == 'spearman' else 'Pearson'}_heatmap_top50"
    path = _save(fig, name, settings)
    return path, matrices
