"""Instances de demande pour l'engagement des unités (200 générateurs, 48 h).

Reproduit le protocole de Sugishita, Grothey & McKinnon (2024),
sucpy/problems/uc.py (load_demand_data et _sample_demand) :

1. demande horaire = moyenne des deux demi-heures NESO (colonne ND) ;
2. jours invalides = jours sans exactement 48 demi-heures (changements d'heure) ;
3. mise à l'échelle GLOBALE : demande / pic_typique * 0.5 * capacité_totale ;
4. fenêtre de 48 heures (2 jours) à partir de minuit d'une date ;
5. bruit multiplicatif : processus gaussien (écart-type 0.05, longueur 0.5 jour) ;
6. ensembles par année : train 2017, val 2018, test 2019.

Lancer `python uc/instances.py` pour construire data/instances/{train,val,test}.npz,
afficher les vérifications et tracer results/instances_examples.png.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uc.data import load_generators

DEMAND_CSV = ROOT / "data" / "demand" / "out" / "demand.csv"
INSTANCE_DIR = ROOT / "data" / "instances"
INSTANCE_DIR_KDE = ROOT / "data" / "instances_kde"   # variante de sensibilité (pic typique = KDE)

N_HOURS = 48            # horizon : 48 heures = 2 jours
NOISE_STD = 0.05        # écart-type du bruit relatif à chaque heure
NOISE_LENGTH = 0.5      # longueur de corrélation du bruit, en jours
PEAK_RATIO = 0.5        # un jour « typique » a un pic à 50 % de la capacité
KDE_TOLERANCE = 0.02    # on garde le pic KDE si l'écart avec celui des auteurs est < 2 %

# Ensembles : année, nombre d'instances, graine.
SPLITS = {
    "train": {"year": 2017, "n": 1000, "seed": 0},
    "val":   {"year": 2018, "n": 30,   "seed": 1},
    "test":  {"year": 2019, "n": 100,  "seed": 2},
}


# --------------------------------------------------------------------------
# 1-2. Demande horaire et jours invalides
# --------------------------------------------------------------------------

def load_hourly_demand(path=DEMAND_CSV):
    """Demande horaire (MW) et liste des jours invalides.

    Retourne
      hourly  : DataFrame avec colonnes date, hour, datetime, demand
                (une ligne par (date, heure), moyenne de ND sur les 2 demi-heures)
      invalid : ensemble des dates qui n'ont pas exactement 48 demi-heures
    """
    df = pd.read_csv(path, usecols=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "ND"])
    df["date"] = pd.to_datetime(df["SETTLEMENT_DATE"])
    df["hour"] = (df["SETTLEMENT_PERIOD"] - 1) // 2

    counts = df.groupby("date").size()
    invalid = set(counts.index[counts != 48])

    hourly = df.groupby(["date", "hour"], as_index=False)["ND"].mean()
    hourly = hourly.rename(columns={"ND": "demand"})
    hourly["datetime"] = hourly["date"] + pd.to_timedelta(hourly["hour"], unit="h")
    return hourly, invalid


# --------------------------------------------------------------------------
# 3. Pic typique (mise à l'échelle globale)
# --------------------------------------------------------------------------

def daily_peaks(hourly):
    """Pic journalier = max de la demande horaire de chaque date (toutes les données)."""
    return hourly.groupby("date")["demand"].max()


def typical_peak_authors(peaks):
    """Version des auteurs : mode exact, moyenne s'il y a plusieurs modes."""
    return float(np.mean(peaks.mode()))


def typical_peak_kde(peaks):
    """Mode d'une estimation lissée (KDE gaussienne, largeur de bande de Scott)."""
    grid = np.linspace(peaks.min(), peaks.max(), 5000)
    density = gaussian_kde(peaks.to_numpy())(grid)
    return float(grid[np.argmax(density)])


def choose_typical_peak(peaks, peak_method="kde", verbose=True):
    """Calcule les deux pics typiques et retourne celui choisi.

    peak_method = "kde"     : KDE si l'écart relatif avec les auteurs est < 2 %, sinon auteurs
    peak_method = "authors" : toujours la version des auteurs
    peak_method = "kde_forced" : toujours la KDE (variante pour le test de sensibilité)
    """
    p_authors = typical_peak_authors(peaks)
    p_kde = typical_peak_kde(peaks)
    gap = abs(p_kde - p_authors) / p_authors
    if peak_method == "authors":
        chosen, used = p_authors, "authors"
    elif peak_method == "kde":
        chosen, used = (p_kde, "kde") if gap < KDE_TOLERANCE else (p_authors, "authors")
    elif peak_method == "kde_forced":
        chosen, used = p_kde, "kde"
    else:
        raise ValueError(f"peak_method inconnu : {peak_method}")
    if verbose:
        n_modes = len(peaks.mode())
        print(f"pic typique (auteurs, mode exact, {n_modes} modes) : {p_authors:10.1f} MW")
        print(f"pic typique (mode de la KDE)                : {p_kde:10.1f} MW")
        print(f"écart relatif                               : {gap:.2%}")
        print(f"peak_method={peak_method!r} -> utilisé : {used} ({chosen:.1f} MW)")
    return chosen, used, p_authors, p_kde


# --------------------------------------------------------------------------
# 4. Fenêtres de 48 heures
# --------------------------------------------------------------------------

def valid_start_dates(hourly, invalid, year):
    """Dates d de `year` utilisables comme début de fenêtre de 48 h.

    Exclut : les jours invalides, le jour qui précède un jour invalide, le 31 décembre.
    """
    dates = pd.DatetimeIndex(sorted(hourly["date"].unique()))
    dates = dates[dates.year == year]
    next_day = dates + pd.Timedelta(days=1)
    keep = [
        (d not in invalid) and (n not in invalid) and not (d.month == 12 and d.day == 31)
        for d, n in zip(dates, next_day)
    ]
    return dates[np.array(keep)]


def window(series, date):
    """48 valeurs horaires de date 00:00 à date+1 23:00 (series indexée par datetime)."""
    start = pd.Timestamp(date)
    idx = pd.date_range(start, periods=N_HOURS, freq="h")
    values = series.reindex(idx).to_numpy()
    assert len(values) == N_HOURS and not np.isnan(values).any(), f"fenêtre incomplète : {date}"
    return values


# --------------------------------------------------------------------------
# 5. Bruit : processus gaussien multiplicatif
# --------------------------------------------------------------------------

def noise_factor():
    """L tel que L @ z (z ~ N(0, I)) est un processus gaussien d'écart-type 0.05.

    x en jours sur 2 jours, noyau exponentiel carré de longueur 0.5 jour.
    """
    x = np.linspace(0, 2, N_HOURS)
    K = np.exp(-((x[:, None] - x[None, :]) ** 2) / (2 * NOISE_LENGTH**2))
    K += 1e-10 * np.eye(N_HOURS)
    return np.linalg.cholesky(K) * NOISE_STD


# --------------------------------------------------------------------------
# 6. Construction et chargement des ensembles
# --------------------------------------------------------------------------

def build_instances(gen=None, peak_method="kde", out_dir=INSTANCE_DIR, verbose=True):
    """Construit et sauvegarde train/val/test. Retourne un dict split -> (demand, base_dates)."""
    if gen is None:
        gen = load_generators()
    total_capacity = float(np.sum(gen["max_out"]))

    hourly, invalid = load_hourly_demand()
    peaks = daily_peaks(hourly)
    peak, used, _, _ = choose_typical_peak(peaks, peak_method, verbose)

    # Demande normalisée, indexée par datetime (échelle globale, la même pour tous les jours).
    # Seuls les jours valides sont gardés : un jour à 50 demi-heures a une « heure 24 »
    # qui tomberait sur minuit du lendemain (doublon). Les fenêtres n'utilisent que des jours valides.
    valid_rows = ~hourly["date"].isin(invalid)
    scaled = hourly[valid_rows].set_index("datetime")["demand"] / peak * PEAK_RATIO * total_capacity
    L = noise_factor()

    out_dir.mkdir(parents=True, exist_ok=True)
    result = {}
    for split, cfg in SPLITS.items():
        rng = np.random.default_rng(cfg["seed"])
        starts = valid_start_dates(hourly, invalid, cfg["year"])
        # Tirage uniforme AVEC remise des dates, puis du bruit.
        dates = starts[rng.integers(len(starts), size=cfg["n"])]
        base = np.stack([window(scaled, d) for d in dates])            # (N, 48)
        noise = rng.standard_normal((cfg["n"], N_HOURS)) @ L.T          # ligne i = L @ z_i
        demand = base * (1.0 + noise)
        base_dates = np.array([d.strftime("%Y-%m-%d") for d in dates])

        np.savez(
            out_dir / f"{split}.npz",
            demand=demand,
            base_date=base_dates,
            seed=cfg["seed"],
            peak_method=used,
            typical_peak=peak,
            total_capacity=total_capacity,
        )
        result[split] = (demand, base_dates)
        if verbose:
            print(f"{split:5s} : {len(starts)} dates valides en {cfg['year']}, "
                  f"{cfg['n']} instances, graine {cfg['seed']} -> {out_dir / f'{split}.npz'}")
    return result


def load_instances(split, in_dir=INSTANCE_DIR):
    """Charge un ensemble : retourne (demand (N, 48), base_dates (N,) chaînes ISO)."""
    data = np.load(in_dir / f"{split}.npz")
    return data["demand"], data["base_date"]


# --------------------------------------------------------------------------
# 7. Vérifications et graphique
# --------------------------------------------------------------------------

def check_instances(result, total_capacity):
    """Affiche formes, signes, pic moyen / capacité (global, hiver, été), min/max."""
    print("\nensemble  forme        min(MW)   max(MW)  pic/cap moyen  hiver(DJF)  été(JJA)  négatives")
    for split, (demand, dates) in result.items():
        months = pd.DatetimeIndex(dates).month
        ratio = demand.max(axis=1) / total_capacity
        winter = ratio[np.isin(months, [12, 1, 2])]
        summer = ratio[np.isin(months, [6, 7, 8])]
        print(f"{split:8s}  {str(demand.shape):11s}  {demand.min():8.0f}  {demand.max():8.0f}"
              f"  {ratio.mean():13.3f}  {winter.mean():10.3f}  {summer.mean():8.3f}"
              f"  {(demand < 0).sum():9d}")
        assert demand.shape == (SPLITS[split]["n"], N_HOURS)
        assert (demand >= 0).all()


def plot_examples(demand, dates, total_capacity, path, n=4):
    """Tracé de n instances de test sur 48 h (une par panneau)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hours = np.arange(N_HOURS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), sharey=True, facecolor="#fcfcfb")
    for i, ax in enumerate(axes):
        ax.set_facecolor("#fcfcfb")
        ax.grid(True, color="#e6e5e0", lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.axhline(PEAK_RATIO * total_capacity, color="#8a8984", lw=1, ls="--")
        ax.axvline(24, color="#d6d5cf", lw=0.8)
        ax.plot(hours, demand[i], color="#2a78d6", lw=2)
        ax.set_title(f"test {i} - début {dates[i]}", loc="left", fontsize=10)
        ax.set_xticks(range(0, 49, 6))
        ax.set_xlim(0, 47)
        ax.set_xlabel("heure (0 = minuit du jour 1)")
    axes[0].set_ylabel("demande (MW)")
    axes[-1].text(1, PEAK_RATIO * total_capacity, "50 % capacité", transform=axes[-1].get_yaxis_transform(),
                  ha="right", va="bottom", fontsize=8, color="#52514e")
    fig.suptitle("Instances de test 2019 : demande horaire sur 48 h (bruit inclus)", x=0.01, ha="left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    print(f"\ngraphique : {path.relative_to(ROOT)}")


if __name__ == "__main__":
    # python uc/instances.py         -> data/instances/      (règle : auteurs si écart KDE >= 2 %)
    # python uc/instances.py --kde   -> data/instances_kde/  (variante : pic typique = KDE)
    use_kde = "--kde" in sys.argv
    out_dir = INSTANCE_DIR_KDE if use_kde else INSTANCE_DIR
    gen = load_generators()
    total_capacity = float(np.sum(gen["max_out"]))
    print(f"{gen['n_g']} générateurs, capacité totale {total_capacity:.0f} MW\n")

    result = build_instances(gen, peak_method="kde_forced" if use_kde else "kde", out_dir=out_dir)
    check_instances(result, total_capacity)
    if use_kde:
        sys.exit(0)

    # Relecture depuis le disque pour vérifier load_instances.
    demand, dates = load_instances("test")
    assert np.allclose(demand, result["test"][0]) and (dates == result["test"][1]).all()
    print("load_instances('test') : OK", demand.shape)

    plot_examples(demand, dates, total_capacity, ROOT / "results" / "instances_examples.png")
