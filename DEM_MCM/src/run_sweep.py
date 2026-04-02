"""
===================================================================================
SWEEP MARKOVIEN — Lance les calculs pour un type de partitionnement donné
===================================================================================

Usage:
    python run_sweep.py --method voronoi
    python run_sweep.py --method cartesian
    python run_sweep.py --method all
    python run_sweep.py --method voronoi --list   # liste les configs sans lancer

Depuis Python:
    from run_sweep import run_markov_sweep
    run_markov_sweep("cylindrical")
===================================================================================
"""

import os
import json
import argparse
import numpy as np
import polars as pl
import torch
from tqdm import tqdm
from dataclasses import dataclass, field, asdict
from huggingface_hub import HfFileSystem

# from partitioners import create_partitioner, REGISTRY
# from bucket_io import save_experiment_to_bucket, BUCKET_BASE
from partitioners import create_partitioner, REGISTRY
from bucket_io import save_experiment_to_bucket, BUCKET_BASE



# =============================================================================
# CONFIGURATION GÉNÉRALE
# =============================================================================

# BASE_OUTPUT_DIR = "NewResultsMCM"
BASE_OUTPUT_DIR = "ResultsDtMCM"
HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
SAMPLE_RATE = 50  # pour le fit des partitionneurs


# =============================================================================
# DATACLASS EXPÉRIENCE
# =============================================================================
torch.set_num_threads(4)

@dataclass # crée et ajoute automatiquement le constructeur de classe
class ExperimentConfig:
    """Configuration d'une expérience."""

    method: str = "cartesian"
    method_kwargs: dict = field(default_factory=dict) # type par defaut de la (dict vide) lors de l'instanciation de la classe ExperimentConfig sans passage explicite de method_kwargs
    nlt: int = 100
    dt:int=None
    step_size: int = 1 # pas de temps d'apprentissage telque le temps d'apprentissage soit T=nlt*step_size
    start_index: int = 250 # début de l'apprentissage

    def __post_init__(self):
        if self.method_kwargs is None:
            self.method_kwargs = {}
        if self.dt is None:
            # Par défaut: 1/10 du step_size, minimum 1
            # self.dt = min(1, self.step_size // 10)    
            self.dt=.1 
            
    def output_folder(self, base_dir=BASE_OUTPUT_DIR,sample_coords=None):
        part = create_partitioner(self.method, **self.method_kwargs)
        if sample_coords is not None:
            part.fit(sample_coords)
        return os.path.join(
            base_dir,
            f"{part.label}_NLT{self.nlt}_step{self.step_size}_start{self.start_index}_dt{self.dt}",
        )
   


# =============================================================================
# CONFIGURATIONS PAR MÉTHODE
# =============================================================================


def get_configs(method):
    """
    Retourne la liste de configs pour une méthode donnée.

    Axes de sweep:
      1. Paramètres de discrétisation (propres à chaque méthode)
      2. Nombre de pas de temps (NLT)
      3. Pas de sous-échantillonnage temporel (step_size)
      4. Index de départ (start_index)
      5. Pas de glissement (dt)
    """

    configs = []

    # ══════════════════════════════════════════════════════════════════════
    # Sweep de discrétisation spatiale
    # ══════════════════════════════════════════════════════════════════════

    if method == "cartesian":
        for n in [2, 3, 5, 7, 10, 12, 15, 18, 20]:
            configs.append(
                ExperimentConfig(
                    method="cartesian",
                    method_kwargs={"nx": n, "ny": n, "nz": n},
                )
            )

    elif method == "cylindrical":
        # nr variable (axisymétrique pur)
        for nr in [3, 5, 8, 10, 15, 20]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": nr, "ntheta": 1, "nz": 5,
                        "radial_mode": "equal_area",
                    },
                )
            )
        # ntheta variable
        for nth in [1, 4, 8, 12, 16]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 5, "ntheta": nth, "nz": 5,
                        "radial_mode": "equal_area",
                    },
                )
            )
        # nz variable
        for nz in [3, 5, 8, 10, 15]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 5, "ntheta": 8, "nz": nz,
                        "radial_mode": "equal_area",
                    },
                )
            )
        # equal_dr vs equal_area
        for mode in ["equal_dr", "equal_area"]:
            configs.append(
                ExperimentConfig(
                    method="cylindrical",
                    method_kwargs={
                        "nr": 10, "ntheta": 8, "nz": 10,
                        "radial_mode": mode,
                    },
                )
            )

    elif method == "voronoi":
        for nc in [8, 27, 64, 125, 216, 343, 512, 1000, 2000, 4000]:
            configs.append(
                ExperimentConfig(
                    method="voronoi",
                    method_kwargs={"n_cells": nc},
                )
            )

    elif method == "quantile":
        for n in [2, 3, 5, 7, 10, 12, 15, 18, 20]:
            configs.append(
                ExperimentConfig(
                    method="quantile",
                    method_kwargs={"nx": n, "ny": n, "nz": n},
                )
            )

    elif method == "octree":
        # max_particles variable
        for mp in [20, 50, 100, 200, 500, 1000]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": mp, "max_depth": 5},
                )
            )
        # max_depth variable
        for md in [3, 4, 5, 6, 7]:
            configs.append(
                ExperimentConfig(
                    method="octree",
                    method_kwargs={"max_particles": 100, "max_depth": md},
                )
            )

    elif method == "physics":
        for nc in [27, 64, 125, 216, 512]:
            configs.append(
                ExperimentConfig(
                    method="physics",
                    method_kwargs={"n_cells": nc},
                )
            )

    elif method == "adaptive":
        # ── Sweep z_split (quantile) ─────────────────────────────────
        for z_q in [0.5, 0.6, 0.7, 0.8, 0.9]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": z_q,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 5, "ntheta": 8, "nz": 1,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Sweep finesse zone basse (nr) ────────────────────────────
        for nr in [3, 5, 8, 10, 15]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": nr, "ntheta": 8, "nz": 8,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Sweep finesse zone basse (nz) ────────────────────────────
        for nz in [4, 6, 8, 10, 12, 15]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 5, "ntheta": 8, "nz": nz,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Sweep ntheta zone basse ──────────────────────────────────
        for nth in [1, 4, 8, 12, 16]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 5, "ntheta": nth, "nz": 8,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Zone haute avec quelques cellules ────────────────────────
        for n_top in [1, 2, 4, 8]:
            top_method = "single" if n_top == 1 else "cylindrical"
            top_kwargs = {} if n_top == 1 else {
                "nr": 1, "ntheta": n_top, "nz": 1,
                "radial_mode": "equal_area",
            }
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": n_top,
                        "top_method": top_method,
                        "top_kwargs": top_kwargs,
                        "bottom_method": "cylindrical",
                        "bottom_kwargs": {
                            "nr": 5, "ntheta": 8, "nz": 8,
                            "radial_mode": "equal_area",
                        },
                    },
                )
            )

        # ── Voronoï en bas au lieu de cylindrique ────────────────────
        for nc in [64, 125, 250, 500]:
            configs.append(
                ExperimentConfig(
                    method="adaptive",
                    method_kwargs={
                        "z_split": 0.75,
                        "z_split_mode": "quantile",
                        "n_cells_top": 1,
                        "top_method": "single",
                        "top_kwargs": {},
                        "bottom_method": "voronoi",
                        "bottom_kwargs": {"n_cells": nc},
                    },
                )
            )

    elif method == "multizone":
        # ── 2 zones: fin en bas, grossier en haut ────────────────────
        configs.append(
            ExperimentConfig(
                method="multizone",
                method_kwargs={
                    "z_mode": "quantile",
                    "zones": [
                        {
                            "z_min": 0.0, "z_max": 0.8,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 5, "ntheta": 8, "nz": 10,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.8, "z_max": 1.0,
                            "method": "single",
                            "kwargs": {},
                        },
                    ],
                },
            )
        )

        # ── 3 zones: gradient de finesse ─────────────────────────────
        for split1, split2 in [(0.5, 0.8), (0.6, 0.85), (0.7, 0.9)]:
            configs.append(
                ExperimentConfig(
                    method="multizone",
                    method_kwargs={
                        "z_mode": "quantile",
                        "zones": [
                            {
                                "z_min": 0.0, "z_max": split1,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": 6, "ntheta": 12, "nz": 8,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": split1, "z_max": split2,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": 3, "ntheta": 6, "nz": 4,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": split2, "z_max": 1.0,
                                "method": "single",
                                "kwargs": {},
                            },
                        ],
                    },
                )
            )

        # ── 3 zones avec Voronoï en bas ──────────────────────────────
        for nc_bottom in [125, 250, 500]:
            configs.append(
                ExperimentConfig(
                    method="multizone",
                    method_kwargs={
                        "z_mode": "quantile",
                        "zones": [
                            {
                                "z_min": 0.0, "z_max": 0.6,
                                "method": "voronoi",
                                "kwargs": {"n_cells": nc_bottom},
                            },
                            {
                                "z_min": 0.6, "z_max": 0.85,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": 3, "ntheta": 6, "nz": 4,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": 0.85, "z_max": 1.0,
                                "method": "single",
                                "kwargs": {},
                            },
                        ],
                    },
                )
            )

        # ── 4 zones (très gradué) ────────────────────────────────────
        configs.append(
            ExperimentConfig(
                method="multizone",
                method_kwargs={
                    "z_mode": "quantile",
                    "zones": [
                        {
                            "z_min": 0.0, "z_max": 0.4,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 8, "ntheta": 16, "nz": 10,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.4, "z_max": 0.7,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 5, "ntheta": 10, "nz": 6,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.7, "z_max": 0.9,
                            "method": "cylindrical",
                            "kwargs": {
                                "nr": 3, "ntheta": 6, "nz": 3,
                                "radial_mode": "equal_area",
                            },
                        },
                        {
                            "z_min": 0.9, "z_max": 1.0,
                            "method": "single",
                            "kwargs": {},
                        },
                    ],
                },
            )
        )

        # ── Sweep nb cellules zone basse ─────────────────────────────
        for nr, nz in [(3, 5), (5, 8), (8, 10), (10, 12)]:
            configs.append(
                ExperimentConfig(
                    method="multizone",
                    method_kwargs={
                        "z_mode": "quantile",
                        "zones": [
                            {
                                "z_min": 0.0, "z_max": 0.75,
                                "method": "cylindrical",
                                "kwargs": {
                                    "nr": nr, "ntheta": 8, "nz": nz,
                                    "radial_mode": "equal_area",
                                },
                            },
                            {
                                "z_min": 0.75, "z_max": 1.0,
                                "method": "single",
                                "kwargs": {},
                            },
                        ],
                    },
                )
            )

    elif method == "single":
        configs.append(
            ExperimentConfig(
                method="single",
                method_kwargs={},
            )
        )

    else:
        raise ValueError(f"Méthode inconnue: {method}")

    # ══════════════════════════════════════════════════════════════════════
    # Sweeps temporels (avec discrétisation spatiale par défaut)
    # ══════════════════════════════════════════════════════════════════════

    default_kwargs = _get_default_kwargs(method)

    # ── Sweep NLT ────────────────────────────────────────────────────────
    for nlt in [10, 20, 50, 100, 150, 200, 300, 500]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                nlt=nlt,
            )
        )

    # ── Sweep step_size ──────────────────────────────────────────────────
    for step in [1, 2, 3, 5, 8, 10, 15, 20]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                step_size=step,
            )
        )

    # ── Sweep start_index ────────────────────────────────────────────────
    for start in [250, 500, 1000, 2000, 3000, 5000]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                start_index=start,
            )
        )

    # ── Sweep dt ─────────────────────────────────────────────────────────
    for dt in [1, 5, 10, 25, 50]:
        configs.append(
            ExperimentConfig(
                method=method,
                method_kwargs=default_kwargs,
                dt=dt,
            )
        )

    # ── Dédoublonner ─────────────────────────────────────────────────────
    seen = set()
    unique = []
    for c in configs:
        key = c.output_folder()
        if key not in seen:
            seen.add(key)
            unique.append(c)

    return unique


def _get_default_kwargs(method):
    """Paramètres de discrétisation par défaut pour les sweeps temporels."""
    defaults = {
        "cartesian": {"nx": 5, "ny": 5, "nz": 5},
        "cylindrical": {
            "nr": 5, "ntheta": 8, "nz": 5,
            "radial_mode": "equal_area",
        },
        "voronoi": {"n_cells": 125},
        "quantile": {"nx": 5, "ny": 5, "nz": 5},
        "octree": {"max_particles": 100, "max_depth": 5},
        "physics": {"n_cells": 125},
        "adaptive": {
            "z_split": 0.75,
            "z_split_mode": "quantile",
            "n_cells_top": 1,
            "top_method": "single",
            "top_kwargs": {},
            "bottom_method": "cylindrical",
            "bottom_kwargs": {
                "nr": 5, "ntheta": 8, "nz": 8,
                "radial_mode": "equal_area",
            },
        },
        "multizone": {
            "z_mode": "quantile",
            "zones": [
                {
                    "z_min": 0.0, "z_max": 0.75,
                    "method": "cylindrical",
                    "kwargs": {
                        "nr": 5, "ntheta": 8, "nz": 8,
                        "radial_mode": "equal_area",
                    },
                },
                {
                    "z_min": 0.75, "z_max": 1.0,
                    "method": "single",
                    "kwargs": {},
                },
            ],
        },
        "single": {},
    }
    return defaults.get(method, {})

# =============================================================================
# CHARGEMENT DES DONNÉES
# =============================================================================


def sample_coordinates(files, fs, sample_rate=SAMPLE_RATE):
    """
    Échantillonne des coordonnées pour le fit des partitionneurs.

    Returns:
        np.ndarray shape (N, 3)
    """
    all_coords = []
    for f in tqdm(files[::sample_rate], desc="   Échantillonnage", leave=False):
        with fs.open(f, "rb") as fh:
            df = pl.read_csv(fh)
        coords = np.column_stack(
            [
                df["coordinates:0"].to_numpy(),
                df["coordinates:1"].to_numpy(),
                df["coordinates:2"].to_numpy(),
            ]
        )
        all_coords.append(coords)
    return np.vstack(all_coords)


# =============================================================================
# CALCUL MATRICE DE TRANSITION
# =============================================================================




# def phi_particule(state: int, partition: int) -> bool:
#     """Vérifie si une particule est bien dans une partition"""
#     return 1 if state == partition else 0

# def phi_sum_partition(states, partition: int) -> int:
#     """Somme les particules qui sont dans une partition"""
#     phi_s = 0
#     for i in range(len(states)):
#         phi_s += phi_particule(states[i], partition=partition)
#     return phi_s

# def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu"):
#     """
#     Calcule P_n pour un timestep en utilisant phi_particule et phi_sum_partition.
#     Normalisation par colonnes (somme des colonnes = 1).
#     """
#     # Conversion en tensor si nécessaire
#     if isinstance(states_curr, np.ndarray):
#         states_curr = torch.from_numpy(states_curr)
#     if isinstance(states_prev, np.ndarray):
#         states_prev = torch.from_numpy(states_prev)
    
#     s_prev = states_prev.to(device).long()
#     s_curr = states_curr.to(device).long()
    
#     # Initialisation de la matrice de transition
#     P = torch.zeros((n_states, n_states), device=device, dtype=torch.float64)
    
#     # Calcul des transitions P[i,j] = probabilité d'aller de i à j
#     for i in range(n_states):
#         for j in range(n_states):
#             # Compte les transitions de i vers j
#             inter = 0
#             n = min(len(s_prev), len(s_curr))
#             for p in range(n):
#                 inter += phi_particule(state=s_prev[p].item(), partition=i) * phi_particule(state=s_curr[p].item(), partition=j)
            
#             # Normalisation par le nombre de particules dans l'état i au temps précédent
#             denominator = phi_sum_partition(s_prev.cpu().numpy(), i)
#             P[i, j] = inter / denominator if denominator > 0 else 0.0
    
#     # Transposition pour avoir les états courants en lignes, précédents en colonnes
#     P = P.T
    
#     # # Normalisation par colonnes (somme des colonnes = 1) avec torch.sum(dim=0)
#     # col_sums = torch.sum(P, dim=0)
    
#     # P = torch.where(col_sums > 0, P / col_sums, torch.zeros_like(P))
    
#     return P



def compute_P_matrix_torch(states_prev, states_curr, n_states, device="cpu"):
    """
    Version robuste et documentée.
    P[curr, prev] = probabilité de transition de l'état prev vers curr.
    """
    if n_states == 0:
        return torch.empty((0, 0), device=device, dtype=torch.float64)

    # Conversion et envoi sur device
    if isinstance(states_prev, np.ndarray):
        states_prev = torch.from_numpy(states_prev)
    if isinstance(states_curr, np.ndarray):
        states_curr = torch.from_numpy(states_curr)

    s_prev = states_prev.to(device).long()
    s_curr = states_curr.to(device).long()

    n = min(len(s_prev), len(s_curr))
    s_prev = s_prev[:n]
    s_curr = s_curr[:n]

    # Masques one-hot en float64
    phi_prev = (s_prev.unsqueeze(1) == torch.arange(n_states, device=device)).to(torch.float64)
    phi_curr = (s_curr.unsqueeze(1) == torch.arange(n_states, device=device)).to(torch.float64)

    # Matrice de co-occurrence : transitions[i,j] = nb de i->j
    transitions = phi_prev.T @ phi_curr  # (n_states, n_states)

    # Dénominateur : nombre de particules dans chaque état au temps t
    denominator = phi_prev.sum(dim=0)    # (n_states,)

    # Construction de P sans division par zéro
    P = torch.zeros((n_states, n_states), device=device, dtype=torch.float64)
    nonzero = denominator > 0
    if nonzero.any():
        # P[curr, prev] = transitions[prev, curr] / denominator[prev]
        # On utilise l'indexation booléenne
        P[:, nonzero] = (transitions.T[:, nonzero] / denominator[nonzero].unsqueeze(0))

    return P

def compute_P_matrix_batch(
    states_prev_batch: torch.Tensor,   # (n_pairs, n_particles)
    states_curr_batch: torch.Tensor,   # (n_pairs, n_particles)
    n_states: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Calcule la matrice de transition pour plusieurs paires en batch.
    
    Returns:
        P_batch: (n_pairs, n_states, n_states) avec P_batch[k, curr, prev] = prob(prev -> curr) pour la paire k
    """
    n_pairs, n_particles = states_prev_batch.shape
    assert states_curr_batch.shape == (n_pairs, n_particles)
    
    # Reshape pour traiter toutes les paires en parallèle
    # On met les particules en dimension 1, les paires en dimension 0
    s_prev = states_prev_batch.to(device).long()  # (n_pairs, n_particles)
    s_curr = states_curr_batch.to(device).long()
    
    # One-hot encoding vectorisé sur les deux premières dimensions
    # shape: (n_pairs, n_particles, n_states)
    phi_prev = (s_prev.unsqueeze(-1) == torch.arange(n_states, device=device)).to(torch.float64)
    phi_curr = (s_curr.unsqueeze(-1) == torch.arange(n_states, device=device)).to(torch.float64)
    
    # Transitions pour chaque paire : (n_pairs, n_states, n_states)
    transitions = torch.einsum('pij,pik->pjk', phi_prev, phi_curr)  # ou torch.bmm(phi_prev.transpose(1,2), phi_curr)
    denominator = phi_prev.sum(dim=1)  # (n_pairs, n_states)
    
    # Normalisation (éviter division par zéro)
    P_batch = torch.zeros((n_pairs, n_states, n_states), device=device, dtype=torch.float64)
    mask = denominator > 0
    # P_batch[:, curr, prev] = transitions[:, prev, curr] / denominator[:, prev]
    # On utilise l'indexation avancée
    for k in range(n_pairs):
        P_batch[k, :, mask[k]] = (transitions[k].T[:, mask[k]] / denominator[k, mask[k]].unsqueeze(0))
    
    return P_batch





# =============================================================================
# EXPÉRIENCE
# =============================================================================
import torch
import numpy as np
import polars as pl
from tqdm import tqdm

def compute_P_matrix_batch(
    states_prev_batch: torch.Tensor,   # (n_pairs, n_particles)
    states_curr_batch: torch.Tensor,   # (n_pairs, n_particles)
    n_states: int,
    device: str = "cpu"
) -> torch.Tensor:
    """
    Calcule la matrice de transition pour plusieurs paires en batch.

    Args:
        states_prev_batch: tenseur (n_pairs, n_particles) des états au temps t
        states_curr_batch: tenseur (n_pairs, n_particles) des états au temps t+1
        n_states: nombre total d'états
        device: 'cpu' ou 'cuda'

    Returns:
        P_batch: (n_pairs, n_states, n_states) avec P_batch[k, curr, prev] = prob(prev -> curr) pour la paire k
    """
    n_pairs, n_particles = states_prev_batch.shape
    assert states_curr_batch.shape == (n_pairs, n_particles)

    s_prev = states_prev_batch.to(device).long()
    s_curr = states_curr_batch.to(device).long()

    # Masques one-hot vectorisés sur les paires et les particules
    # shape: (n_pairs, n_particles, n_states)
    phi_prev = (s_prev.unsqueeze(-1) == torch.arange(n_states, device=device)).to(torch.float64)
    phi_curr = (s_curr.unsqueeze(-1) == torch.arange(n_states, device=device)).to(torch.float64)

    # Transitions pour chaque paire : (n_pairs, n_states, n_states)
    # einsum: p = indice de paire, i = état prev, j = état curr, k = particule
    transitions = torch.einsum('pki,pkj->pij', phi_prev, phi_curr)  # transitions[p, i, j]
    denominator = phi_prev.sum(dim=1)  # (n_pairs, n_states)

    # Normalisation : P[p, curr, prev] = transitions[p, prev, curr] / denominator[p, prev]
    P_batch = torch.zeros((n_pairs, n_states, n_states), device=device, dtype=torch.float64)
    mask = denominator > 0  # (n_pairs, n_states)

    # Pour chaque paire, on ne normalise que les colonnes (prev) non nulles
    for p in range(n_pairs):
        # indices des états prev avec au moins une particule
        prev_nonzero = mask[p]
        if prev_nonzero.any():
            # transitions[p, prev_nonzero, :]  -> shape (nz, n_states)
            # on transpose pour obtenir (n_states, nz)
            trans_sub = transitions[p, :, :].T[:, prev_nonzero]  # (n_states, nz)
            denom_sub = denominator[p, prev_nonzero]             # (nz,)
            P_batch[p, :, prev_nonzero] = trans_sub / denom_sub.unsqueeze(0)

    return P_batch


def run_experiment(config, partitioner, files, fs, device):
    """
    Exécute une expérience complète avec fenêtre glissante et traitement par batch.
    """
    n_states = partitioner.n_cells
    step = config.step_size
    dt = config.dt

    # Vérification de faisabilité
    last_needed = config.start_index + (config.nlt - 1) * dt + step
    if last_needed >= len(files):
        max_nlt = (len(files) - 1 - config.start_index - step) // dt + 1
        max_nlt = max(max_nlt, 0)
        print(f"   ⚠️  Seulement {max_nlt} paires possibles (demandé: {config.nlt})")
        actual_nlt = max_nlt
    else:
        actual_nlt = config.nlt

    if actual_nlt <= 0:
        raise ValueError(f"Aucune paire possible: start={config.start_index}, step={step}, dt={dt}")

    # Construction des paires
    pairs = [(config.start_index + k * dt, config.start_index + k * dt + step) for k in range(actual_nlt)]

    # Affichage d'info
    ratio = dt / step
    if ratio < 1:
        overlap_pct = round((1 - ratio) * 100, 1)
        print(f"   🔄 Recouvrement: {overlap_pct}% (step/dt = {step/dt:.1f})")
    print(f"   📐 {actual_nlt} paires | step={step} | dt={dt}")
    print(f"   📂 Première paire: fichiers {pairs[0][0]} → {pairs[0][1]}")
    print(f"   📂 Dernière paire:  fichiers {pairs[-1][0]} → {pairs[-1][1]}")

    # Paramètres du batch
    batch_size = 100  # ajustable selon mémoire GPU
    P_acc = torch.zeros((n_states, n_states), device=device, dtype=torch.float64)

    # Boucle sur les batches
    for start_idx in range(0, len(pairs), batch_size):
        end_idx = min(start_idx + batch_size, len(pairs))
        batch_pairs = pairs[start_idx:end_idx]

        batch_prev = []
        batch_curr = []

        # Lecture et calcul des états pour toutes les paires du batch
        for idx_prev, idx_curr in batch_pairs:
            with fs.open(files[idx_prev], "rb") as f:
                df_prev = pl.read_csv(f)
            with fs.open(files[idx_curr], "rb") as f:
                df_curr = pl.read_csv(f)

            states_prev = partitioner.compute_states(
                df_prev["coordinates:0"],
                df_prev["coordinates:1"],
                df_prev["coordinates:2"],
            )
            states_curr = partitioner.compute_states(
                df_curr["coordinates:0"],
                df_curr["coordinates:1"],
                df_curr["coordinates:2"],
            )

            batch_prev.append(torch.from_numpy(states_prev))
            batch_curr.append(torch.from_numpy(states_curr))

        # Empilement en tenseurs (batch_size, n_particles)
        batch_prev_t = torch.stack(batch_prev).to(device)
        batch_curr_t = torch.stack(batch_curr).to(device)

        # Calcul des matrices de transition pour tout le batch
        P_batch = compute_P_matrix_batch(batch_prev_t, batch_curr_t, n_states, device)

        # Accumulation : somme sur la dimension des paires
        P_acc += P_batch.sum(dim=0)

    # Moyenne sur les paires
    P = P_acc / actual_nlt
    P_np = P.cpu().numpy()

    # Statistiques (inchangées)
    column_sums = P_np.sum(axis=0)
    visited = column_sums > 0
    diag = np.diag(P_np)

    stats = {
        "n_timesteps_used": actual_nlt,
        "n_states": n_states,
        "n_states_visited": int(visited.sum()),
        "n_states_empty": int((~visited).sum()),
        "fraction_visited": round(float(visited.sum()) / n_states, 4),
        "column_sum_min": float(column_sums[visited].min()) if visited.any() else 0,
        "column_sum_max": float(column_sums[visited].max()) if visited.any() else 0,
        "column_sum_mean": float(column_sums[visited].mean()) if visited.any() else 0,
        "diagonal_mean": float(diag.mean()),
        "diagonal_std": float(diag.std()),
        "method": config.method,
        "step_size": step,
        "dt": dt,
        "overlap_ratio": round(max(0, 1 - dt / step), 4),
        "n_paires_par_step": step // dt if dt > 0 else 0,
        "plage_temporelle": int(pairs[-1][1] - pairs[0][0]),
        "start_index": config.start_index,
        "first_pair": list(pairs[0]),
        "last_pair": list(pairs[-1]),
    }

    return P_np, stats

def save_results(config, partitioner, P, stats, output_dir):
    """Sauvegarde les résultats dans le bucket HuggingFace."""
    
    folder_name = os.path.basename(output_dir)
    
    # Préparer les données du partitionneur
    partitioner_data = {}
    if hasattr(partitioner, 'centroids') and partitioner.centroids is not None:
        partitioner_data["centroids"] = partitioner.centroids
    if hasattr(partitioner, '_r_edges') and partitioner._r_edges is not None:
        partitioner_data["r_edges"] = partitioner._r_edges
    if hasattr(partitioner, '_leaves') and partitioner._leaves:
        partitioner_data["leaves"] = np.array(partitioner._leaves)
    if hasattr(partitioner, '_x_edges') and partitioner._x_edges is not None:
        partitioner_data["x_edges"] = partitioner._x_edges
        partitioner_data["y_edges"] = partitioner._y_edges
        partitioner_data["z_edges"] = partitioner._z_edges
    
    # Métadonnées du partitionneur
    partitioner_data["partitioner_meta"] = {
        "type": type(partitioner).__name__,
        "label": partitioner.label,
        "n_cells": partitioner.n_cells,
    }
    
    # Sauvegarder dans le bucket
    save_experiment_to_bucket(
        folder_name=folder_name,
        matrix=P,
        stats=stats,
        config=asdict(config),
        partitioner_data=partitioner_data,
    )
    
    print(f"   💾 Bucket: {BUCKET_BASE}/{folder_name}/")
# =============================================================================
# FONCTION PRINCIPALE
# =============================================================================


def run_markov_sweep(method:str, configs:list[ExperimentConfig]=None, base_dir=BASE_OUTPUT_DIR)-> list[dict]:
    """
    Lance le sweep Markovien pour une méthode de partitionnement.

    Args:
        method: str — "cartesian", "cylindrical", "voronoi",
                       "quantile", "octree", "physics", ou "all"
        configs: liste de ExperimentConfig (None = configs par défaut)
        base_dir: dossier de sortie

    Exemple:
        run_markov_sweep("voronoi")
        run_markov_sweep("cylindrical", configs=[
            ExperimentConfig(method="cylindrical",
                             method_kwargs={"nr":10, "ntheta":8, "nz":10}),
        ])
    """

    print("=" * 70)
    print(f"  SWEEP MARKOVIEN — méthode: {method.upper()}")
    print("=" * 70)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # ── Fichiers ──
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))
    print(f"📁 Fichiers disponibles: {len(files)}")

    # ── Coordonnées pour fit ──
    print("\n🔍 Échantillonnage des coordonnées pour le fit...")
    sample_coords = sample_coordinates(files, fs)
    print(f"   {len(sample_coords)} points échantillonnés")

    # ── Configs ──
    if method == "all":
        methods = list(REGISTRY.keys())
    else:
        methods = [method]

    if configs is None:
        all_configs = []
        for m in methods:
            all_configs.extend(get_configs(m))
    else:
        all_configs = configs

    print(f"\n📋 {len(all_configs)} expériences à lancer:")
    print("-" * 70)
    for i, c in enumerate(all_configs):
        part = create_partitioner(c.method, **c.method_kwargs)
        part.fit(sample_coords)
        print(
            f"  {i + 1:3d}. [{c.method:12s}] {part.label:40s} "
            f"NLT={c.nlt:4d} step={c.step_size:2d} start={c.start_index}"
        )
    print("-" * 70)

    # ── Cache des partitionneurs fittés ──
    fitted_cache = {}

    # ── Boucle principale ──
    results = []
    for i, config in enumerate(all_configs):
        if config.method=="adaptive" or config.method=="multizone":
            output_dir=config.output_folder(base_dir=base_dir,sample_coords=sample_coords)
        else :
            output_dir = config.output_folder(base_dir)
        print(f"\n[{i + 1}/{len(all_configs)}] {os.path.basename(output_dir)}")

        try:
            # Créer ou récupérer le partitionneur
            partitioner = create_partitioner(config.method, **config.method_kwargs)
            cache_key = partitioner.label

            if cache_key in fitted_cache:
                partitioner = fitted_cache[cache_key]
                print(f"   ♻️  Partitionneur en cache: {cache_key}")
            else:
                print(f"   🔧 Fit: {cache_key}...")
                partitioner.fit(sample_coords)
                fitted_cache[cache_key] = partitioner

                # Diagnostics
                diag = partitioner.diagnostics(sample_coords)
                print(
                    f"   📊 {partitioner.n_cells} cellules | "
                    f"{diag['n_visited']} visitées | "
                    f"pop: [{diag['pop_min']}, {diag['pop_max']}] "
                    f"μ={diag['pop_mean']:.0f} σ={diag['pop_std']:.0f}"
                )

            # Lancer l'expérience
            P, stats = run_experiment(config, partitioner, files, fs, device)

            # Sauvegarder
            save_results(config, partitioner, P, stats, output_dir)

            results.append(
                {"config": asdict(config), "stats": stats, "success": True}
            )
            print(
                f"   ✅ {stats['n_states_visited']}/{stats['n_states']} états | "
                f"P(rester)={stats['diagonal_mean']:.4f} | "
                f"Σrow=[{stats['column_sum_min']:.4f}, {stats['column_sum_max']:.4f}]"
            )

        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            results.append(
                {
                    "config": asdict(config),
                    "stats": None,
                    "success": False,
                    "error": str(e),
                }
            )

    # ── Résumé ──
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)

    ok = [r for r in results if r["success"]]
    ko = [r for r in results if not r["success"]]
    print(f"\n✅ Réussies: {len(ok)}/{len(results)}")
    if ko:
        print(f"❌ Échouées: {len(ko)}")
        for r in ko:
            print(f"   - {r['config']['method']}: {r.get('error', '?')}")

    # Sauvegarder le résumé
    summary_path = os.path.join(base_dir, f"summary_{method}.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Résumé: {summary_path}")
    print("✨ Terminé!")

    return results


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Sweep Markovien multi-partitionnement"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cartesian",
        choices=list(REGISTRY.keys()) + ["all"],  # ← inclut automatiquement adaptive, multizone, single
        help="Type de partitionnement (default: cartesian)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=BASE_OUTPUT_DIR,
        help=f"Dossier de sortie (default: {BASE_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lister les configurations sans lancer les calculs",
    )
    args = parser.parse_args()

    if args.list:
        if args.method == "all":
            for m in REGISTRY:
                configs = get_configs(m)
                print(f"\n{m.upper()} ({len(configs)} configs):")
                for c in configs:
                    p = create_partitioner(c.method, **c.method_kwargs)
                    print(f"  {p.label} NLT={c.nlt} step={c.step_size} dt={c.dt}")
        else:
            configs = get_configs(args.method)
            print(f"{args.method.upper()} ({len(configs)} configs):")
            for c in configs:
                p = create_partitioner(c.method, **c.method_kwargs)
                print(f"  {p.label} NLT={c.nlt} step={c.step_size} dt={c.dt}")
        return

    run_markov_sweep(args.method, base_dir=args.output)


if __name__ == "__main__":
    main()