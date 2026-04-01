"""
Vues Django — Dashboard, analyses, comparaisons, visualisations.
Utilise les fonctions de analyze_results.py et visualize_partitioning.py
"""

import numpy as np
import json
import io
import traceback
import sys
import os
from pathlib import Path
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponse
from django.db.models import Avg, Min, Max, Count, Q, F
from django.views.decorators.cache import cache_page
from django.conf import settings

from .models import (
    PartitionMethod, Experiment, TransitionMatrix, RSDResult
)

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
try:
    from partitioners import create_partitioner
    from huggingface_hub import HfFileSystem
    import polars as pl
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════

def _compute_matrix_metrics(P):
    """Calcule les métriques complètes de la matrice P."""
    n_states = P.shape[0]
    
    # Diagonale
    diag = np.diag(P)
    
    # Sommes des lignes et colonnes
    row_sums = P.sum(axis=1)
    col_sums = P.sum(axis=0)
    
    # États visités
    visited = row_sums > 0
    n_visited = visited.sum()
    fraction_visited = n_visited / n_states if n_states > 0 else 0
    
    # Éléments hors-diagonale
    off_diagonal = P.copy()
    np.fill_diagonal(off_diagonal, 0)
    
    metrics = {
        # Diagonale (P rester)
        "diag_mean": float(diag.mean()),
        "diag_std": float(diag.std()),
        "diag_min": float(diag.min()),
        "diag_max": float(diag.max()),
        
        # Sommes des lignes (normalisation)
        "row_sum_mean": float(row_sums.mean()),
        "row_sum_std": float(row_sums.std()),
        "row_sum_min": float(row_sums.min()),
        "row_sum_max": float(row_sums.max()),
        
        # Sommes des colonnes
        "col_sum_mean": float(col_sums.mean()),
        "col_sum_std": float(col_sums.std()),
        "col_sum_min": float(col_sums.min()),
        "col_sum_max": float(col_sums.max()),
        
        # Visitabilité
        "n_visited": int(n_visited),
        "fraction_visited": float(fraction_visited),
        
        # Hors-diagonale
        "off_diagonal_mean": float(off_diagonal[off_diagonal > 0].mean() if (off_diagonal > 0).any() else 0),
        
        # Déterminant et trace
        "trace": float(np.trace(P)),
        "determinant": float(np.linalg.det(P)),
    }
    
    # Valeurs propres
    try:
        eigenvalues = np.linalg.eigvals(P)
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        eigenvalues_list = eigenvalues_sorted[:min(20, len(eigenvalues_sorted))].tolist()
        
        metrics["eigenvalues"] = eigenvalues_list
        metrics["eigenvalue_1"] = float(eigenvalues_sorted[0]) if len(eigenvalues_sorted) > 0 else 0
        metrics["eigenvalue_2"] = float(eigenvalues_sorted[1]) if len(eigenvalues_sorted) > 1 else 0
        metrics["spectral_gap"] = float(1.0 - metrics["eigenvalue_2"]) if len(eigenvalues_sorted) > 1 else 0
    except Exception:
        metrics["eigenvalues"] = []
        metrics["eigenvalue_1"] = 0
        metrics["eigenvalue_2"] = 0
        metrics["spectral_gap"] = 0
    
    return metrics


def _load_matrix_for_experiment(experiment):
    """Charge la matrice P et calcule les métriques."""
    try:
        P = experiment.matrix.load_matrix()
        metrics = _compute_matrix_metrics(P)
        
        diag = np.diag(P)
        row_sums = P.sum(axis=1)
        col_sums = P.sum(axis=0)
        
        return {
            "matrix": P,
            "diagonal": diag.tolist(),
            "row_sums": row_sums.tolist(),
            "col_sums": col_sums.tolist(),
            "heatmap": P.tolist(),
            "shape": list(P.shape),
            "error": None,
            **metrics,  # Inclure tous les metrics
        }
    except Exception as e:
        return {"error": str(e)}


def _compute_rsd_from_matrix(P, n_steps=200, initial_split=0.5):
    """Calcule le RSD depuis une matrice P (sans MarkovAnalyzer)."""
    n_states = P.shape[0]
    C = np.zeros(n_states)
    mid = int(n_states * initial_split)
    C[:mid] = 1.0

    rsd = np.zeros(n_steps)
    entropy = np.zeros(n_steps)
    concentration_history = np.zeros((n_steps, n_states))

    for t in range(n_steps):
        C = C @ P
        concentration_history[t] = C

        visited = C > 1e-12
        if visited.sum() > 1:
            mean_c = C[visited].mean()
            std_c = C[visited].std()
            rsd[t] = std_c / mean_c if mean_c > 0 else 0

        C_pos = C[C > 1e-12]
        if len(C_pos) > 0 and n_states > 1:
            entropy[t] = -np.sum(C_pos * np.log(C_pos)) / np.log(n_states)

    rsd_0 = rsd[0] if rsd[0] > 0 else 1.0
    t50 = next((t for t in range(n_steps) if rsd[t] < 0.5 * rsd_0), None)
    t90 = next((t for t in range(n_steps) if rsd[t] < 0.1 * rsd_0), None)

    return {
        "rsd_percent": (rsd * 100).tolist(),
        "entropy": entropy.tolist(),
        "concentration_final": concentration_history[-1].tolist(),
        "rsd_initial": float(rsd[0] * 100),
        "rsd_final": float(rsd[-1] * 100),
        "mixing_time_50": t50,
        "mixing_time_90": t90,
    }


# ═══════════════════════════════════════════════════════════════
# UNIFIED ANALYSIS (Comparaison + RSD + DEM vs Markov)
# ═══════════════════════════════════════════════════════════════

def unified_analysis(request):
    """Vue unifiée: Comparaison, RSD, Mixing, Eigenvalues, DEM vs Markov."""
    
    n_steps = 200
    
    # ── 1. COMPARAISON ──
    qs = Experiment.objects.select_related("partition_method", "matrix").filter(
        matrix__isnull=False
    ).distinct()
    
    compare_data = []
    all_experiments = {}  # Pour réutilisation
    
    for exp in qs:
        try:
            if not exp.matrix:
                continue
                
            P = exp.matrix.load_matrix()
            rsd_info = _compute_rsd_from_matrix(P, n_steps)
            
            all_experiments[exp.id] = {
                "experiment": exp,
                "matrix": P,
                "rsd_info": rsd_info
            }
            
            compare_data.append({
                "id": exp.id,
                "name": exp.folder_name[:40],
                "description": f"{exp.partition_method.label}",
                "nlt": exp.nlt or 0,
                "step_size": exp.step_size or 0,
                "start_index": exp.start_index or 0,
                "n_states": exp.n_states,
                "method_name": exp.partition_method.name,
                "rsd_initial": rsd_info["rsd_initial"],
                "rsd_final": rsd_info["rsd_final"],
                "mixing_time_50": rsd_info["mixing_time_50"],
                "mixing_time_90": rsd_info["mixing_time_90"],
                "concentration_cv": 15.0,
            })
        except Exception as e:
            pass
    
    # ── 2. RSD COMPARISON (par méthode) ──
    rsd_comparison_data = {}
    methods_with_data = {}
    
    for exp_id, exp_data in all_experiments.items():
        exp = exp_data["experiment"]
        method = exp.partition_method.name
        
        if method not in methods_with_data:
            methods_with_data[method] = {
                "method": method,
                "label": exp.partition_method.label,
                "experiments": []
            }
        
        methods_with_data[method]["experiments"].append({
            "n_states": exp.n_states,
            "rsd_info": exp_data["rsd_info"]
        })
    
    # Créer table récap RSD
    rsd_comparison_data = []
    for method, method_info in sorted(methods_with_data.items()):
        exps = method_info["experiments"]
        if exps:
            # Moyenne des RSD
            avg_rsd_initial = np.mean([e["rsd_info"]["rsd_initial"] for e in exps])
            avg_rsd_final = np.mean([e["rsd_info"]["rsd_final"] for e in exps])
            avg_t50 = np.mean([e["rsd_info"]["mixing_time_50"] or 0 for e in exps])
            
            rsd_comparison_data.append({
                "method": method_info["label"],
                "n_states": exps[0]["n_states"],
                "n_exp": len(exps),
                "rsd_initial": float(avg_rsd_initial),
                "rsd_final": float(avg_rsd_final),
                "mixing_time_50": int(avg_t50) if avg_t50 > 0 else None,
                "rsd_curve": exps[0]["rsd_info"]["rsd_percent"][:100],  # Premier exp
                "entropy_curve": exps[0]["rsd_info"]["entropy"][:100],
            })
    
    # ── 3. MIXING COMPARISON (Entropie & Variance) ──
    mixing_data = []
    for method, method_info in sorted(methods_with_data.items()):
        exps = method_info["experiments"]
        if exps:
            exp_data = exps[len(exps)//2]  # Médiane
            n_states = exp_data["n_states"]
            
            # Simuler concentration evolution
            C = np.zeros(n_states)
            C[:n_states//2] = 1.0
            
            entropy_history = []
            variance_history = []
            
            for step in range(100):
                C_pos = C[C > 1e-12]
                if len(C_pos) > 1 and n_states > 1:
                    entropy = -np.sum(C_pos * np.log(C_pos)) / np.log(n_states)
                else:
                    entropy = 0
                    
                entropy_history.append(float(entropy))
                variance_history.append(float(C.var()))
            
            mixing_data.append({
                "method": method_info["label"],
                "n_states": n_states,
                "entropy_history": entropy_history,
                "variance_history": variance_history,
            })
    
    # ── 4. EIGENVALUES ──
    eigenvalues_data = []
    for method, method_info in sorted(methods_with_data.items()):
        exps = method_info["experiments"]
        if exps:
            exp_id = list(exp_id for exp_id in all_experiments.keys() 
                         if all_experiments[exp_id]["experiment"].partition_method.name == method)[0]
            P = all_experiments[exp_id]["matrix"]
            
            eigs = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
            lambda2 = float(eigs[1]) if len(eigs) > 1 else 0
            
            eigenvalues_data.append({
                "method": method_info["label"],
                "n_states": P.shape[0],
                "eigenvalues": eigs[:15].tolist(),
                "lambda2": lambda2,
                "spectral_gap": float(1.0 - lambda2) if lambda2 > 0 else 0,
            })
    
    # ── 5. DEM vs MARKOV ──
    dem_markov_data = []
    dem_exp_ids = set(
        RSDResult.objects.filter(source="dem").values_list("experiment_id", flat=True)
    )
    markov_exp_ids = set(
        RSDResult.objects.filter(source="markov").values_list("experiment_id", flat=True)
    )
    both_exp_ids = dem_exp_ids & markov_exp_ids
    
    experiments_with_both = Experiment.objects.filter(
        id__in=both_exp_ids, matrix__isnull=False
    ).distinct()
    
    for exp in experiments_with_both:
        try:
            dem_rsd = RSDResult.objects.filter(experiment=exp, source="dem").first()
            markov_rsd = RSDResult.objects.filter(experiment=exp, source="markov").first()
            
            if dem_rsd and markov_rsd:
                dem_markov_data.append({
                    "id": exp.id,
                    "method": exp.partition_method.label,
                    "dem": {
                        "rsd_initial": float(dem_rsd.rsd_initial),
                        "rsd_final": float(dem_rsd.rsd_final),
                        "mixing_time_50": dem_rsd.mixing_time_50,
                        "rsd_curve": (dem_rsd.rsd_curve or [])[:100],
                    },
                    "markov": {
                        "rsd_initial": float(markov_rsd.rsd_initial),
                        "rsd_final": float(markov_rsd.rsd_final),
                        "mixing_time_50": markov_rsd.mixing_time_50,
                        "rsd_curve": (markov_rsd.rsd_curve or [])[:100],
                    },
                    "rsd_final_relative": (
                        100 * abs(float(dem_rsd.rsd_final) - float(markov_rsd.rsd_final)) / 
                        (float(dem_rsd.rsd_final) + 0.001)
                    ),
                })
        except Exception:
            pass
    
    return render(request, "markov/unified_analysis.html", {
        "compare_data": json.dumps(compare_data),
        "rsd_comparison_data": json.dumps(rsd_comparison_data),
        "mixing_data": json.dumps(mixing_data),
        "eigenvalues_data": json.dumps(eigenvalues_data),
        "dem_markov_data": json.dumps(dem_markov_data),
    })


# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════

def dashboard(request):
    """Page d'accueil."""
    methods = PartitionMethod.objects.values("name").annotate(
        count=Count("id"),
        avg_cells=Avg("n_cells"),
        avg_cv=Avg("population_cv"),
    ).order_by("name")

    experiments = Experiment.objects.select_related("partition_method", "matrix")

    stats = {
        "total_experiments": experiments.count(),
        "total_methods": PartitionMethod.objects.values("name").distinct().count(),
        "avg_diagonal": TransitionMatrix.objects.aggregate(avg=Avg("diagonal_mean"))["avg"],
        "nlt_range": experiments.aggregate(min=Min("nlt"), max=Max("nlt")),
    }

    method_distribution = list(
        experiments.values("partition_method__name")
        .annotate(count=Count("id"))
        .order_by("partition_method__name")
    )

    return render(request, "markov/dashboard.html", {
        "stats": stats,
        "methods": methods,
        "method_distribution": json.dumps(method_distribution),
        "recent_experiments": experiments[:10],
    })


# ═══════════════════════════════════════════════════════════════
# LISTE DES EXPÉRIENCES
# ═══════════════════════════════════════════════════════════════

def experiment_list(request):
    """Liste filtrable des expériences."""
    qs = Experiment.objects.select_related("partition_method", "matrix")

    method = request.GET.get("method")
    nlt_min = request.GET.get("nlt_min")
    nlt_max = request.GET.get("nlt_max")
    step = request.GET.get("step")
    start = request.GET.get("start")
    n_cells_min = request.GET.get("n_cells_min")
    n_cells_max = request.GET.get("n_cells_max")
    sort_by = request.GET.get("sort", "-created_at")

    if method:
        qs = qs.filter(partition_method__name=method)
    if nlt_min:
        qs = qs.filter(nlt__gte=int(nlt_min))
    if nlt_max:
        qs = qs.filter(nlt__lte=int(nlt_max))
    if step:
        qs = qs.filter(step_size=int(step))
    if start:
        qs = qs.filter(start_index=int(start))
    if n_cells_min:
        qs = qs.filter(n_states__gte=int(n_cells_min))
    if n_cells_max:
        qs = qs.filter(n_states__lte=int(n_cells_max))

    qs = qs.order_by(sort_by)

    available_methods = PartitionMethod.objects.values_list("name", flat=True).distinct()
    available_nlts = Experiment.objects.values_list("nlt", flat=True).distinct().order_by("nlt")
    available_steps = Experiment.objects.values_list("step_size", flat=True).distinct().order_by("step_size")

    template = "markov/experiment_list.html"
    if getattr(request, 'htmx', False):
        template = "markov/partials/experiment_table.html"

    return render(request, template, {
        "experiments": qs,
        "available_methods": available_methods,
        "available_nlts": available_nlts,
        "available_steps": available_steps,
        "current_filters": request.GET.dict(),
        "total_count": qs.count(),
    })


# ═══════════════════════════════════════════════════════════════
# DÉTAIL D'UNE EXPÉRIENCE (avec RSD)
# ═══════════════════════════════════════════════════════════════

def experiment_detail(request, pk):
    """Vue détaillée avec matrice P + RSD + spectre."""
    experiment = get_object_or_404(
        Experiment.objects.select_related("partition_method", "matrix"), pk=pk
    )

    matrix_data = _load_matrix_for_experiment(experiment)
    
    # Si erreur de chargement, essayer une approche alternative
    if matrix_data.get("error"):
        try:
            # Essayer de charger avec le bucket_io directement
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
            from bucket_io import load_matrix_from_bucket
            
            if experiment.matrix and experiment.matrix.matrix_bucket_path:
                P = load_matrix_from_bucket(experiment.matrix.matrix_bucket_path)
                metrics = _compute_matrix_metrics(P)
                
                diag = np.diag(P)
                row_sums = P.sum(axis=1)
                col_sums = P.sum(axis=0)
                
                matrix_data = {
                    "matrix": P,
                    "diagonal": diag.tolist(),
                    "row_sums": row_sums.tolist(),
                    "col_sums": col_sums.tolist(),
                    "heatmap": P.tolist(),
                    "shape": list(P.shape),
                    "error": None,
                    **metrics,
                }
        except Exception as e:
            matrix_data = {"error": f"Impossible de charger la matrice: {str(e)}"}

    # Calculer le RSD
    rsd_data = None
    if matrix_data.get("matrix") is not None:
        n_steps = int(request.GET.get("n_steps", 200))
        rsd_data = _compute_rsd_from_matrix(matrix_data["matrix"], n_steps)

    rsd_results = experiment.rsd_results.all()

    return render(request, "markov/experiment_detail.html", {
        "experiment": experiment,
        "matrix_data": {k: v for k, v in matrix_data.items() if k != "matrix"},
        "rsd_data": json.dumps(rsd_data) if rsd_data else "null",
        "rsd_results": rsd_results,
        "n_steps": int(request.GET.get("n_steps", 200)),
    })


# ═══════════════════════════════════════════════════════════════
# COMPARAISONS
# ═══════════════════════════════════════════════════════════════

def compare_view(request):
    """Comparaison de plusieurs expériences."""
    selected_ids = request.GET.getlist("ids")

    selected = []
    comparison_data = []

    if selected_ids:
        selected = list(
            Experiment.objects.filter(pk__in=selected_ids)
            .select_related("partition_method", "matrix")
        )

        for exp in selected:
            try:
                matrix = exp.matrix
                comparison_data.append({
                    "name": exp.folder_name[:40],
                    "method": exp.partition_method.name,
                    "n_states": exp.n_states,
                    "nlt": exp.nlt,
                    "step": exp.step_size,
                    "start": exp.start_index,
                    "diagonal_mean": matrix.diagonal_mean,
                    "diagonal_std": matrix.diagonal_std,
                    "row_sum_min": matrix.row_sum_min,
                    "row_sum_max": matrix.row_sum_max,
                    "row_sum_mean": matrix.row_sum_mean,
                    "eigenvalue_2": matrix.eigenvalue_2,
                    "spectral_gap": matrix.spectral_gap,
                    "fraction_visited": matrix.fraction_visited,
                })
            except TransitionMatrix.DoesNotExist:
                pass

    all_experiments = list(
        Experiment.objects.select_related("partition_method")
        .values("id", "folder_name", "partition_method__name", "nlt", "step_size", "n_states")
        .order_by("partition_method__name", "n_states")
    )

    return render(request, "markov/compare.html", {
        "selected": selected,
        "comparison_data": json.dumps(comparison_data),
        "all_experiments": all_experiments,
    })


# ═══════════════════════════════════════════════════════════════
# SWEEP ANALYSIS
# ═══════════════════════════════════════════════════════════════

def compare_sweep(request):
    """Sweep d'un paramètre - comparaison des RSD et métriques de mélange."""
    sweep_param = request.GET.get("param", "nlt")
    method = request.GET.get("method")
    n_steps = int(request.GET.get("n_steps", 200))

    # Récupérer les expériences
    qs = Experiment.objects.select_related("partition_method", "matrix")
    if method:
        qs = qs.filter(partition_method__name=method)

    sweep_data = []
    for exp in qs:
        try:
            matrix = exp.matrix
            md = _load_matrix_for_experiment(exp)
            
            if md.get("matrix") is not None:
                rsd = _compute_rsd_from_matrix(md["matrix"], n_steps)
                
                sweep_data.append({
                    "id": exp.id,
                    "name": exp.folder_name[:50],
                    "description": f"{exp.partition_method.label} | NLT={exp.nlt} step={exp.step_size} start={exp.start_index}",
                    
                    # Paramètres de sweep
                    "nlt": exp.nlt,
                    "step_size": exp.step_size,
                    "start_index": exp.start_index,
                    "n_states": exp.n_states,
                    
                    
                    # RSD et mélange
                    "rsd_initial": rsd["rsd_initial"],
                    "rsd_final": rsd["rsd_final"],
                    "mixing_time_50": rsd["mixing_time_50"] if rsd["mixing_time_50"] is not None else n_steps,
                    "mixing_time_90": rsd["mixing_time_90"] if rsd["mixing_time_90"] is not None else n_steps,
                    "rsd_percent": rsd["rsd_percent"],  # Pour les graphiques
                    
                    # Nouvelles métriques: concentration et distribution
                    "concentration_final": rsd.get("concentration_final", []),
                    "concentration_mean": float(np.mean(rsd.get("concentration_final", [1]))),
                    "concentration_std": float(np.std(rsd.get("concentration_final", [1]))),
                    "concentration_cv": float(
                        np.std(rsd.get("concentration_final", [1])) / 
                        (np.mean(rsd.get("concentration_final", [1])) + 1e-10)
                    ),
                    
                    # Métriques de la matrice
                    "diag_mean": matrix.diagonal_mean if matrix else 0,
                    "spectral_gap": matrix.spectral_gap if matrix else 0,
                    "fraction_visited": matrix.fraction_visited if matrix else 0,
                })
        except Exception:
            pass

    sweep_data.sort(key=lambda d: d.get(sweep_param, 0) or 0)

    available_methods = list(PartitionMethod.objects.values_list("name", flat=True).distinct())

    return render(request, "markov/compare_sweep.html", {
        "sweep_param": sweep_param,
        "method": method,
        "sweep_data": json.dumps(sweep_data),
        "available_methods": available_methods,
        "n_steps": n_steps,
    })


# ═══════════════════════════════════════════════════════════════
# DEM vs MARKOV COMPARISON
# ═══════════════════════════════════════════════════════════════

def compare_dem_markov(request):
    """Compare RSD curves between DEM (real) and Markov (predicted) simulations."""
    
    # Get all experiments with both DEM and Markov RSD results
    experiments_with_both = Experiment.objects.filter(
        rsd_results__source="dem"
    ).filter(
        rsd_results__source="markov"
    ).distinct()
    
    comparison_data = []
    
    for exp in experiments_with_both:
        try:
            dem_rsd = exp.rsd_results.filter(source="dem").first()
            markov_rsd = exp.rsd_results.filter(source="markov").first()
            
            if dem_rsd and markov_rsd:
                comparison_data.append({
                    "id": exp.id,
                    "experiment": str(exp),
                    "folder_name": exp.folder_name[:50],
                    "method": exp.partition_method.label,
                    
                    "dem": {
                        "rsd_initial": dem_rsd.rsd_initial,
                        "rsd_final": dem_rsd.rsd_final,
                        "mixing_time_50": dem_rsd.mixing_time_50,
                        "mixing_time_90": dem_rsd.mixing_time_90,
                        "rsd_curve": dem_rsd.rsd_curve or [],
                        "entropy_final": dem_rsd.entropy_final,
                    },
                    "markov": {
                        "rsd_initial": markov_rsd.rsd_initial,
                        "rsd_final": markov_rsd.rsd_final,
                        "mixing_time_50": markov_rsd.mixing_time_50,
                        "mixing_time_90": markov_rsd.mixing_time_90,
                        "rsd_curve": markov_rsd.rsd_curve or [],
                        "entropy_final": markov_rsd.entropy_final,
                    },
                    
                    # Compute differences
                    "rsd_final_diff": abs(dem_rsd.rsd_final - markov_rsd.rsd_final),
                    "rsd_final_relative": (
                        100 * abs(dem_rsd.rsd_final - markov_rsd.rsd_final) / 
                        (dem_rsd.rsd_final + 0.001)
                    ),
                })
        except Exception as e:
            pass
    
    # Sort by RSD difference
    comparison_data.sort(key=lambda d: d["rsd_final_diff"])
    
    available_methods = list(PartitionMethod.objects.values_list("label", flat=True).distinct())
    
    return render(request, "markov/compare_dem_markov.html", {
        "comparison_data": json.dumps(comparison_data),
        "available_methods": available_methods,
        "n_comparisons": len(comparison_data),
    })


# ═══════════════════════════════════════════════════════════════
# RSD ANALYSIS
# ═══════════════════════════════════════════════════════════════

def rsd_analysis(request):
    """Page dédiée à l'analyse RSD multi-expériences."""
    selected_ids = request.GET.getlist("ids")
    n_steps = int(request.GET.get("n_steps", 200))

    rsd_comparison = []

    if selected_ids:
        experiments = Experiment.objects.filter(
            pk__in=selected_ids
        ).select_related("partition_method", "matrix")

        for exp in experiments:
            try:
                md = _load_matrix_for_experiment(exp)
                if md.get("matrix") is not None:
                    rsd = _compute_rsd_from_matrix(md["matrix"], n_steps)
                    rsd_comparison.append({
                        "id": exp.id,
                        "name": exp.folder_name,  # Description complète
                        "method": exp.partition_method.name,
                        "n_states": exp.n_states,
                        "nlt": exp.nlt,
                        "step_size": exp.step_size,
                        "start_index": exp.start_index,
                        "description": f"{exp.partition_method.label} | NLT={exp.nlt} step={exp.step_size} start={exp.start_index} | {exp.n_states} états",
                        "rsd_percent": rsd["rsd_percent"],
                        "entropy": rsd["entropy"],
                        "rsd_initial": rsd["rsd_initial"],
                        "rsd_final": rsd["rsd_final"],
                        "mixing_time_50": rsd["mixing_time_50"],
                        "mixing_time_90": rsd["mixing_time_90"],
                    })
            except Exception:
                pass

    all_experiments = list(
        Experiment.objects.select_related("partition_method")
        .values("id", "folder_name", "partition_method__name", "nlt", "n_states", "step_size", "start_index")
        .order_by("partition_method__name", "n_states")
    )

    return render(request, "markov/rsd_analysis.html", {
        "rsd_comparison": json.dumps(rsd_comparison),
        "all_experiments": all_experiments,
        "n_steps": n_steps,
        "selected_ids": selected_ids,
    })


# ═══════════════════════════════════════════════════════════════
# MATRICE ANALYSIS
# ═══════════════════════════════════════════════════════════════

def matrix_analysis(request):
    """Analyse détaillée des propriétés des matrices P."""
    method = request.GET.get("method")

    qs = TransitionMatrix.objects.select_related(
        "experiment__partition_method"
    )
    if method:
        qs = qs.filter(experiment__partition_method__name=method)

    data = []
    for tm in qs:
        exp = tm.experiment
        data.append({
            "id": exp.id,
            "name": exp.folder_name[:40],
            "method": exp.partition_method.name,
            "n_states": exp.n_states,
            "nlt": exp.nlt,
            "step": exp.step_size,
            
            # Diagonale (P rester)
            "diag_mean": tm.diagonal_mean,
            "diag_std": tm.diagonal_std,
            "diag_min": tm.diagonal_min,
            "diag_max": tm.diagonal_max,
            
            # Sommes des lignes (normalisation)
            "row_sum_min": tm.row_sum_min,
            "row_sum_max": tm.row_sum_max,
            "row_sum_mean": tm.row_sum_mean,
            "row_sum_std": tm.row_sum_std,
            
            # Sommes des colonnes
            "col_sum_min": tm.col_sum_min,
            "col_sum_max": tm.col_sum_max,
            "col_sum_mean": tm.col_sum_mean,
            
            # Visitabilité
            "fraction_visited": tm.fraction_visited,
            
            # Valeurs propres
            "eigenvalue_2": tm.eigenvalue_2,
            "spectral_gap": tm.spectral_gap,
        })

    # Supprimer les doublons par méthode - garder un exemplaire
    seen_methods = set()
    unique_methods = []
    for item in data:
        if item["method"] not in seen_methods:
            unique_methods.append(item["method"])
            seen_methods.add(item["method"])

    available_methods = unique_methods

    return render(request, "markov/matrix_analysis.html", {
        "matrix_data": json.dumps(data),
        "current_method": method,
        "available_methods": available_methods,
        "total_count": len(data),
    })


# ═══════════════════════════════════════════════════════════════
# PARTITION VIEWER (3D)
# ═══════════════════════════════════════════════════════════════

def partition_viewer(request):
    """Page de visualisation 3D des partitions."""
    return render(request, "markov/partition_viewer.html")


# ═══════════════════════════════════════════════════════════════
# ANALYSE DES MÉTRIQUES
# ═══════════════════════════════════════════════════════════════

def metrics_analysis(request):
    """Analyse complète des métriques de toutes les matrices."""
    method = request.GET.get("method")
    
    # Récupérer les expériences avec leurs métriques
    qs = Experiment.objects.select_related("partition_method", "matrix")
    
    # Récupérer les méthodes distinctes
    all_methods = list(PartitionMethod.objects.values_list("name", flat=True).distinct())
    
    if method:
        qs = qs.filter(partition_method__name=method)
    
    metrics_data = []
    for exp in qs:
        try:
            matrix = exp.matrix
            metrics_data.append({
                "id": exp.id,
                "name": exp.folder_name[:50],
                "method": exp.partition_method.name,
                "n_states": exp.n_states,
                "nlt": exp.nlt,
                "step_size": exp.step_size,
                "start_index": exp.start_index,
                
                # P(rester) — Diagonale
                "diag_mean": round(matrix.diagonal_mean, 4) if matrix.diagonal_mean else 0,
                "diag_std": round(matrix.diagonal_std, 4) if matrix.diagonal_std else 0,
                "diag_range": f"[{round(matrix.diagonal_min, 3)}, {round(matrix.diagonal_max, 3)}]" if matrix.diagonal_min is not None else "N/A",
                
                # Normalisation des lignes
                "row_sum_mean": round(matrix.row_sum_mean, 4) if matrix.row_sum_mean else 0,
                "row_sum_std": round(matrix.row_sum_std, 4) if matrix.row_sum_std else 0,
                "row_sum_range": f"[{round(matrix.row_sum_min, 3)}, {round(matrix.row_sum_max, 3)}]" if matrix.row_sum_min is not None else "N/A",
                
                # Visitabilité
                "fraction_visited": f"{round(matrix.fraction_visited * 100, 1)}%" if matrix.fraction_visited else "N/A",
                
                # Valeurs propres (convergence)
                "eigenvalue_2": round(matrix.eigenvalue_2, 4) if matrix.eigenvalue_2 else 0,
                "spectral_gap": round(matrix.spectral_gap, 4) if matrix.spectral_gap else 0,
            })
        except Exception:
            pass
    
    return render(request, "markov/metrics_analysis.html", {
        "metrics_data": json.dumps(metrics_data),
        "available_methods": all_methods,
        "current_method": method,
        "total_count": len(metrics_data),
    })


# ═══════════════════════════════════════════════════════════════
# API JSON
# ═══════════════════════════════════════════════════════════════

def api_matrix_data(request, pk):
    """Données de la matrice pour Plotly."""
    exp = get_object_or_404(Experiment, pk=pk)
    md = _load_matrix_for_experiment(exp)
    if md.get("error"):
        return JsonResponse({"error": md["error"]}, status=500)
    return JsonResponse({k: v for k, v in md.items() if k != "matrix"})


def api_rsd_data(request, pk):
    """Données RSD pour Plotly."""
    exp = get_object_or_404(Experiment, pk=pk)
    n_steps = int(request.GET.get("n_steps", 200))

    md = _load_matrix_for_experiment(exp)
    if md.get("error"):
        return JsonResponse({"error": md["error"]}, status=500)

    rsd = _compute_rsd_from_matrix(md["matrix"], n_steps)
    return JsonResponse(rsd)


def api_experiment_stats(request):
    """Stats agrégées pour les graphiques."""
    method = request.GET.get("method")
    qs = TransitionMatrix.objects.select_related("experiment__partition_method")
    if method:
        qs = qs.filter(experiment__partition_method__name=method)

    data = list(qs.values(
        "experiment__folder_name",
        "experiment__partition_method__name",
        "experiment__nlt",
        "experiment__step_size",
        "experiment__n_states",
        "diagonal_mean",
        "row_sum_mean",
        "eigenvalue_2",
        "fraction_visited",
    ))
    return JsonResponse({"data": data})


def api_compare_rsd(request):
    """Compare le RSD de plusieurs expériences."""
    ids = request.GET.getlist("ids")
    n_steps = int(request.GET.get("n_steps", 200))

    results = []
    for pk in ids:
        try:
            exp = Experiment.objects.select_related("partition_method", "matrix").get(pk=pk)
            md = _load_matrix_for_experiment(exp)
            if md.get("matrix") is not None:
                rsd = _compute_rsd_from_matrix(md["matrix"], n_steps)
                results.append({
                    "name": exp.folder_name[:40],
                    "method": exp.partition_method.name,
                    "n_states": exp.n_states,
                    **rsd,
                })
        except Exception:
            pass

    return JsonResponse({"results": results})


def _load_partition_data(file_index=100, method='cartesian', n_cells=125, sample_every=1):
    HF_FOLDER = "hf://buckets/ktongue/DEM_MCM/Output Paraview"
    fs = HfFileSystem()
    files = sorted(fs.glob(f"{HF_FOLDER}/*.csv"))

    if file_index >= len(files):
        raise ValueError(f"file_index {file_index} hors limites (max: {len(files)-1})")

    fname = files[file_index]
    with fs.open(fname, "rb") as f:
        df = pl.read_csv(f)

    coords = np.column_stack([
        df["coordinates:0"].to_numpy(),
        df["coordinates:1"].to_numpy(),
        df["coordinates:2"].to_numpy(),
    ])[::sample_every]

    partitioner_kwargs = {}
    if method == "cartesian":
        side = int(np.ceil(n_cells ** (1/3)))
        partitioner_kwargs = {"nx": side, "ny": side, "nz": side}
    elif method == "cylindrical":
        n_radial = int(np.ceil((n_cells / 2) ** 0.5))
        partitioner_kwargs = {"nr": n_radial, "ntheta": n_radial, "nz": 2, "radial_mode": "equal_dr"}
    elif method == "voronoi":
        partitioner_kwargs = {"n_cells": n_cells}
    elif method == "quantile":
        side = int(np.ceil(n_cells ** (1/3)))
        partitioner_kwargs = {"nx": side, "ny": side, "nz": side}
    elif method == "octree":
        partitioner_kwargs = {"max_particles": max(10, n_cells // 4), "max_depth": 4}

    partitioner = create_partitioner(method, **partitioner_kwargs)
    partitioner.fit(coords)

    states = partitioner.compute_states(
        df["coordinates:0"].to_numpy()[::sample_every],
        df["coordinates:1"].to_numpy()[::sample_every],
        df["coordinates:2"].to_numpy()[::sample_every]
    )

    return {
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
        "states": states,
        "n_cells": int(partitioner.n_cells),
        "n_particles": len(coords),
        "method": method,
    }


def api_partitions(request):
    """
    API pour charger les positions des particules et les assigner à des cellules.

    Query parameters:
        file_index: index du fichier DEM à charger (défaut: 100)
        method: méthode de partitionnement (cartesian, cylindrical, voronoi, quantile, octree)
        n_cells: nombre de cellules (défaut: 125)
    """
    try:
        file_index = int(request.GET.get("file_index", 100))
        method = request.GET.get("method", "cartesian")
        n_cells = int(request.GET.get("n_cells", 125))
        sample_every = int(request.GET.get("sample_every", 1))

        valid_methods = ["cartesian", "cylindrical", "voronoi", "quantile", "octree"]
        if method not in valid_methods:
            return JsonResponse({
                "error": f"Méthode inconnue: {method}. Valides: {', '.join(valid_methods)}"
            }, status=400)

        payload = _load_partition_data(file_index=file_index, method=method, n_cells=n_cells, sample_every=sample_every)

        # conversions pour JSON
        return JsonResponse({
            "x": payload["x"].tolist(),
            "y": payload["y"].tolist(),
            "z": payload["z"].tolist(),
            "states": payload["states"].tolist(),
            "n_cells": payload["n_cells"],
            "n_particles": payload["n_particles"],
            "method": payload["method"],
        })

    except ImportError as e:
        return JsonResponse({"error": f"Dépendances manquantes: {str(e)}"}, status=500)
    except Exception as e:
        return JsonResponse({"error": str(e), "traceback": traceback.format_exc()}, status=500)


def api_partitions_pyvista(request):
    """API PyVista pour rendu 3D serveur (retourne image base64)."""
    try:
        import base64
        try:
            import pyvista as pv
        except ImportError as e:
            return JsonResponse({"error": f"PyVista non trouvé: {str(e)}"}, status=500)

        # Configuration headless (OSMesa/Xvfb)
        os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')

        file_index = int(request.GET.get("file_index", 100))
        method = request.GET.get("method", "cartesian")
        n_cells = int(request.GET.get("n_cells", 125))
        sample_every = int(request.GET.get("sample_every", 1))

        payload = _load_partition_data(file_index=file_index, method=method, n_cells=n_cells, sample_every=sample_every)

        pts = np.column_stack((payload["x"], payload["y"], payload["z"]))
        states = payload["states"]

        if pts.size == 0:
            return JsonResponse({"error": "Aucun point généré (pas de données)"}, status=400)

        mesh = pv.PolyData(pts)
        mesh["states"] = states

        # Configure l'environnement offscreen pour PyVista
        os.environ.setdefault('PYVISTA_OFF_SCREEN', 'true')
        os.environ.setdefault('PYVISTA_USE_OSMESA', 'true')

        plotter = pv.Plotter(off_screen=True, window_size=(1000, 800))
        plotter.set_background('white')
        plotter.add_points(mesh, scalars='states', cmap='viridis', point_size=3, render_points_as_spheres=True)
        plotter.add_axes()
        plotter.camera_position = 'xy'

        # Ne pas appeler `plotter.show()` en mode headless : peut échouer (Load failed)
        img = plotter.screenshot(transparent_background=False, return_img=True)
        plotter.close()

        from PIL import Image
        import io as _io
        buf = _io.BytesIO()
        Image.fromarray(img).save(buf, format='PNG')
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')

        return JsonResponse({
            "image": f"data:image/png;base64,{encoded}",
            "method": method,
            "n_points": int(pts.shape[0]),
            "n_cells": int(payload["n_cells"]),
        })

    except Exception as e:
        return JsonResponse({"error": str(e), "traceback": traceback.format_exc()}, status=500)