"""
Commande Django pour calculer et sauvegarder les résultats RSD DEM et Markov.

Lit les snapshots DEM depuis le bucket HuggingFace, calcule le RSD réel
pour chaque méthode de partitionnement, et sauvegarde les résultats
dans la table RSDResult.

Usage:
    python manage.py compute_dem_rsd
    python manage.py compute_dem_rsd --method voronoi
    python manage.py compute_dem_rsd --criterion diameter
    python manage.py compute_dem_rsd --max-experiments 10
    python manage.py compute_dem_rsd --force
"""

import sys
import numpy as np
from pathlib import Path
from django.core.management.base import BaseCommand

# Add core/ directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "core"))

from markov.models import PartitionMethod, Experiment, TransitionMatrix, RSDResult
from core.analyze_results import MarkovAnalyzer
from core.partitioners import create_partitioner
from core.bucket_io import load_matrix_from_bucket


def _to_list(arr):
    """Convert numpy array to JSON-serializable list of floats, cleaning NaN/Inf."""
    if isinstance(arr, np.ndarray):
        cleaned = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        return [float(x) for x in cleaned]
    if isinstance(arr, list):
        cleaned = [
            0.0 if (x != x) or (x == float("inf")) or (x == float("-inf")) else float(x)
            for x in arr
        ]
        return cleaned
    return arr


class Command(BaseCommand):
    help = "Calcule et sauvegarde les résultats RSD DEM et Markov"

    def add_arguments(self, parser):
        parser.add_argument(
            "--method",
            type=str,
            default=None,
            help="Filtrer par méthode (voronoi, cartesian, cylindrical, ...)",
        )
        parser.add_argument(
            "--criterion",
            type=str,
            default="z_median",
            help="Critère d'étiquetage des espèces: z_median, diameter, x_median, ...",
        )
        parser.add_argument(
            "--max-experiments",
            type=int,
            default=None,
            help="Limiter le nombre d'expériences à traiter",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Recalculer même si les résultats existent déjà",
        )
        parser.add_argument(
            "--n-steps",
            type=int,
            default=200,
            help="Nombre de pas pour la simulation Markov",
        )
        parser.add_argument(
            "--sample-every",
            type=int,
            default=10,
            help="Sous-échantillonnage spatial des particules",
        )
        parser.add_argument(
            "--max-snapshots",
            type=int,
            default=50,
            help="Nombre maximum de snapshots DEM à charger",
        )

    def handle(self, *args, **options):
        method_filter = options["method"]
        criterion = options["criterion"]
        max_experiments = options["max_experiments"]
        force = options["force"]
        n_steps = options["n_steps"]
        sample_every = options["sample_every"]
        max_snapshots = options["max_snapshots"]

        self.stdout.write(
            self.style.SUCCESS(
                f"🔄 Calcul RSD DEM | critère={criterion} | "
                f"méthode={method_filter or 'toutes'}"
            )
        )

        # ── 1. Charger les snapshots DEM ──
        analyzer = MarkovAnalyzer()
        n_files = min(max_snapshots * sample_every, 500)
        file_indices = list(range(0, n_files, sample_every))[:max_snapshots]

        try:
            analyzer.load_dem_snapshots(file_indices=file_indices, sample_every=1)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Chargement DEM échoué: {e}"))
            return

        # ── 2. Étiqueter les espèces ──
        try:
            analyzer.label_species(criterion=criterion)
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"❌ Étiquetage échoué: {e}"))
            return

        # ── 3. Récupérer les expériences avec matrices ──
        qs = (
            Experiment.objects.select_related("partition_method", "matrix")
            .filter(matrix__isnull=False)
            .distinct()
        )

        if method_filter:
            qs = qs.filter(partition_method__name=method_filter)

        if max_experiments:
            qs = qs[:max_experiments]

        experiments = list(qs)
        self.stdout.write(f"📋 {len(experiments)} expériences à traiter")

        # ── 4. Traiter chaque expérience ──
        total_saved = 0
        total_skipped = 0
        total_errors = 0

        for exp in experiments:
            try:
                self.stdout.write(f"\n   🔬 {exp.folder_name[:60]}")

                # Vérifier si déjà calculé
                dem_exists = RSDResult.objects.filter(
                    experiment=exp, source="dem", species_criterion=criterion
                ).exists()
                markov_exists = RSDResult.objects.filter(
                    experiment=exp, source="markov", species_criterion=criterion
                ).exists()

                if dem_exists and markov_exists and not force:
                    self.stdout.write(f"   ⏭️  Déjà calculé (criterion={criterion})")
                    total_skipped += 1
                    continue

                # Créer et fit le partitionneur
                pm = exp.partition_method
                method = pm.name
                method_kwargs = pm.parameters or {}

                # Utiliser les paramètres du partitionneur tels quels
                # (stockés correctement par sync_bucket)
                # Ne pas forcer n_cells : seuls voronoi/physics l'acceptent
                # Les autres utilisent nx/ny/nz, nr/ntheta/nz, max_particles, etc.
                part = create_partitioner(method, **method_kwargs)

                # Fit sur les données DEM agrégées
                all_coords = np.vstack([s["coords"] for s in analyzer.dem_snapshots])
                part.fit(all_coords)

                self.stdout.write(f"   🔧 {method}: {part.n_cells} cellules")

                # ── Calculer RSD DEM ──
                dem_result = analyzer.compute_dem_rsd(part, analyzer.species_labels)

                # ── Charger la matrice P et calculer RSD Markov ──
                matrix = exp.matrix
                if matrix and matrix.matrix_bucket_path:
                    P = load_matrix_from_bucket(matrix.matrix_bucket_path)
                    P = np.nan_to_num(P, nan=0)

                    # S'assurer que la taille de P correspond au partitionneur
                    if P.shape[0] != part.n_cells:
                        self.stdout.write(
                            f"   ⚠️  Taille P ({P.shape[0]}) != n_cells ({part.n_cells}), ignoré"
                        )
                        total_skipped += 1
                        continue

                    markov_result = analyzer.compute_markov_rsd_from_dem(
                        P, part, analyzer.species_labels
                    )

                    # ── Sauvegarder RSD DEM ──
                    if force:
                        RSDResult.objects.filter(
                            experiment=exp, source="dem", species_criterion=criterion
                        ).delete()

                    RSDResult.objects.create(
                        experiment=exp,
                        source="dem",
                        species_criterion=criterion,
                        n_steps=len(dem_result["times"]),
                        rsd_initial=dem_result["rsd_initial"] * 100,
                        rsd_final=dem_result["rsd_final"] * 100,
                        mixing_time_50=dem_result["mixing_time_50"],
                        mixing_time_90=dem_result["mixing_time_90"],
                        entropy_final=dem_result["entropy"][-1]
                        if len(dem_result["entropy"]) > 0
                        else 0,
                        rsd_curve=_to_list(dem_result["rsd_percent"]),
                        entropy_curve=_to_list(dem_result["entropy"]),
                        concentration_final=_to_list(dem_result["concentrations"][-1])
                        if dem_result["concentrations"]
                        else [],
                    )

                    # ── Sauvegarder RSD Markov ──
                    if force:
                        RSDResult.objects.filter(
                            experiment=exp, source="markov", species_criterion=criterion
                        ).delete()

                    RSDResult.objects.create(
                        experiment=exp,
                        source="markov",
                        species_criterion=criterion,
                        n_steps=len(markov_result["times"]),
                        rsd_initial=markov_result["rsd_initial"] * 100,
                        rsd_final=markov_result["rsd_final"] * 100,
                        mixing_time_50=markov_result["mixing_time_50"],
                        mixing_time_90=markov_result["mixing_time_90"],
                        entropy_final=markov_result["entropy"][-1]
                        if len(markov_result["entropy"]) > 0
                        else 0,
                        rsd_curve=_to_list(markov_result["rsd_percent"]),
                        entropy_curve=_to_list(markov_result["entropy"]),
                        concentration_final=_to_list(
                            markov_result["concentrations"][-1]
                        )
                        if markov_result["concentrations"]
                        else [],
                    )

                    total_saved += 2
                    self.stdout.write(
                        f"   ✅ RSD DEM: {dem_result['rsd_initial'] * 100:.1f}% → "
                        f"{dem_result['rsd_final'] * 100:.1f}% | "
                        f"RSD Markov: {markov_result['rsd_initial'] * 100:.1f}% → "
                        f"{markov_result['rsd_final'] * 100:.1f}%"
                    )
                else:
                    # Pas de matrice, sauvegarder seulement DEM
                    if force:
                        RSDResult.objects.filter(
                            experiment=exp, source="dem", species_criterion=criterion
                        ).delete()

                    RSDResult.objects.create(
                        experiment=exp,
                        source="dem",
                        species_criterion=criterion,
                        n_steps=len(dem_result["times"]),
                        rsd_initial=dem_result["rsd_initial"] * 100,
                        rsd_final=dem_result["rsd_final"] * 100,
                        mixing_time_50=dem_result["mixing_time_50"],
                        mixing_time_90=dem_result["mixing_time_90"],
                        entropy_final=dem_result["entropy"][-1]
                        if len(dem_result["entropy"]) > 0
                        else 0,
                        rsd_curve=_to_list(dem_result["rsd_percent"]),
                        entropy_curve=_to_list(dem_result["entropy"]),
                        concentration_final=_to_list(dem_result["concentrations"][-1])
                        if dem_result["concentrations"]
                        else [],
                    )

                    total_saved += 1
                    self.stdout.write(
                        f"   ✅ RSD DEM seul: {dem_result['rsd_initial'] * 100:.1f}% → "
                        f"{dem_result['rsd_final'] * 100:.1f}% (pas de matrice P)"
                    )

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"   ❌ Erreur: {e}"))
                total_errors += 1

        self.stdout.write(
            self.style.SUCCESS(
                f"\n✅ Terminé | Sauvegardés: {total_saved} | "
                f"Ignorés: {total_skipped} | Erreurs: {total_errors}"
            )
        )
