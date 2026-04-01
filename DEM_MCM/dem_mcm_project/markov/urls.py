from django.urls import path
from . import views

app_name = "markov"

urlpatterns = [
    # Pages
    path("", views.dashboard, name="dashboard"),
    path("analysis/", views.unified_analysis, name="unified_analysis"),
    path("experiments/", views.experiment_list, name="experiment_list"),
    path("experiments/<int:pk>/", views.experiment_detail, name="experiment_detail"),
    path("compare/", views.compare_view, name="compare"),
    path("compare/sweep/", views.compare_sweep, name="compare_sweep"),
    path("compare/dem-markov/", views.compare_dem_markov, name="compare_dem_markov"),
    path("rsd/", views.rsd_analysis, name="rsd_analysis"),
    path("matrix/", views.matrix_analysis, name="matrix_analysis"),
    path("metrics/", views.metrics_analysis, name="metrics_analysis"),
    path("partitions/", views.partition_viewer, name="partition_viewer"),
    path("api/partitions-pyvista/", views.api_partitions_pyvista, name="api_partitions_pyvista"),

    # API
    path("api/matrix/<int:pk>/", views.api_matrix_data, name="api_matrix"),
    path("api/rsd/<int:pk>/", views.api_rsd_data, name="api_rsd"),
    path("api/stats/", views.api_experiment_stats, name="api_stats"),
    path("api/compare-rsd/", views.api_compare_rsd, name="api_compare_rsd"),
    path("api/partitions/", views.api_partitions, name="api_partitions"),
]