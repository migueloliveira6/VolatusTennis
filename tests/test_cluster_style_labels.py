import pandas as pd

from clustering.style_clustering.clustering import choose_best_cluster_row, merge_similar_clusters


def test_merge_similar_clusters_reduces_near_duplicate_profiles() -> None:
    frame = pd.DataFrame(
        {
            "player_name": ["A", "B", "C", "D"],
            "season": [2020, 2021, 2022, 2023],
            "cluster": [0, 0, 1, 1],
            "ace_rate": [0.80, 0.79, 0.81, 0.80],
            "service_hold_rate": [0.75, 0.74, 0.76, 0.75],
            "return_points_won_pct": [0.20, 0.21, 0.19, 0.20],
        }
    )

    merged = merge_similar_clusters(frame, feature_columns=["ace_rate", "service_hold_rate", "return_points_won_pct"])

    # Os labels de estilo são atribuídos depois, em build_cluster_profile; aqui só se testa o merge.
    assert merged["cluster"].nunique() == 1


def test_choose_best_cluster_row_prefers_lower_k_when_scores_are_similar() -> None:
    metrics = pd.DataFrame(
        [
            {"method": "kmeans", "k": 4, "silhouette": 0.30, "davies_bouldin": 1.0, "stability": 0.80},
            {"method": "kmeans", "k": 6, "silhouette": 0.30, "davies_bouldin": 1.0, "stability": 0.80},
        ]
    )

    best_row = choose_best_cluster_row(metrics, preferred_k_min=4)

    assert int(best_row["k"]) == 4
