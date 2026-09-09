from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture


@dataclass(slots=True)
class ClusteringFit:
    """Container for a fitted clustering method and its evaluation metrics."""

    method: str
    model: Any
    labels: np.ndarray
    metrics: dict[str, float]
    parameters: dict[str, Any]


def _safe_cluster_scores(matrix: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    valid = labels != -1
    filtered_labels = labels[valid]
    filtered_matrix = matrix[valid]
    unique_labels = np.unique(filtered_labels)
    if filtered_matrix.shape[0] < 3 or unique_labels.size < 2:
        return {"silhouette": float("nan"), "davies_bouldin": float("nan"), "calinski_harabasz": float("nan")}
    return {
        "silhouette": float(silhouette_score(filtered_matrix, filtered_labels)),
        "davies_bouldin": float(davies_bouldin_score(filtered_matrix, filtered_labels)),
        "calinski_harabasz": float(calinski_harabasz_score(filtered_matrix, filtered_labels)),
    }


def _refit_result(method: str, matrix: np.ndarray, parameters: dict[str, Any], random_state: int = 42) -> np.ndarray:
    if method == "kmeans":
        return KMeans(n_clusters=parameters["n_clusters"], n_init="auto", random_state=random_state).fit_predict(matrix)
    if method == "gmm":
        return GaussianMixture(n_components=parameters["n_components"], covariance_type="full", random_state=random_state).fit_predict(matrix)
    if method == "hierarchical":
        return AgglomerativeClustering(n_clusters=parameters["n_clusters"], linkage="ward").fit_predict(matrix)
    if method == "hdbscan":
        import hdbscan

        return hdbscan.HDBSCAN(min_cluster_size=parameters["min_cluster_size"]).fit_predict(matrix)
    raise ValueError(f"Unknown clustering method: {method}")


def _bootstrap_stability(
    matrix: np.ndarray,
    method: str,
    parameters: dict[str, Any],
    reference_labels: np.ndarray,
    iterations: int = 10,
    sample_fraction: float = 0.8,
    random_state: int = 42,
) -> float:
    rng = np.random.default_rng(random_state)
    scores: list[float] = []
    sample_size = max(3, int(round(matrix.shape[0] * sample_fraction)))
    for _ in range(iterations):
        sample_idx = rng.choice(matrix.shape[0], size=sample_size, replace=False)
        sample_matrix = matrix[sample_idx]
        sample_labels = _refit_result(method, sample_matrix, parameters, random_state=random_state)
        scores.append(float(adjusted_rand_score(reference_labels[sample_idx], sample_labels)))
    return float(np.mean(scores)) if scores else float("nan")


def fit_kmeans(matrix: np.ndarray, n_clusters: int, random_state: int = 42) -> ClusteringFit:
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    metrics["inertia"] = float(model.inertia_)
    return ClusteringFit(method="kmeans", model=model, labels=labels, metrics=metrics, parameters={"n_clusters": n_clusters})


def fit_gmm(matrix: np.ndarray, n_components: int, random_state: int = 42) -> ClusteringFit:
    model = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    metrics["bic"] = float(model.bic(matrix))
    metrics["aic"] = float(model.aic(matrix))
    return ClusteringFit(method="gmm", model=model, labels=labels, metrics=metrics, parameters={"n_components": n_components})


def fit_hierarchical(matrix: np.ndarray, n_clusters: int) -> ClusteringFit:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    return ClusteringFit(method="hierarchical", model=model, labels=labels, metrics=metrics, parameters={"n_clusters": n_clusters})


def fit_hdbscan(matrix: np.ndarray, min_cluster_size: int) -> ClusteringFit:
    try:
        import hdbscan
    except Exception as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("hdbscan is required for density-based clustering") from exc

    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = model.fit_predict(matrix)
    metrics = _safe_cluster_scores(matrix, labels)
    metrics["noise_fraction"] = float((labels == -1).mean())
    return ClusteringFit(method="hdbscan", model=model, labels=labels, metrics=metrics, parameters={"min_cluster_size": min_cluster_size})


def evaluate_k_family(
    matrix: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
    stability_iterations: int = 10,
    stability_sample_fraction: float = 0.8,
) -> pd.DataFrame:
    """Compare multiple cluster counts for KMeans, GMM and hierarchical clustering."""

    rows: list[dict[str, Any]] = []
    if matrix.shape[0] < 3:
        return pd.DataFrame(rows)
    upper_k = min(k_max, matrix.shape[0] - 1)
    for k in range(k_min, upper_k + 1):
        for result in [fit_kmeans(matrix, k, random_state=random_state), fit_gmm(matrix, k, random_state=random_state), fit_hierarchical(matrix, k)]:
            stability = _bootstrap_stability(
                matrix,
                result.method,
                result.parameters,
                result.labels,
                iterations=stability_iterations,
                sample_fraction=stability_sample_fraction,
                random_state=random_state,
            )
            rows.append({"method": result.method, "k": k, **result.metrics, "stability": stability})
    return pd.DataFrame(rows)


def evaluate_hdbscan_family(
    matrix: np.ndarray,
    min_cluster_sizes: list[int],
    random_state: int = 42,
    stability_iterations: int = 10,
    stability_sample_fraction: float = 0.8,
) -> pd.DataFrame:
    """Compare several min_cluster_size values for HDBSCAN."""

    rows: list[dict[str, Any]] = []
    if matrix.shape[0] < 3:
        return pd.DataFrame(rows)
    for size in min_cluster_sizes:
        result = fit_hdbscan(matrix, size)
        stability = _bootstrap_stability(
            matrix,
            result.method,
            result.parameters,
            result.labels,
            iterations=stability_iterations,
            sample_fraction=stability_sample_fraction,
            random_state=random_state,
        )
        rows.append({"method": result.method, "k": size, **result.metrics, "stability": stability})
    return pd.DataFrame(rows)


def choose_best_cluster_row(
    metrics: pd.DataFrame,
    preferred_k_min: int = 5,
    k_diversity_bonus: float = 0.06,
    low_k_penalty: float = 0.08,
) -> pd.Series:
    """Select the strongest configuration while preferring simpler solutions when scores are similar."""

    if metrics.empty:
        raise ValueError("No clustering metrics available")
    scored = metrics.copy()
    k_values = pd.to_numeric(scored["k"], errors="coerce").fillna(float(preferred_k_min))
    k_min = float(k_values.min())
    k_max = float(k_values.max())
    k_range = max(k_max - k_min, 1.0)
    normalized_k = (k_values - k_min) / k_range

    scored["score"] = (
        scored["silhouette"].fillna(-1.0)
        - scored["davies_bouldin"].fillna(999.0) / 10.0
        + scored["stability"].fillna(0.0)
        + (normalized_k * k_diversity_bonus)
        - np.maximum(0, preferred_k_min - k_values) * low_k_penalty
    )

    # Penalize larger k values slightly so the selector prefers simpler solutions
    # when the clustering quality is essentially tied.
    scored["score"] = scored["score"] - (normalized_k * 0.08)

    scored["_k_numeric"] = pd.to_numeric(scored["k"], errors="coerce").fillna(float(preferred_k_min))
    best_score = float(scored["score"].max())
    near_best = scored["score"] >= (best_score - 0.03)
    if near_best.any():
        candidate_rows = scored.loc[near_best].sort_values(["_k_numeric", "score"], ascending=[True, False])
        best_idx = candidate_rows.index[0]
    else:
        best_idx = scored["score"].idxmax()
    return scored.loc[best_idx]


def attach_cluster_labels(frame: pd.DataFrame, labels: np.ndarray, label_column: str = "cluster") -> pd.DataFrame:
    """Return a copy of the frame with cluster labels attached."""

    enriched = frame.copy()
    enriched[label_column] = labels
    return enriched


def merge_similar_clusters(
    frame: pd.DataFrame,
    feature_columns: list[str],
    similarity_threshold: float = 0.92,
    label_column: str = "cluster",
) -> pd.DataFrame:
    """Merge clusters whose centroids are highly similar in feature space."""

    if frame.empty or label_column not in frame.columns:
        return frame.copy()

    valid_features = [column for column in feature_columns if column in frame.columns]
    if not valid_features:
        return frame.copy()

    result = frame.copy()
    cluster_ids = sorted(pd.unique(pd.to_numeric(result[label_column], errors="coerce").dropna().astype(int)))
    if len(cluster_ids) <= 1:
        return result

    centroids = result.groupby(label_column, dropna=False)[valid_features].mean(numeric_only=True).loc[cluster_ids]
    centroid_matrix = np.asarray(centroids.to_numpy(), dtype=float)
    similarity_matrix = cosine_similarity(centroid_matrix)
    cluster_index = {cluster_id: idx for idx, cluster_id in enumerate(cluster_ids)}

    merged_groups: list[list[int]] = []
    for cluster_id in cluster_ids:
        matched_group: list[int] | None = None
        for group in merged_groups:
            if cluster_id in group:
                matched_group = group
                break
            # Require the candidate to be similar to EVERY member already in the
            # group (single-linkage on the whole group), not just the first
            # element. Comparing only against a fixed "representative" allowed
            # transitive chains (A~B, B~C) to merge clusters A and C that are
            # not actually similar to each other, which corrupted the resulting
            # centroid and therefore the inferred style label.
            similarities = [
                float(similarity_matrix[cluster_index[member], cluster_index[cluster_id]])
                for member in group
            ]
            if min(similarities) >= similarity_threshold:
                matched_group = group
                break
        if matched_group is None:
            merged_groups.append([cluster_id])
        else:
            matched_group.append(cluster_id)

    remapped_labels: dict[int, int] = {}
    for new_id, group in enumerate(merged_groups):
        for cluster_id in group:
            remapped_labels[int(cluster_id)] = new_id

    result[label_column] = pd.to_numeric(result[label_column], errors="coerce").astype("Int64")
    result[label_column] = result[label_column].map(remapped_labels).astype("Int64")

    # style_label is intentionally NOT computed here anymore. It's derived once,
    # downstream, from build_cluster_profile (which uses the same centroid-based
    # infer_style_label logic) and merged back onto the frame in pipeline.py.
    # Computing it here too used to leave a stale/duplicate "style_label" column
    # on `result`, which caused pandas to auto-suffix both copies (style_label_x
    # / style_label_y) during the later merge in pipeline.py instead of cleanly
    # overwriting it — so the "final" label users inspected wasn't reliably the
    # one computed on the post-merge clusters.
    return result


def _format_feature_name(feature: str) -> str:
    """Turn a snake_case feature column name into a short, readable label."""
    cleaned = feature.replace("_pct", "").replace("_rate", "").replace("_", " ")
    return cleaned.strip().title() or feature


def generate_data_driven_style_label(
    z_scores: pd.Series,
    top_n: int = 2,
    threshold: float = 0.15,
) -> str:
    """Build a human-readable cluster name straight from the features that most
    distinguish this cluster from the overall player population (z_scores =
    cluster centroid - overall mean, already computed by the caller).

    This replaces infer_style_label's fixed set of hand-weighted archetype
    formulas. Instead of deciding upfront which combination of features means
    "Serve Bot" vs "Counterpuncher" and hand-tuning weights until it looks
    right, the label is derived directly from whatever the clustering itself
    found to be most different about this group -- so it can surface
    archetypes (or combinations) that were never explicitly anticipated, and
    it can't silently misfire the way overlapping hand-built scores did.

    `threshold` is on whatever scale the input features are on (z-scored
    features from preprocessing will typically need a smaller threshold than
    raw percentage features) -- tune it against your own data rather than
    trusting the default blindly.
    """
    ranked = z_scores.reindex(z_scores.abs().sort_values(ascending=False).index)
    positives = [f for f in ranked.index if ranked[f] > threshold][:top_n]
    negatives = [f for f in ranked.index if ranked[f] < -threshold][:top_n]

    if not positives and not negatives:
        return "Balanced / Average Profile"

    parts = []
    if positives:
        parts.append("High " + " & ".join(_format_feature_name(f) for f in positives))
    if negatives:
        parts.append("Low " + " & ".join(_format_feature_name(f) for f in negatives))
    return ", ".join(parts)


# Tunable thresholds for infer_style_label below. All features arriving here
# are already standardized (z-scored within season, then globally rescaled by
# preprocessing.py), so these thresholds are in standard-deviation units, not
# raw percentages -- retune against your own data if cluster boundaries shift.
WEAK_PROFILE_THRESHOLD = -0.4  # below this, a cluster is a competence tier, not a style
SERVE_STRENGTH_THRESHOLD = 0.15  # service_aggression must clear this to be eligible for "Serve Bot"
RETURN_STRENGTH_THRESHOLD = 0.15  # return_aggression must clear this to be eligible for "Counterpuncher"
BALANCE_THRESHOLD = 0.1  # serve_return_balance must clear this (in the relevant direction) too
SURFACE_ADVANTAGE_THRESHOLD = 0.15  # a surface must beat the average of the OTHER present surfaces by this much
ALL_COURT_MAX_SPREAD = 0.3  # max allowed gap between best and worst present surface for "All-Court"
ALL_COURT_MIN_AVG = 0.1  # average across present surfaces must clear this -- rules out "uniformly mediocre"


# Kept for reference / as an optional alternative labeling strategy. Not
# called by build_cluster_profile by default -- infer_style_label (the fixed
# archetype taxonomy, below) is used instead.
def style_label_scores(centroid: pd.Series) -> dict[str, float]:
    """Return every archetype's raw score for this cluster centroid (only the
    archetypes that pass their eligibility gates are included -- see
    infer_style_label for what each gate checks). Useful for diagnosing why a
    particular archetype never wins: check whether it's missing from this
    dict entirely (didn't clear its eligibility gate) or present with a score
    close to the winner (narrowly lost, worth retuning thresholds for)."""

    service_aggression = float(centroid.get("service_aggression_score", 0))
    return_aggression = float(centroid.get("return_aggression_score", 0))
    tiebreak_frequency = float(centroid.get("tiebreak_frequency", 0))
    ace_rate = float(centroid.get("ace_rate", 0))
    service_hold_rate = float(centroid.get("service_hold_rate", 0))
    first_serve_win_pct = float(centroid.get("first_serve_win_pct", 0))
    return_points_won_pct = float(centroid.get("return_points_won_pct", 0))
    return_games_won_pct = float(centroid.get("return_games_won_pct", 0))
    # average_match_length arrives here already standardized (see module note
    # above) -- NOT raw minutes. A previous version of this function divided
    # it by 120 assuming raw minutes (~60-200), which -- given the actual
    # z-scored values (~-2..+2) -- crushed it to near zero and silently
    # removed it from every formula that used it. Use it directly.
    average_match_length = float(centroid.get("average_match_length", 0))
    # Properly-normalized serve/return imbalance: this is z-scored as its OWN
    # feature by season_zscore, against the population's typical serve/return
    # gap -- unlike (service_aggression - return_aggression), it isn't biased
    # by the fact that service_aggression and return_aggression are each
    # independently z-scored and needn't share a comparable scale or
    # distribution shape. Falls back to the raw difference only if the
    # pipeline hasn't been rerun yet with this feature included.
    serve_return_balance = float(centroid.get("serve_return_balance", service_aggression - return_aggression))

    overall_aggression = (service_aggression + return_aggression) / 2.0

    # A cluster that's weak on BOTH service and return isn't a playing STYLE
    # -- it's a competence tier (players losing most of their matches).
    # Forcing one of the six archetype labels onto it used to mean whichever
    # formula subtracted the *more* negative of two bad numbers won by having
    # its sign flipped positive -- e.g. a cluster with weak service AND
    # catastrophic return got called "Serve Bot" because its service was
    # merely less bad, not because it was actually strong.
    if overall_aggression < WEAK_PROFILE_THRESHOLD:
        return {}

    # Only surfaces the centroid actually has data for. hard_win_rate is
    # assumed always present; clay/grass may be entirely missing from the
    # centroid (dropped upstream by drop_high_missing_columns -- grass in
    # particular has a very short ATP season and often fails the missing
    # threshold). Defaulting a missing surface to 0 would silently treat "no
    # data" as "average performance in z-score terms," inflating the
    # calculated advantage of whichever surface IS present when compared
    # against a partly-fabricated baseline -- so missing surfaces are left
    # out of the comparison entirely instead.
    present_surfaces: dict[str, float] = {}
    if "hard_win_rate" in centroid.index:
        present_surfaces["hard"] = float(centroid["hard_win_rate"])
    if "clay_win_rate" in centroid.index:
        present_surfaces["clay"] = float(centroid["clay_win_rate"])
    if "grass_win_rate" in centroid.index:
        present_surfaces["grass"] = float(centroid["grass_win_rate"])

    def _surface_advantage(name: str) -> float | None:
        if name not in present_surfaces:
            return None
        others = [v for key, v in present_surfaces.items() if key != name]
        if not others:
            return None
        return present_surfaces[name] - (sum(others) / len(others))

    scores: dict[str, float] = {}

    clay_advantage = _surface_advantage("clay")
    if clay_advantage is not None and clay_advantage > SURFACE_ADVANTAGE_THRESHOLD:
        scores["Clay Specialist"] = clay_advantage

    hard_advantage = _surface_advantage("hard")
    if hard_advantage is not None and hard_advantage > SURFACE_ADVANTAGE_THRESHOLD:
        scores["Hard Specialist"] = hard_advantage

    # Requires genuine balance (small spread across whichever surfaces are
    # present) AND a competent absolute level -- not just "equally mediocre
    # everywhere," which the old sum-only formula couldn't distinguish from
    # true versatility.
    if len(present_surfaces) >= 2:
        surface_values = list(present_surfaces.values())
        surface_avg = sum(surface_values) / len(surface_values)
        surface_spread = max(surface_values) - min(surface_values)
        if surface_spread < ALL_COURT_MAX_SPREAD and surface_avg > ALL_COURT_MIN_AVG:
            scores["All-Court"] = surface_avg - surface_spread

    # Only eligible if service_aggression is genuinely positive in absolute
    # terms (not just bigger than a possibly-also-negative return_aggression)
    # AND serve_return_balance shows genuine serve-skew relative to the wider
    # player population, not just relative to this cluster's own return side.
    if service_aggression > SERVE_STRENGTH_THRESHOLD and serve_return_balance > BALANCE_THRESHOLD:
        scores["Serve Bot"] = float(
            ace_rate
            + service_hold_rate
            + first_serve_win_pct
            - return_points_won_pct
            + 0.5 * serve_return_balance
            + 0.25 * tiebreak_frequency
        )

    # Mirrored guard: return_aggression must clear its own floor AND
    # serve_return_balance must show genuine return-skew.
    if return_aggression > RETURN_STRENGTH_THRESHOLD and serve_return_balance < -BALANCE_THRESHOLD:
        scores["Counterpuncher"] = float(
            return_points_won_pct
            + return_games_won_pct
            - 0.5 * serve_return_balance
        )

    # Requires an actual grinding signal (tiebreaks and/or longer matches),
    # not just "both aggression scores happen to be very negative" flipping
    # the subtraction term positive.
    if tiebreak_frequency > 0 or average_match_length > 0:
        scores["Defensive Player"] = float(
            tiebreak_frequency
            + average_match_length
            - overall_aggression
        )

    # Requires BOTH sides to clear zero -- min() alone doesn't prevent a
    # cluster with two mildly-negative aggression scores from still "winning"
    # this label relative to worse alternatives.
    if service_aggression > 0 and return_aggression > 0:
        scores["Short-Rally Attacker"] = float(
            min(service_aggression, return_aggression)
            - 0.25 * tiebreak_frequency
        )

    return scores


def infer_style_label(centroid: pd.Series) -> str:
    """Assign an interpretable label to a cluster profile using heuristic
    rules. Thin wrapper around style_label_scores -- see that function to
    inspect runner-up archetypes instead of just the winner."""

    scores = style_label_scores(centroid)
    if not scores:
        return "Developing / Limited Output"
    return max(scores, key=scores.get)


def build_cluster_profile(frame: pd.DataFrame, feature_columns: list[str], label_column: str = "cluster") -> pd.DataFrame:
    """Summarize cluster centroids, dominant features and representative players."""

    rows: list[dict[str, Any]] = []
    overall_means = frame[feature_columns].mean(numeric_only=True)
    for cluster_id, subset in frame.groupby(label_column):
        centroid = subset[feature_columns].mean(numeric_only=True)
        z_scores = centroid - overall_means
        top_positive = z_scores.sort_values(ascending=False).head(5)
        top_negative = z_scores.sort_values(ascending=True).head(5)
        if feature_columns:
            distances = np.sqrt(((subset[feature_columns] - centroid) ** 2).sum(axis=1))
            representative_players = (
                subset.assign(_distance=distances)
                .sort_values(["_distance", "player_name", "season"], ascending=[True, True, True])
                [["player_name", "season"]]
                .head(5)
                .to_dict(orient="records")
            )
        else:
            representative_players = subset[["player_name", "season"]].head(5).to_dict(orient="records")
        cluster_style_scores = style_label_scores(centroid)
        cluster_style_label = max(cluster_style_scores, key=cluster_style_scores.get) if cluster_style_scores else "Developing / Limited Output"
        rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(subset)),
                "top_positive_features": ", ".join(top_positive.index.tolist()),
                "top_negative_features": ", ".join(top_negative.index.tolist()),
                "mean_features": centroid.to_dict(),
                "representatives": representative_players,
                "style_label": cluster_style_label,
                "style_scores": {
                    name: round(score, 4)
                    for name, score in sorted(cluster_style_scores.items(), key=lambda item: item[1], reverse=True)
                },
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)