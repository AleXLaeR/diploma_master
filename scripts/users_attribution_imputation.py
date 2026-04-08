"""
users_attribution_imputation.py
===============================
Fold-aware implementation of the `users_attribution` density-floor imputation.

Spec reference:
    docs/data/initial/initial_dataset_augmentation.md §2
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from google.cloud import bigquery

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldBoundary:
    """ROCV fold boundary used to enforce train-only imputation."""

    fold_id: str
    train_end: pd.Timestamp


def _to_date(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()


def _allocate_additions(
    media_counts: pd.Series,
    additions_total: int,
) -> dict[str, int]:
    """Allocate country-level synthetic volume across eligible media sources."""
    if additions_total <= 0:
        return {}

    total_count = int(media_counts.sum())
    if total_count <= 0:
        return {}

    allocations: dict[str, int] = {}
    fractions: list[tuple[str, float]] = []
    assigned = 0

    for media_source, count in media_counts.items():
        exact = additions_total * (float(count) / float(total_count))
        base = int(np.floor(exact))
        allocations[media_source] = base
        fractions.append((media_source, exact - base))
        assigned += base

    remainder = additions_total - assigned
    for media_source, _ in sorted(fractions, key=lambda x: (-x[1], x[0]))[:remainder]:
        allocations[media_source] += 1

    return allocations


def _build_fold_imputation(
    raw_df: pd.DataFrame,
    fold: FoldBoundary,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build imputed rows for a single fold with strict anti-leakage filtering."""
    fold_df = raw_df[raw_df["created_at"].dt.date < fold.train_end.date()].copy()
    if fold_df.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "user_id",
                "created_at",
                "country_code",
                "media_source",
                "is_synthetic",
            ]
        )

    fold_df["fold_id"] = fold.fold_id
    fold_df["is_synthetic"] = False
    fold_df = fold_df[
        [
            "fold_id",
            "user_id",
            "created_at",
            "country_code",
            "media_source",
            "is_synthetic",
            "macro_region",
        ]
    ]

    # Region-level country density baseline.
    country_counts = (
        fold_df.groupby(["macro_region", "country_code"], as_index=False)
        .size()
        .rename(columns={"size": "country_count"})
    )
    region_medians = (
        country_counts.groupby("macro_region", as_index=False)["country_count"]
        .median()
        .rename(columns={"country_count": "region_median"})
    )
    country_counts = country_counts.merge(region_medians, on="macro_region", how="left")

    synthetic_rows: list[dict[str, object]] = []
    synthetic_counter = 0

    for row in country_counts.sort_values(["macro_region", "country_code"]).itertuples():
        region = str(row.macro_region)
        country = str(row.country_code)
        observed_count = int(row.country_count)
        region_median = float(row.region_median)

        # Candidate rule: country below 50% of region median.
        if region_median <= 0 or observed_count >= 0.5 * region_median:
            continue

        # Volume cap: total country attributions may not exceed the region median.
        capped_target = int(np.floor(region_median))
        additions_total = max(capped_target - observed_count, 0)
        if additions_total <= 0:
            continue

        country_rows = fold_df[fold_df["country_code"] == country]
        media_counts = (
            country_rows.groupby("media_source", as_index=True)
            .size()
            .sort_index()
        )
        # Eligibility rule: only media sources already observed in the country.
        media_counts = media_counts[media_counts > 0]
        if media_counts.empty:
            continue

        allocations = _allocate_additions(media_counts, additions_total)
        if not allocations:
            continue

        # Preserve candidate-country temporal distribution.
        candidate_timestamps = country_rows["created_at"].to_numpy()
        if len(candidate_timestamps) == 0:
            continue

        for media_source, synth_n in allocations.items():
            if synth_n <= 0:
                continue

            # Imputation source: same region + same media source.
            donor_pool = fold_df[
                (fold_df["macro_region"] == region)
                & (fold_df["media_source"] == media_source)
            ]
            if donor_pool.empty:
                continue

            sampled_ts_idx = rng.integers(
                low=0,
                high=len(candidate_timestamps),
                size=synth_n,
            )

            for idx in sampled_ts_idx:
                synthetic_counter += 1
                synthetic_rows.append(
                    {
                        "fold_id": fold.fold_id,
                        "user_id": (
                            f"imp_{fold.fold_id}_{country}_{media_source}_"
                            f"{synthetic_counter:07d}"
                        ),
                        "created_at": pd.Timestamp(candidate_timestamps[idx]),
                        "country_code": country,
                        "media_source": media_source,
                        "is_synthetic": True,
                    }
                )

    base_rows = fold_df.drop(columns=["macro_region"])
    synthetic_df = pd.DataFrame(synthetic_rows)
    if synthetic_df.empty:
        out = base_rows
    else:
        out = pd.concat([base_rows, synthetic_df], ignore_index=True)

    return out.sort_values(["fold_id", "created_at", "country_code", "user_id"]).reset_index(drop=True)


def build_imputed_users_attribution(
    users_attribution_df: pd.DataFrame,
    countries_df: pd.DataFrame,
    folds_df: pd.DataFrame,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build fold-scoped `users_attribution_imputed` DataFrame.

    Output schema:
        fold_id, user_id, created_at, country_code, media_source, is_synthetic
    """
    required_users = {"user_id", "created_at", "country_code", "media_source"}
    required_countries = {"country_code", "region"}
    required_folds = {"fold_id", "train_end"}

    missing_users = required_users.difference(users_attribution_df.columns)
    missing_countries = required_countries.difference(countries_df.columns)
    missing_folds = required_folds.difference(folds_df.columns)

    if missing_users:
        raise ValueError(f"users_attribution is missing required columns: {sorted(missing_users)}")
    if missing_countries:
        raise ValueError(f"countries is missing required columns: {sorted(missing_countries)}")
    if missing_folds:
        raise ValueError(f"rocv_folds is missing required columns: {sorted(missing_folds)}")

    users_df = users_attribution_df.copy()
    users_df["created_at"] = pd.to_datetime(users_df["created_at"], utc=True)
    before_filter = len(users_df)
    users_df = users_df.dropna(subset=["user_id", "created_at", "country_code", "media_source"]).copy()
    users_df["user_id"] = users_df["user_id"].astype(str).str.strip()
    users_df["country_code"] = users_df["country_code"].astype(str).str.strip()
    users_df["media_source"] = users_df["media_source"].astype(str).str.strip()
    users_df = users_df[
        (users_df["user_id"] != "")
        & (users_df["country_code"] != "")
        & (users_df["media_source"] != "")
    ].copy()
    dropped = before_filter - len(users_df)
    if dropped > 0:
        logger.warning("Dropped %d invalid users_attribution rows with null/blank keys", dropped)

    country_region = countries_df[["country_code", "region"]].copy()
    country_region["region"] = country_region["region"].fillna("ROW")
    users_df = users_df.merge(country_region, on="country_code", how="left")
    users_df["macro_region"] = users_df["region"].fillna("ROW")

    fold_boundaries = [
        FoldBoundary(
            fold_id=str(row.fold_id),
            train_end=_to_date(row.train_end),
        )
        for row in folds_df.sort_values("train_end").itertuples()
    ]
    rng = np.random.default_rng(seed)

    fold_outputs = [
        _build_fold_imputation(users_df, fold, rng)
        for fold in fold_boundaries
    ]
    out = pd.concat(fold_outputs, ignore_index=True)
    out["is_synthetic"] = out["is_synthetic"].astype(bool)
    return out


def fetch_inputs(client: bigquery.Client, project: str, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load source tables from BigQuery."""
    users_sql = f"""
        SELECT user_id, created_at, country_code, media_source
        FROM `{project}.{dataset}.users_attribution`
    """
    countries_sql = f"""
        SELECT country_code, region
        FROM `{project}.{dataset}.countries`
    """
    folds_sql = f"""
        SELECT fold_id, train_end
        FROM `{project}.{dataset}.rocv_folds`
    """

    users_df = client.query(users_sql).to_dataframe()
    countries_df = client.query(countries_sql).to_dataframe()
    folds_df = client.query(folds_sql).to_dataframe()
    return users_df, countries_df, folds_df


def write_imputed_table(
    client: bigquery.Client,
    project: str,
    dataset: str,
    imputed_df: pd.DataFrame,
    table: str = "users_attribution_imputed",
) -> None:
    """Write the imputed table with idempotent replace semantics."""
    table_id = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        schema=[
            bigquery.SchemaField("fold_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("user_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
            bigquery.SchemaField("country_code", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("media_source", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("is_synthetic", "BOOL", mode="REQUIRED"),
        ],
        time_partitioning=bigquery.TimePartitioning(field="created_at"),
        clustering_fields=["fold_id", "country_code", "media_source"],
    )
    client.load_table_from_dataframe(imputed_df, table_id, job_config=job_config).result()


def run(
    bq_project: str,
    bq_dataset: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Execute fold-aware users attribution imputation and persist to BigQuery."""
    client = bigquery.Client(project=bq_project)
    users_df, countries_df, folds_df = fetch_inputs(client, bq_project, bq_dataset)

    logger.info("Fetched users_attribution rows: %d", len(users_df))
    logger.info("Fetched countries rows: %d", len(countries_df))
    logger.info("Fetched rocv_folds rows: %d", len(folds_df))

    imputed_df = build_imputed_users_attribution(users_df, countries_df, folds_df, seed=seed)
    write_imputed_table(client, bq_project, bq_dataset, imputed_df)

    logger.info(
        "users_attribution_imputed written: %d rows (%d synthetic)",
        len(imputed_df),
        int(imputed_df["is_synthetic"].sum()),
    )
    return imputed_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run users_attribution fold-aware density-floor imputation.")
    parser.add_argument("--project", required=True, help="BigQuery project id")
    parser.add_argument("--dataset", required=True, help="BigQuery dataset id")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = parse_args()
    run(bq_project=args.project, bq_dataset=args.dataset, seed=args.seed)


if __name__ == "__main__":
    main()
