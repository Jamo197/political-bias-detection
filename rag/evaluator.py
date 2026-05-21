import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from retrieval import PoliticalRAGRetriever
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


class RAGRetrievalEvaluator:
    """
    Evaluates RAG retrieval quality and ideological alignment against CHES ground truth.
    """

    def __init__(
        self,
        retriever: Any,
        test_data_path: str,
        ches_data_path: str,
        sample_size: int = 50,
        run_name: Optional[str] = None,
        country_context: str = "Germany",
    ):
        self.retriever = retriever
        self.test_data_path = test_data_path
        self.sample_size = sample_size
        self.country_context = country_context
        self.df = None
        self.results = []
        self.ches_df = None

        self.run_name = run_name or self._generate_run_name()
        self.run_timestamp = datetime.now()
        self.run_config = {
            "run_name": self.run_name,
            "timestamp": self.run_timestamp.isoformat(),
            "sample_size": sample_size,
            "country_context": country_context,
            "retrieval_mode": (
                getattr(retriever, "retrieval_mode", "unknown")
                if not hasattr(retriever, "retrieval_strategy")
                else getattr(retriever.retrieval_strategy, "__class__", None).__name__
            ),
        }

        self._load_ches_ground_truth(ches_data_path)
        self.load_and_preprocess_data()
        logger.info(f"Initialized evaluator with run: {self.run_name}")

    @staticmethod
    def _generate_run_name() -> str:
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def load_and_preprocess_data(self):
        """Loads test data and extracts temporal metadata for evaluation."""
        logger.info(f"Loading test data from {self.test_data_path}...")
        try:
            self.df = pd.read_csv(self.test_data_path)
            self.df["full_text"] = (
                self.df["full_text"]
                .astype(str)
                .str.replace(r"\/comma ", ",", regex=True)
            )
            self.df = self.df.dropna(subset=["full_text", "party"])
            self.df = self.df[self.df["full_text"].str.strip() != ""]

            party_mapping = {
                "AfD": "AfD",
                "B90Grune": "BÜNDNIS 90/DIE GRÜNEN",
                "CDU": "CDU",
                "CSU": "CSU",
                "FDP": "FDP",
                "Linke": "DIE LINKE",
                "SPD": "SPD",
            }
            self.df["party"] = (
                self.df["party"].map(party_mapping).fillna(self.df["party"])
            )

            if "date" in self.df.columns:
                self.df["year"] = pd.to_datetime(
                    self.df["date"], errors="coerce"
                ).dt.year
            else:
                self.df["year"] = 2021

            if len(self.df) > self.sample_size:
                self.df = self.df.sample(
                    n=self.sample_size, random_state=42
                ).reset_index(drop=True)

            logger.info(f"Loaded {len(self.df)} validation samples.")
            self.run_config["actual_sample_size"] = len(self.df)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _load_ches_ground_truth(self, ches_path: str):
        """Loads full CHES panel data."""
        logger.info(f"Loading temporal CHES ground truth from {ches_path}...")
        try:
            self.id_to_std_name = {
                301: "CDU/CSU",
                308: "CDU/CSU",
                302: "SPD",
                303: "FDP",
                304: "BÜNDNIS 90/DIE GRÜNEN",
                306: "DIE LINKE",
                310: "AfD",
                313: "BSW",
            }

            ches_raw = pd.read_csv(ches_path)
            self.ches_df = ches_raw[
                ches_raw["party_id"].isin(self.id_to_std_name.keys())
            ].copy()
            self.ches_df["std_party"] = self.ches_df["party_id"].map(
                self.id_to_std_name
            )

            self.ches_df = self.ches_df[
                ["std_party", "year", "lrecon", "galtan", "lrgen"]
            ].dropna()
            logger.info(f"Loaded {len(self.ches_df)} CHES party-wave data points.")

        except Exception as e:
            logger.error(f"Failed to load CHES dataset: {e}")
            raise

    def _get_closest_ches_score(
        self, party: str, year: int
    ) -> Tuple[float, float, float]:
        """Finds the CHES score from the wave closest to the document's year."""
        # Align query party names with consolidated CHES party names
        # FIXME: In the DB party is saved as "CDU/CSU", but in CHES it is sepereated
        # Currently only CSU values are used -> maybe use mean of both
        normalized_party = "CDU/CSU" if party in ["CDU", "CSU"] else party

        if pd.isna(year) or normalized_party not in self.ches_df["std_party"].values:
            return np.nan, np.nan, np.nan

        party_ches = self.ches_df[self.ches_df["std_party"] == normalized_party]
        if party_ches.empty:
            return np.nan, np.nan, np.nan

        closest_idx = (party_ches["year"] - year).abs().idxmin()
        row = party_ches.loc[closest_idx]
        return float(row["lrecon"]), float(row["galtan"]), float(row["lrgen"])

    def run_evaluation(self, k: int = 5) -> Dict[str, Any]:
        """
        Executes the evaluation loop across the dataset, computing
        both categorical and continuous ideological retrieval metrics.
        """
        logger.info(f"Starting evaluation run loop (k={k})...")
        self.results = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Evaluating"):
            query_text = row["full_text"]
            query_party = row["party"]
            query_year = int(row["year"]) if not pd.isna(row["year"]) else 2021

            gt_lrecon, gt_galtan, gt_lrgen = self._get_closest_ches_score(
                query_party, query_year
            )

            if np.isnan(gt_lrgen):
                continue

            try:
                retrieved_points = self.retriever.search(query=query_text, limit=k)
            except Exception as e:
                logger.error(f"Retrieval error: {e}")
                continue

            if not retrieved_points:
                continue

            retrieved_items = []
            for idx, point in enumerate(retrieved_points):
                payload = point.payload
                score = point.score
                raw_date = payload.get("date", "")
                ret_year = int(raw_date[:4]) if raw_date else query_year

                ret_party = payload.get("party", "")
                ret_lrecon, ret_galtan, ret_lrgen = self._get_closest_ches_score(
                    ret_party, ret_year
                )

                retrieved_items.append(
                    {
                        "pos": idx + 1,
                        "party": ret_party,
                        "year": ret_year,
                        "score": score,
                        "lrecon": ret_lrecon,
                        "galtan": ret_galtan,
                        "lrgen": ret_lrgen,
                    }
                )

            # Retun values: {'pos': 1, 'party': 'SPD', 'year': 2020, 'score': 0.66501206, 'lrecon': 3.714285612106323, 'galtan': 3.38095235824585, 'lrgen': 3.61904764175415}
            # Calculate Categorical Performance
            top_1_item = retrieved_items[0]
            top_1_match = 1.0 if top_1_item["party"] == query_party else 0.0
            party_density = sum(
                1 for item in retrieved_items if item["party"] == query_party
            ) / len(retrieved_items)

            mrr_party = 0.0
            for rank, item in enumerate(retrieved_items, start=1):
                if item["party"] == query_party:
                    mrr_party = 1.0 / rank
                    break
            # Process Continuous Structural Deviations
            query_metrics = {
                "query_party": query_party,
                "query_year": query_year,
                "top_1_party_match": top_1_match,
                "top_k_party_density": party_density,
                "mrr_party": mrr_party,
            }

            axes = {"lrecon": gt_lrecon, "galtan": gt_galtan, "lrgen": gt_lrgen}
            for axis, gt_val in axes.items():
                valid_scores = [
                    item[axis] for item in retrieved_items if not np.isnan(item[axis])
                ]
                valid_sims = [
                    item["score"]
                    for item in retrieved_items
                    if not np.isnan(item[axis])
                ]

                if not valid_scores:
                    continue

                # Top-1 Absolute Error
                query_metrics[f"{axis}_top1_ae"] = abs(gt_val - valid_scores[0])
                # Unweighted Aggregate Average Deviation
                query_metrics[f"{axis}_meanK_ae"] = abs(gt_val - np.mean(valid_scores))

                # Weight scores by retrieval similarity using Softmax # TODO: Explain
                exp_sims = np.exp(valid_sims)
                sim_weights = exp_sims / np.sum(exp_sims)
                weighted_pred = np.average(valid_scores, weights=sim_weights)
                query_metrics[f"{axis}_weightedK_ae"] = abs(gt_val - weighted_pred)

            self.results.append(query_metrics)

        # Aggregate Run Metrics Across the Sample Population
        summary_statistics = self._compute_summary_metrics()
        self._save_run_artifacts(summary_statistics)

        return summary_statistics

    def _compute_summary_metrics(self) -> Dict[str, float]:
        """Calculates structural mean averages over all single query results."""
        if not self.results:
            return {}

        res_df = pd.DataFrame(self.results)
        summary = {
            "top_1_party_match_rate": float(res_df["top_1_party_match"].mean()),
            "top_k_party_density_avg": float(res_df["top_k_party_density"].mean()),
            "mean_reciprocal_rank_party": float(res_df["mrr_party"].mean()),
        }

        for axis in ["lrecon", "galtan", "lrgen"]:
            for metric in ["top1_ae", "meanK_ae", "weightedK_ae"]:
                col_name = f"{axis}_{metric}"
                if col_name in res_df.columns:
                    summary[f"mae_{col_name}"] = float(res_df[col_name].mean())

        return summary

    def _save_run_artifacts(self, summary: Dict[str, float]):
        """Persists structural test results into disk log folders."""
        artifact = {
            "config": self.run_config,
            "metrics": summary,
            "detailed_results": self.results,
        }
        output_path = LOG_DIR / f"{self.run_name}_evaluation_report.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(artifact, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation metrics saved to storage path: {output_path}")


if __name__ == "__main__":
    test_data_path = "src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv"
    ches_path = "src/datasets/ground_truth/1999-2024_CHES.csv"

    retriever_simple = PoliticalRAGRetriever(retrieval_mode="hyde")

    eval_simple = RAGRetrievalEvaluator(
        retriever=retriever_simple,
        test_data_path=test_data_path,
        ches_data_path=ches_path,
        sample_size=50,
        run_name="hyde_baseline_run",
    )

    metrics_report = eval_simple.run_evaluation(k=5)

    print(json.dumps(metrics_report, indent=2))
