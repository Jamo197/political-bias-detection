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
        retriever: PoliticalRAGRetriever,
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
                getattr(retriever.retrieval_strategy, "__class__", None).__name__
                if hasattr(retriever, "retrieval_strategy")
                else "unknown"
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

            # Map parties to standard names
            # Values: ['AfD', 'B90Grune', 'CDU', 'CSU', 'FDP', 'Linke', 'SPD']
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

            # TODO: Use random sample_size=50 with fixed state=42
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
        """
        Loads full CHES panel data.
        """
        logger.info(f"Loading temporal CHES ground truth from {ches_path}...")
        try:
            # Map CHES party_id to standardized names (extendable for RQ2.2)
            self.id_to_std_name = {
                301: "CDU/CSU",  # CDU
                308: "CDU/CSU",  # CSU
                302: "SPD",
                303: "FDP",
                304: "BÜNDNIS 90/DIE GRÜNEN",
                306: "DIE LINKE",
                310: "AfD",
                313: "BSW",
            }

            ches_raw = pd.read_csv(ches_path)
            # Filter for relevant parties and required columns
            self.ches_df = ches_raw[
                ches_raw["party_id"].isin(self.id_to_std_name.keys())
            ].copy()
            self.ches_df["std_party"] = self.ches_df["party_id"].map(
                self.id_to_std_name
            )

            # Keep historical waves instead of just the latest
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
        if pd.isna(year) or party not in self.ches_df["std_party"].values:
            return np.nan, np.nan, np.nan

        party_ches = self.ches_df[self.ches_df["std_party"] == party]
        if party_ches.empty:
            return np.nan, np.nan, np.nan

        # Find closest wave
        closest_idx = (party_ches["year"] - year).abs().idxmin()
        row = party_ches.loc[closest_idx]
        return float(row["lrecon"]), float(row["galtan"]), float(row["lrgen"])

    # FIXME: Create new evaluation run
    def run_evaluation(self, k: int = 5):
        if self.df is None:
            self.load_and_preprocess_data()

    def run_evaluation_old(self, k: int = 5):
        if self.df is None:
            self.load_and_preprocess_data()

        logger.info(f"Running retrieval evaluation (Top-{k} search)...")
        self.run_config["k_value"] = k
        start_time = datetime.now()
        self.results = []

        for index, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Evaluating"
        ):
            query_text = row["full_text"]
            true_party = row["party"]
            query_year = row["year"]

            try:
                retrieved_points = self.retriever.search(query=query_text, limit=k)
                retrieved_parties = []
                top_1_year = np.nan

                if retrieved_points:
                    for i, point in enumerate(retrieved_points):
                        retrieved_parties.append(point.payload.get("party", "UNKNOWN"))
                        if i == 0:
                            date_str = point.payload.get("date", "")
                            top_1_year = int(date_str[:4]) if date_str else query_year
                else:
                    retrieved_parties = ["NO_HIT"] * k

                # CDU/CSU is saved in the vector database
                hit_at_k = true_party in retrieved_parties
                # if true_party == "CDU" or true_party == "CSU":
                #     true_party = "CDU/CSU"
                #     print("Stop")
                try:
                    rank = retrieved_parties.index(true_party) + 1
                    reciprocal_rank = 1.0 / rank
                except ValueError:
                    reciprocal_rank = 0.0

                top_1_party = retrieved_parties[0] if retrieved_parties else "NO_HIT"

                true_lrecon, true_galtan, true_lrgen = self._get_closest_ches_score(
                    true_party, query_year
                )
                pred_lrecon, pred_galtan, pred_lrgen = self._get_closest_ches_score(
                    top_1_party, top_1_year
                )

                self.results.append(
                    {
                        "test_index": int(row.get("index", index)),
                        "true_party": true_party,
                        "true_party_lrgen": true_lrgen,
                        "top_1_pred_party": top_1_party,
                        "top_1_pred_party_lrgen": pred_lrgen,
                        "retrieved_parties_top_k": retrieved_parties,
                        "hit_at_k": bool(hit_at_k),
                        "reciprocal_rank": float(reciprocal_rank),
                        "true_lrecon": true_lrecon,
                        "pred_lrecon": pred_lrecon,
                        "true_galtan": true_galtan,
                        "pred_galtan": pred_galtan,
                    }
                )

            except Exception as e:
                logger.warning(f"Failed retrieval for index {index}: {e}")
                self.results.append(
                    {
                        "test_index": int(row.get("index", index)),
                        "true_party": true_party,
                        "true_party_lrgen": true_lrgen,
                        "top_1_pred_party": "ERROR",
                        "top_1_pred_party_lrgen": pred_lrgen,
                        "hit_at_k": False,
                        "reciprocal_rank": 0.0,
                        "true_lrecon": -1.0,
                        "pred_lrecon": -1.0,
                        "true_galtan": -1.0,
                        "pred_galtan": -1.0,
                        "error_message": str(e),
                    }
                )

        self.run_config["evaluation_time_seconds"] = (
            datetime.now() - start_time
        ).total_seconds()
        return self.results

    def generate_metrics(self) -> Dict[str, Any]:
        valid_results = [
            r
            for r in self.results
            if r.get("top_1_pred_party") not in ["ERROR", "UNKNOWN", "NO_HIT"]
            and r.get("true_party_lrgen", -1.0) != -1.0
            and r.get("true_lrecon", -1.0) != -1.0
            and r.get("pred_lrecon", -1.0) != -1.0
        ]

        if not valid_results:
            logger.error("Insufficient valid results for metrics computation.")
            return {}

        # IR Metrics
        hits = [r["hit_at_k"] for r in valid_results]
        rrs = [r["reciprocal_rank"] for r in valid_results]
        hit_rate = np.mean(hits)
        mrr = np.mean(rrs)

        # Classification F1
        y_true_party = [r["true_party"] for r in valid_results]
        y_pred_party = [r["top_1_pred_party"] for r in valid_results]
        f1 = f1_score(y_true_party, y_pred_party, average="macro", zero_division=0)

        # Classification lrgen (how near was the results in times of general left right projection)

        # Ideological Arrays
        t_lrgen = [r["true_party_lrgen"] for r in valid_results]
        p_lrgen = [r["top_1_pred_party_lrgen"] for r in valid_results]
        t_lrecon = [r["true_lrecon"] for r in valid_results]
        p_lrecon = [r["pred_lrecon"] for r in valid_results]
        t_galtan = [r["true_galtan"] for r in valid_results]
        p_galtan = [r["pred_galtan"] for r in valid_results]

        # MAE
        mae_lrgen = mean_absolute_error(t_lrgen, p_lrgen)
        mae_lrecon = mean_absolute_error(t_lrecon, p_lrecon)
        mae_galtan = mean_absolute_error(t_galtan, p_galtan)

        # # Spearman's Rank Correlation (Crucial for relative ideological ordering)
        # spearman_lrgen, _ = spearmanr(t_lrgen, p_lrgen)
        # spearman_lrecon, _ = spearmanr(t_lrecon, p_lrecon)
        # spearman_galtan, _ = spearmanr(t_galtan, p_galtan)

        self.metrics = {
            "hit_rate": float(hit_rate),
            "mean_reciprocal_rank": float(mrr),
            "f1_score_macro": float(f1),
            "mae_lrgen": float(mae_lrgen),
            "mae_lrecon": float(mae_lrecon),
            "mae_galtan": float(mae_galtan),
            # "spearman_lrgen": (
            #     float(spearman_lrgen) if not np.isnan(spearman_lrgen) else 0.0
            # ),
            # "spearman_lrecon": (
            #     float(spearman_lrecon) if not np.isnan(spearman_lrecon) else 0.0
            # ),
            # "spearman_galtan": (
            #     float(spearman_galtan) if not np.isnan(spearman_galtan) else 0.0
            # ),
            "valid_samples": len(valid_results),
            "error_rate": (
                1.0 - (len(valid_results) / len(self.results)) if self.results else 0.0
            ),
        }

        self._print_report()
        return self.metrics

    def save_results_to_json(self, output_path: Optional[str] = None) -> str:
        path = Path(output_path or LOG_DIR / f"{self.run_name}_results.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_config": self.run_config,
                    "metrics": getattr(self, "metrics", {}),
                    "results": self.results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        return str(path)

    def save_metrics_summary(self, output_path: Optional[str] = None) -> str:
        path = Path(output_path or LOG_DIR / "runs_summary.tsv")
        path.parent.mkdir(parents=True, exist_ok=True)
        m = getattr(self, "metrics", {})

        row = {
            "run_name": self.run_name,
            "mode": self.run_config.get("retrieval_mode"),
            "hit_rate": m.get("hit_rate", "N/A"),
            "mrr": m.get("mean_reciprocal_rank", "N/A"),
            "f1": m.get("f1_score_macro", "N/A"),
            "lrgen_mae": m.get("mae_lrgen", "N/A"),
            "lrecon_mae": m.get("mae_lrecon", "N/A"),
            # "lrecon_spearman": m.get("spearman_lrecon", "N/A"),
            "galtan_mae": m.get("mae_galtan", "N/A"),
            # "galtan_spearman": m.get("spearman_galtan", "N/A"),
        }

        write_header = not path.exists()
        with open(path, "a", encoding="utf-8") as f:
            if write_header:
                f.write("\t".join(row.keys()) + "\n")
            f.write(
                "\t".join(
                    str(v) if not isinstance(v, float) else f"{v:.4f}"
                    for v in row.values()
                )
                + "\n"
            )
        return str(path)

    def _print_report(self):
        print("\n" + "=" * 55)
        print(f"EVALUATION RUN: {self.run_name}")
        print("=" * 55)
        print(f"Retrieval Mode: {self.run_config.get('retrieval_mode')}")
        print(f"Valid Samples:  {self.metrics['valid_samples']}/{len(self.results)}")

        print("\n--- 1. Information Retrieval Metrics ---")
        print(f"Hit Rate @ K:   {self.metrics['hit_rate']:.4f}")
        print(f"MRR:            {self.metrics['mean_reciprocal_rank']:.4f}")
        print(f"Party F1 Macro: {self.metrics['f1_score_macro']:.4f}")

        print("\n--- 2. Ideological Alignment (CHES) ---")
        print(f"LRGEN MAE:     {self.metrics['mae_lrgen']:.4f}")
        print(f"LRECON MAE:     {self.metrics['mae_lrecon']:.4f}")
        print(f"GALTAN MAE:     {self.metrics['mae_galtan']:.4f}")
        print("=" * 55 + "\n")


if __name__ == "__main__":
    retriever_simple = PoliticalRAGRetriever(retrieval_mode="simple")
    eval_simple = RAGRetrievalEvaluator(
        retriever=retriever_simple,
        test_data_path="src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv",
        ches_data_path="src/datasets/ground_truth/1999-2024_CHES.csv",
        sample_size=50,
        run_name="simple_baseline",
    )
    # eval_simple._get_closest_ches_score("CDU", 2021)
    eval_simple.run_evaluation(k=5)

    # 1. SIMPLE PIPELINE
    # retriever_simple = PoliticalRAGRetriever(retrieval_mode="simple")
    # eval_simple = RAGRetrievalEvaluator(
    #     retriever=retriever_simple,
    #     test_data_path="src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv",
    #     ches_data_path="src/datasets/ground_truth/1999-2024_CHES.csv",
    #     sample_size=50,
    #     run_name="simple_baseline",
    # )
    # eval_simple.run_evaluation(k=5)
    # eval_simple.generate_metrics()
    # p1 = eval_simple.save_results_to_json()
    # eval_simple.save_metrics_summary()

    # NOTE: Uncomment to run further pipelines and comparisons
    # 2. HyDE PIPELINE
    # retriever_hyde = PoliticalRAGRetriever(retrieval_mode="hyde")
    # eval_hyde = RAGRetrievalEvaluator(
    #     retriever=retriever_hyde,
    #     test_data_path="src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv",
    #     ches_data_path="src/datasets/ground_truth/1999-2024_CHES.csv",
    #     sample_size=50,
    #     run_name="hyde",
    # )
    # eval_hyde.run_evaluation(k=5)
    # eval_hyde.generate_metrics()
    # p2 = eval_hyde.save_results_to_json()

    # 3. COMPARISON (RQ1.2)
    # df_comp = RunComparator.compare_runs([p1, p2])
    # RunComparator.visualize_comparison(df_comp)
