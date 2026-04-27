import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from retrieval import PoliticalRAGRetriever
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RAGRetrievalEvaluator:

    def __init__(
        self, retriever, test_data_path: str, ches_data_path: str, sample_size=50
    ):
        """
        Initializes the evaluator with the configured retriever and test dataset.
        """
        self.retriever = retriever
        self.test_data_path = test_data_path
        self.sample_size = sample_size
        self.df = None
        self.results = []
        self._load_ches_ground_truth(ches_data_path)

    def load_and_preprocess_data(self):
        """
        Loads the test dataset, handles necessary text cleaning,
        and selects a random sample of 50 rows.
        """
        logging.info(f"Loading test data from {self.test_data_path}...")
        try:
            self.df = pd.read_csv(self.test_data_path)

            self.df["full_text"] = (
                self.df["full_text"]
                .astype(str)
                .str.replace(r"\\comma", ",", regex=True)
            )

            self.df = self.df.dropna(subset=["full_text", "party"])
            self.df = self.df[self.df["full_text"].str.strip() != ""]

            party_mapping = {
                "AfD": "AfD",
                "B90Grune": "BÜNDNIS 90/DIE GRÜNEN",
                "CDU": "CDU/CSU",
                "CSU": "CDU/CSU",
                "FDP": "FDP",
                "Linke": "DIE LINKE",
                "SPD": "SPD",
            }

            self.df["party"] = (
                self.df["party"].map(party_mapping).fillna(self.df["party"])
            )

            if len(self.df) > self.sample_size:
                self.df = self.df.sample(
                    n=self.sample_size, random_state=42
                ).reset_index(drop=True)

            logging.info(
                f"Successfully loaded and preprocessed {len(self.df)} validation samples (random sample)."
            )
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

    def _load_ches_ground_truth(self, ches_path: str):
        """
        Dynamically loads CHES data, extracts the most recent ideological scores
        for German parties, and standardizes them to match the Bundestag dataset.
        """
        logging.info(f"Loading CHES ground truth from {ches_path}...")
        try:
            ches_df = pd.read_csv(ches_path)

            # Map CHES party_id to your standardized Bundestag names
            id_to_std_name = {
                301: "CDU/CSU",  # CDU
                308: "CDU/CSU",  # CSU
                302: "SPD",
                303: "FDP",
                304: "BÜNDNIS 90/DIE GRÜNEN",
                306: "DIE LINKE",
                310: "AfD",
                313: "BSW",
            }

            german_party_ids = list(id_to_std_name.keys())
            ches_de = ches_df[ches_df["party_id"].isin(german_party_ids)].copy()

            latest_ches = ches_de.sort_values("year", ascending=False).drop_duplicates(
                "party_id"
            )

            latest_ches["std_party"] = latest_ches["party_id"].map(id_to_std_name)

            aggregated_ches = (
                latest_ches.groupby("std_party")[["lrecon", "galtan"]]
                .mean()
                .reset_index()
            )

            mapping = {}
            for _, row in aggregated_ches.iterrows():
                mapping[row["std_party"]] = {
                    "lrecon": row["lrecon"],
                    "galtan": row["galtan"],
                }

            for fallback in ["UNKNOWN", "NO_HIT", "ERROR"]:
                mapping[fallback] = {"lrecon": np.nan, "galtan": np.nan}

            self.ches_mapping = mapping
            logging.info(
                "Successfully mapped CHES ground truth for vector distance evaluation."
            )

        except Exception as e:
            logging.error(f"Failed to load CHES dataset: {e}")
            raise

    def run_evaluation(self, k: int = 5):
        """
        Iterates through the test data and queries the Qdrant database to retrieve the top-k chunks.
        """
        if self.df is None:
            self.load_and_preprocess_data()

        logging.info(f"Running retrieval evaluation (Top-{k} search)...")

        self.results = []

        for index, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Evaluating Retrievals"
        ):
            query_text = row["full_text"]
            true_party = row["party"]
            original_index = row.get("index", index)

            # Falls "index" als Spalte in der CSV existiert, nehmen wir diesen,
            # ansonsten nehmen wir den DataFrame-Zeilenindex
            original_index = row.get("index", index)

            try:
                retrieved_points = self.retriever.search(query=query_text, limit=k)

                retrieved_parties = []
                retrieved_chunks = []

                if retrieved_points:
                    for point in retrieved_points:
                        retrieved_parties.append(point.payload.get("party", "UNKNOWN"))
                        retrieved_chunks.append(point.payload.get("text", ""))
                else:
                    retrieved_parties = ["NO_HIT"] * k

                # 1. Hit Rate @ K Logic
                hit_at_k = true_party in retrieved_parties

                # 2. Reciprocal Rank Logic
                try:
                    rank = retrieved_parties.index(true_party) + 1
                    reciprocal_rank = 1.0 / rank
                except ValueError:
                    reciprocal_rank = 0.0

                top_1_party = retrieved_parties[0] if retrieved_parties else "NO_HIT"

                if top_1_party == "UNKNOWN":
                    print("HI")
                # Fetch 2D ideological dictionaries
                true_ideo_dict = self.ches_mapping.get(true_party, -1.0)
                pred_ideo_dict = self.ches_mapping.get(top_1_party, -1.0)

                def safe_float(val):
                    try:
                        if val is None or (isinstance(val, float) and np.isnan(val)):
                            return -1.0
                        return float(val)
                    except Exception:
                        return -1.0

                self.results.append(
                    {
                        "test_index": int(original_index),
                        "true_party": true_party,
                        "top_1_pred_party": top_1_party,
                        "retrieved_parties_top_k": retrieved_parties,
                        "hit_at_k": bool(hit_at_k),
                        "reciprocal_rank": float(reciprocal_rank),
                        "true_lrecon": safe_float(float(true_ideo_dict["lrecon"])),
                        "pred_lrecon": safe_float(pred_ideo_dict["lrecon"]),
                        "true_galtan": safe_float(float(true_ideo_dict["galtan"])),
                        "pred_galtan": safe_float(pred_ideo_dict["galtan"]),
                        "query": query_text,
                    }
                )

            except Exception as e:
                logging.warning(f"Failed retrieval for index {index}: {e}")

                self.results.append(
                    {
                        "test_index": int(original_index),
                        "true_party": true_party,
                        "top_1_pred_party": "ERROR",
                        "retrieved_parties_top_k": "ERROR",
                        "hit_at_k": False,
                        "reciprocal_rank": float(0),
                        "true_lrecon": float(0),
                        "pred_lrecon": float(0),
                        "true_galtan": float(0),
                        "pred_galtan": float(0),
                        "query": query_text,
                        "error_message": str(e),
                    }
                )

        return self.results

    def generate_metrics(self):
        """
        Calculates and outputs aggregated IR and Ideological metrics.
        """
        logging.info("Generating advanced evaluation metrics...")

        valid_results = [
            r
            for r in self.results
            if not r.get("top_1_pred_party") == "ERROR"
            and not r.get("top_1_pred_party") == "UNKNOWN"
        ]

        if not valid_results:
            logging.error("No valid results to compute metrics.")
            return

        # IR Metrics
        hits = [r["hit_at_k"] for r in valid_results]
        rrs = [r["reciprocal_rank"] for r in valid_results]

        hit_rate = np.mean(hits)
        mrr = np.mean(rrs)

        # Multi-dimensional Ideological Metrics
        y_true_lrecon = [r["true_lrecon"] for r in valid_results]
        y_pred_lrecon = [r["pred_lrecon"] for r in valid_results]

        y_true_galtan = [r["true_galtan"] for r in valid_results]
        y_pred_galtan = [r["pred_galtan"] for r in valid_results]

        mae_lrecon = mean_absolute_error(y_true_lrecon, y_pred_lrecon)
        mae_galtan = mean_absolute_error(y_true_galtan, y_pred_galtan)

        y_true_party = [r["true_party"] for r in valid_results]
        y_pred_party = [r["top_1_pred_party"] for r in valid_results]

        try:
            f1 = f1_score(y_true_party, y_pred_party, average="macro", zero_division=0)
        except Exception as e:
            logging.warning(f"Error calculating F1-score: {e}")
            f1 = float("nan")

        print("\n--- Advanced Retrieval Performance Report ---")
        print(f"Total Valid Samples Evaluated: {len(valid_results)}")
        print(f"Hit Rate @ K:                  {hit_rate:.4f}")
        print(f"Mean Reciprocal Rank (MRR):    {mrr:.4f}")
        print(f"Macro F1-Score (Top-1 Party):  {f1:.4f}")
        print("---------------------------------------------")
        print(f"LRECON MAE (Economic Scale):   {mae_lrecon:.4f}")
        print(f"GALTAN MAE (Cultural Scale):   {mae_galtan:.4f}")
        print("---------------------------------------------")

    def save_results_to_json(self, output_path="logs/retrieval_logs.json"):
        """
        Saves self.results to a json file at the specified path.
        """
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved results to {output_path}")
        except Exception as e:
            logging.error(f"Failed to save results to json: {e}")
            raise


if __name__ == "__main__":
    retriever = PoliticalRAGRetriever()

    # Initialisiere die Evaluierungs-Klasse
    evaluator = RAGRetrievalEvaluator(
        retriever=retriever,
        test_data_path="src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv",
        ches_data_path="src/datasets/ground_truth/1999-2024_CHES.csv",
        sample_size=50,
    )

    # 1. Starte den Lauf
    evaluator.run_evaluation()

    # 2. Generiere deine Visualisierungen und Metriken
    evaluator.generate_metrics()
    evaluator.save_results_to_json()

    # # 3. Speichere das qualitative Log
    # evaluator.save_results_to_json("rag_test_results_2.json")
