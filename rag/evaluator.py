import json
import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from retrieval import PoliticalRAGRetriever
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class RAGRetrievalEvaluator:
    def __init__(self, retriever, test_data_path: str, sample_size=50):
        """
        Initializes the evaluator with the configured retriever and test dataset.
        """
        self.retriever = retriever
        self.test_data_path = test_data_path
        self.sample_size = sample_size
        self.df = None
        self.results = []

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

    def run_evaluation(self, k: int = 1):
        """
        Iterates through the test data and queries the Qdrant database to retrieve the top-k chunks.
        Evaluates the top-1 retrieved party against the ground truth and saves detailed records.
        """
        if self.df is None:
            self.load_and_preprocess_data()

        logging.info(f"Running retrieval evaluation (Top-{k} search)...")

        y_true = []
        y_pred = []
        self.results = []  # Zurücksetzen, falls die Methode mehrfach aufgerufen wird

        for index, row in tqdm(
            self.df.iterrows(), total=len(self.df), desc="Evaluating Retrievals"
        ):
            query_text = row["full_text"]
            true_party = row["party"]

            # Falls "index" als Spalte in der CSV existiert, nehmen wir diesen,
            # ansonsten nehmen wir den DataFrame-Zeilenindex
            original_index = row.get("index", index)

            try:
                retrieved_points = self.retriever.search(query=query_text, limit=k)

                if retrieved_points:
                    # Extrahiere die besten Metadaten
                    predicted_party = retrieved_points[0].payload.get(
                        "party", "UNKNOWN"
                    )
                    retrieved_chunk = retrieved_points[0].payload.get("text", "")
                    score = retrieved_points[0].score
                else:
                    predicted_party = "NO_HIT"
                    retrieved_chunk = None
                    score = 0.0

                # Evaluierungs-Logik
                is_correct = true_party == predicted_party

                y_true.append(true_party)
                y_pred.append(predicted_party)

                # Speichere alle Metadaten für die spätere Fehleranalyse
                self.results.append(
                    {
                        "test_index": int(
                            original_index
                        ),  # int() stellt JSON-Kompatibilität sicher
                        "true_party": true_party,
                        "predicted_party": predicted_party,
                        "is_correct": bool(is_correct),
                        "retrieved_chunk": retrieved_chunk,
                        "query": query_text,
                        "score": float(score),
                    }
                )

            except Exception as e:
                logging.warning(f"Failed retrieval for index {index}: {e}")
                y_true.append(true_party)
                y_pred.append("ERROR")

                # Auch Fehler sauber in die Resultate aufnehmen
                self.results.append(
                    {
                        "test_index": int(original_index),
                        "true_party": true_party,
                        "predicted_party": "ERROR",
                        "is_correct": False,
                        "retrieved_chunk": None,
                        "query": query_text,
                        "score": 0.0,
                        "error_message": str(e),
                    }
                )

        self.df["predicted_party"] = y_pred
        return y_true, y_pred

    def save_results_to_json(self, output_filepath: str = "evaluation_results.json"):
        """
        Saves the structured evaluation results into a JSON file for qualitative error analysis
        """
        if not self.results:
            logging.error("No results to save. Please run run_evaluation() first.")
            return

        try:
            with open(output_filepath, "w", encoding="utf-8") as f:
                json.dump(self.results, f, ensure_ascii=False, indent=4)
            logging.info(
                f"Successfully saved detail evaluation to '{output_filepath}'."
            )
        except Exception as e:
            logging.error(f"Failed to save JSON: {e}")

    def generate_metrics(self, y_true, y_pred):
        """
        Calculates and outputs F1-Scores and generates a confusion matrix
        """
        logging.info("Generating evaluation metrics...")

        valid_indices = [
            i for i, p in enumerate(y_pred) if p not in ("NO_HIT", "ERROR")
        ]
        y_true_clean = [y_true[i] for i in valid_indices]
        y_pred_clean = [y_pred[i] for i in valid_indices]

        # 1. Classification Report (Macro & Weighted F1)
        report = classification_report(y_true_clean, y_pred_clean, zero_division=0)
        print("\n--- Retrieval Performance Report ---")
        print(report)

        # 2. Confusion Matrix
        labels = sorted(list(set(y_true_clean) | set(y_pred_clean)))
        cm = confusion_matrix(y_true_clean, y_pred_clean, labels=labels)

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Confusion Matrix: True Party vs. Retrieved Party")
        plt.ylabel("True Party (Social Media)")
        plt.xlabel("Retrieved Party (Bundestag)")
        plt.tight_layout()
        plt.savefig("retrieval_confusion_matrix.png")
        logging.info("Saved confusion matrix to 'retrieval_confusion_matrix.png'.")


if __name__ == "__main__":
    retriever = PoliticalRAGRetriever()

    # Initialisiere die Evaluierungs-Klasse
    evaluator = RAGRetrievalEvaluator(
        retriever=retriever,
        test_data_path="src/datasets/dataset_final_merged_with_mbfc_labels_without_duplicates_index_reset_anonymized_5_party_labels_media_labels_author_labels_stance_labels.csv",
        sample_size=50,
    )

    # 1. Starte den Lauf
    y_true, y_pred = evaluator.run_evaluation(k=1)

    # 2. Generiere deine Visualisierungen und Metriken
    evaluator.generate_metrics(y_true, y_pred)

    # 3. Speichere das qualitative Log
    evaluator.save_results_to_json("rag_test_results.json")
