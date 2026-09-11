# Qualitative Annotation App

This package is a local Streamlit app for annotating the supplied qualitative RAG samples.

## Requirements

- Python 3.14 or newer
- macOS, Linux, or Windows with a shell

## Start the app

From this directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then start the app:

```bash
./run_app.sh
```

The app opens in a browser. On Windows, run `streamlit run main.py` from this directory instead.

## Annotation set

The app always selects the same 25 samples from the bundled JSONL file using random seed `42`. Do not edit or reorder the sample file.

For the full context, quadrant descriptions, rubric definitions, and edge-case rules, open `help_page.md`.

Complete the three rubric dimensions (`R_top`, `R_ideo`, and `A_caus`), and save each sample. Moving to the next sample is only possible through **Save annotations and go next**. Saving replaces the existing record for that sample, so the final file contains one annotation record per sample.

When finished, click **Download annotations** and send the downloaded `qualitative_annotations.jsonl` file back. 