#!/bin/sh
set -eu
cd "$(dirname "$0")"
exec streamlit run main.py
