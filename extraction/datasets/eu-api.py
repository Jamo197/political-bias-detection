import json
import re
import time
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# --- Configuration ---
YEARS = [2019, 2020, 2021, 2022]
TARGET_LANG = "EN"  # "EN" for English/translated, "FR", "DE", etc.
OUTPUT_FILE = "ep_all_speeches_2019_2022.jsonl"

HEADERS = {
    "User-Agent": "EPPlenaryCorpusExtractor/1.0 (Research Text Analysis; mailto:your-email@example.com)"
}


def get_plenary_dates_for_year(year):
    """
    Queries the EP Open Data API to get all plenary sitting dates for a given year.
    """
    url = "https://data.europarl.europa.eu/api/v2/meetings"
    params = {"year": year, "limit": 100}

    try:
        resp = requests.get(
            url, headers={"Accept": "application/ld+json"}, params=params, timeout=15
        )
        if resp.status_code != 200:
            print(f"Failed to fetch meetings for {year} (HTTP {resp.status_code})")
            return []

        data = resp.json()
        items = data.get("data", []) or data.get("@graph", [])

        dates = set()
        for item in items:
            # Plenary sitting dates are formatted as YYYY-MM-DD
            start_date = item.get("activity_start_date") or item.get("date")
            if start_date:
                # Keep only the date portion (YYYY-MM-DD)
                clean_date = start_date.split("T")[0]
                dates.add(clean_date)

        return sorted(list(dates))
    except Exception as e:
        print(f"Error retrieving plenary schedule for {year}: {e}")
        return []


def parse_cre_transcript(date_str, html_content):
    """
    Parses a single day's CRE HTML verbatim report into individual speech items.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    speeches = []
    current_topic = "General Plenary Order"

    # Plenary debate containers or content blocks
    entries = soup.find_all(
        ["p", "div"],
        class_=re.compile(r"doc_subtitle|contents|doc_agent|heading", re.I),
    )

    if not entries:
        # Fallback to all standard text paragraphs
        entries = soup.find_all("p")

    current_speaker = None
    current_speech_chunks = []

    for el in entries:
        text = el.get_text(strip=True)
        if not text:
            continue

        # Check if the block indicates a new agenda item/debate title
        classes = " ".join(el.get("class", []))
        if "subtitle" in classes or el.name in ["h2", "h3", "h4"]:
            if current_speaker and current_speech_chunks:
                speeches.append(
                    {
                        "date": date_str,
                        "topic": current_topic,
                        "speaker": current_speaker,
                        "text": "\n".join(current_speech_chunks),
                    }
                )
                current_speech_chunks = []
                current_speaker = None
            current_topic = text
            continue

        # Detect speaker turn indicator (e.g. "Name (Group). – Speech text...")
        speaker_match = re.match(
            r"^([A-ZÀ-ÿ\s\.\,\-]+(?:\([A-Za-z0-9\/\- ]+\))?)\s*[\.–—]\s*(.*)", text
        )
        if speaker_match:
            # Save preceding speaker's text if present
            if current_speaker and current_speech_chunks:
                speeches.append(
                    {
                        "date": date_str,
                        "topic": current_topic,
                        "speaker": current_speaker,
                        "text": "\n".join(current_speech_chunks),
                    }
                )
                current_speech_chunks = []

            current_speaker = speaker_match.group(1).strip()
            initial_text = speaker_match.group(2).strip()
            if initial_text:
                current_speech_chunks.append(initial_text)
        else:
            if current_speaker:
                current_speech_chunks.append(text)

    # Flush final speaker
    if current_speaker and current_speech_chunks:
        speeches.append(
            {
                "date": date_str,
                "topic": current_topic,
                "speaker": current_speaker,
                "text": "\n".join(current_speech_chunks),
            }
        )

    return speeches


def extract_plenary_day(date_str):
    """
    Downloads and extracts speeches for a specific sitting date from DOCEO.
    """
    parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
    # Term 8 ran until June 30, 2019; Term 9 began July 1, 2019
    term_prefix = "CRE-8" if parsed_date < datetime(2019, 7, 1) else "CRE-9"

    url = f"https://www.europarl.europa.eu/doceo/document/{term_prefix}-{date_str}_{TARGET_LANG}.html"

    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        if resp.status_code == 404:
            # Try fallback without term prefix if structure differs
            alt_url = f"https://www.europarl.europa.eu/doceo/document/CRE-{date_str}_{TARGET_LANG}.html"
            resp = requests.get(alt_url, headers=HEADERS, timeout=20)

        if resp.status_code not in [200, 202]:
            return []

        return parse_cre_transcript(date_str, resp.raw)
    except Exception as e:
        print(f"Failed to extract sitting on {date_str}: {e}")
        return []


def main():
    total_speeches = 0

    print(f"Gathering plenary sitting dates for {YEARS}...")
    all_dates = []
    for year in YEARS:
        year_dates = get_plenary_dates_for_year(year)
        print(f"  {year}: Found {len(year_dates)} sitting dates")
        all_dates.extend(year_dates)

    print(f"\nTotal plenary sitting dates to process: {len(all_dates)}")
    print(f"Output will be streamed to '{OUTPUT_FILE}'\n" + "=" * 60)

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_file:
        for idx, date_str in enumerate(all_dates, 1):
            print(
                f"[{idx}/{len(all_dates)}] Downloading plenary debates for {date_str}...",
                end="",
                flush=True,
            )
            day_speeches = extract_plenary_day(date_str)

            for item in day_speeches:
                out_file.write(json.dumps(item, ensure_ascii=False) + "\n")

            total_speeches += len(day_speeches)
            print(
                f" Extracted {len(day_speeches)} speeches (Total so far: {total_speeches})"
            )

            # Respect server rate limits (0.5s pause between requests)
            time.sleep(0.5)

    print("=" * 60)
    print(f"Extraction complete! Saved {total_speeches} speeches to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    from curl_cffi import requests
    from bs4 import BeautifulSoup

    url = "https://www.europarl.europa.eu/doceo/document/CRE-9-2024-04-24_EN.xml"

    # Impersonates Chrome browser TLS fingerprint
    response = requests.get(url, impersonate="chrome124")

    soup = BeautifulSoup(response.content, "xml")
    for speech in soup.find_all("SPEECH")[:5]:
        speaker = speech.find("SPEAKER")
        print("Speaker:", speaker.get("NAME") if speaker else "Unknown")
        print("Party/Group:", speaker.get("POL_GROUP") if speaker else "N/A")
        print("---")
