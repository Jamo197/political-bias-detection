import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

WHITESPACE_REGEX = re.compile(r"\s+")

# Load environment variables
load_dotenv(Path(".env.local"))
# DIP API Configuration
BASE_URL = "https://search.dip.bundestag.de/api/v1"
API_KEY = os.getenv("BUNDESTAG_DIP_API_KEY")
HEADERS = {"Authorization": f"ApiKey {API_KEY}", "Accept": "application/json"}


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_api_json(url, params):
    """Handles rate-limited JSON API calls with authentication."""
    response = requests.get(url, headers=HEADERS, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def fetch_xml_content(url):
    """Handles large XML document downloads without API headers."""
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return response.content


def extract_parties_from_text(text):
    """
    Scans a string for German political parties and returns a normalized list.
    """
    party_patterns = {
        "SPD": r"\bSPD\b",
        "CDU/CSU": r"\bCDU/CSU\b|\bCDU\b|\bCSU\b|\bUnion\b",
        "GRÜNE": r"\bGRÜNE\b|\bGRÜNEN\b|\bBÜNDNIS 90/DIE GRÜNEN\b",
        "FDP": r"\bFDP\b",
        "AfD": r"\bAfD\b",
        "LINKE": r"\bLinke\b|\bLinken\b|\bDie Linke\b",
        "BSW": r"\bBSW\b|\bBündnis Sahra Wagenknecht\b",
    }

    found_parties = []

    for standard_name, pattern in party_patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            found_parties.append(standard_name)

    return found_parties


class SpeakerRegistry:
    """
    Optimized: Caches API lookups to disk to prevent redundant calls across sessions.
    """

    def __init__(self, cache_file="speaker_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache: Dict[str, str] = self._load_cache()

    def _load_cache(self) -> Dict[str, str]:
        if self.cache_file.exists():
            with open(self.cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        with open(self.cache_file, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def get_party(self, firstname: str, lastname: str, wp: int) -> str:
        cache_key = (
            f"{firstname}_{lastname}_{wp}".lower()
        )  # Added WP to key for historical accuracy
        if cache_key in self.cache:
            return self.cache[cache_key]

        if not firstname and not lastname:
            return "UNKNOWN"

        url = f"{BASE_URL}/person"
        params = {"f.person": f"{lastname}, {firstname}", "apikey": API_KEY}

        try:
            data = fetch_api_json(url, params)
            documents = data.get("documents", [])
            party = "UNKNOWN"

            if documents:
                roles = documents[0].get("person_roles", [])
                party = next(
                    (
                        r["fraktion"]
                        for r in roles
                        if wp in r.get("wahlperiode_nummer", []) and "fraktion" in r
                    ),
                    "UNKNOWN",
                )

                if party == "UNKNOWN":
                    party = next(
                        (r["fraktion"] for r in roles if "fraktion" in r), "UNKNOWN"
                    )

            self.cache[cache_key] = party
            self._save_cache()
            return party

        except Exception as e:
            print(f"Registry Lookup Failed for {firstname} {lastname}: {e}")
            return "UNKNOWN"


speaker_registry = SpeakerRegistry()


def get_protocols_for_period(start_date, end_date):
    """
    Fetches protocol metadata for a specific date range, utilizing exponential backoff.
    """
    url = f"{BASE_URL}/plenarprotokoll"
    protocols = []

    params = {
        "f.datum.start": start_date,
        "f.datum.end": end_date,
        "f.zuordnung": "BT",
        "apikey": API_KEY,
    }

    while True:
        print(f"Fetching protocol list... (Found {len(protocols)} so far)")

        try:
            data = fetch_api_json(url, params)
        except Exception as e:
            print(f"Failed to fetch protocols after retries: {e}")
            break

        protocols.extend(data.get("documents", []))
        protocols_found = data.get("numFound", 0)

        if len(protocols) >= protocols_found:
            break

        cursor = data.get("cursor")
        params = {"cursor": cursor, "apikey": API_KEY}
    return protocols


# --- 2. Data Extraction & Enrichment ---


def extract_speeches_from_xml(
    xml_url: str, protocol_date: str, protocol_id: str, legislative_period: int
) -> List[dict]:
    """
    Parses XML, inlines interjections for context preservation, and enforces
    a flat, Qdrant-ready metadata schema.
    """
    try:
        xml_content = fetch_xml_content(xml_url)
        root = ET.fromstring(xml_content)
    except Exception as e:
        print(f"XML Parsing failed for {protocol_id}: {e}")
        return []

    speeches_data = []
    year = int(protocol_date.split("-")[0])

    for top in root.findall(".//tagesordnungspunkt"):
        topic_id = top.get("top-id", "Unknown Topic")

        for rede in top.findall(".//rede"):
            rede_id = rede.get("id", "Unknown ID")

            redner_tag = rede.find(".//redner")
            if redner_tag is not None:
                vorname = redner_tag.findtext(".//vorname", default="").strip()
                nachname = redner_tag.findtext(".//nachname", default="").strip()
                fraktion = redner_tag.findtext(".//fraktion", default="").strip()

                rolle_tag = redner_tag.find(".//rolle")
                rolle = (
                    rolle_tag.findtext(".//rolle_lang", default="")
                    if rolle_tag is not None
                    else ""
                )

                speaker_name = f"{vorname} {nachname}".strip()

                if not fraktion:
                    fraktion = speaker_registry.get_party(
                        vorname, nachname, legislative_period
                    )
            else:
                speaker_name = "President/Unknown"
                fraktion = "UNKNOWN"
                rolle = "President"

            full_text_blocks = []

            for elem in rede:
                if elem.tag == "p":
                    if "redner" in elem.get("klasse", ""):
                        continue
                    text = "".join(elem.itertext()).strip()
                    if text:
                        full_text_blocks.append(text)

                elif elem.tag == "kommentar":
                    text = "".join(elem.itertext()).strip()
                    if text:
                        full_text_blocks.append(f" [{text}] ")

                elif elem.tag == "name":
                    text = "".join(elem.itertext()).strip()
                    if text:
                        full_text_blocks.append(f" [{text}:] ")

            raw_text = " ".join(full_text_blocks)
            clean_text = WHITESPACE_REGEX.sub(" ", raw_text).strip()

            speeches_data.append(
                {
                    "text": clean_text,
                    "metadata": {
                        "speech_id": rede_id,
                        "protocol_id": protocol_id,
                        "topic_id": topic_id,
                        "speaker": speaker_name,
                        "party": fraktion,
                        "role": rolle,
                        "date": protocol_date,
                        "year": year,
                        "country": "Germany",
                        "legislative_period": legislative_period,
                        "source": "Bundestag DIP",
                    },
                }
            )

    return speeches_data


def download_and_organize_protocols(start_date, end_date, base_dir="bundestag_data"):
    """
    Orchestrates the extraction of protocol metadata and raw XML files,
    organizing them into a structured directory hierarchy for RAG ingestion.
    """
    print(f"--- Starting Extraction for Period: {start_date} to {end_date} ---")

    # Utilizing the robust fetch_api_json logic defined previously
    protocols = get_protocols_for_period(start_date, end_date)
    print(f"Total protocols retrieved from API: {len(protocols)}")

    for protokoll in protocols:
        wp = protokoll.get("wahlperiode", "unknown_wp")
        datum = protokoll.get("datum", "unknown_date")
        protocol_id = protokoll.get("id", "unknown_id")
        safe_protocol_id = protocol_id.replace("/", "")

        try:
            dt_obj = datetime.strptime(datum, "%Y-%m-%d")
            year_month = dt_obj.strftime("%Y-%m")
        except ValueError:
            year_month = "unknown_month"

        wp_dir = os.path.join(
            "extraction", "datasets", base_dir, f"WP_{wp}", year_month
        )
        meta_dir = os.path.join(wp_dir, "metadata")
        xml_dir = os.path.join(wp_dir, "raw_xml")
        speech_dir = os.path.join(wp_dir, "speeches")

        # Create directories safely
        for d in [meta_dir, xml_dir, speech_dir]:
            os.makedirs(d, exist_ok=True)

        # 1. Save Full Metadata as JSON
        meta_filepath = os.path.join(
            meta_dir, f"{datum}_{safe_protocol_id}_metadata.json"
        )
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(protokoll, f, ensure_ascii=False, indent=2)

        # 2. Download Raw XML & Extract Speeches
        fundstelle = protokoll.get("fundstelle", {})
        xml_url = fundstelle.get("xml_url") or (
            fundstelle.get("pdf_url", "").replace(".pdf", ".xml")
        )

        if xml_url:
            try:
                xml_filepath = os.path.join(xml_dir, f"{datum}_{safe_protocol_id}.xml")
                with open(xml_filepath, "wb") as f:
                    f.write(fetch_xml_content(xml_url))

                speeches = extract_speeches_from_xml(xml_url, datum, protocol_id, wp)

                if speeches:
                    speech_filepath = os.path.join(
                        speech_dir, f"{datum}_{safe_protocol_id}_speeches.json"
                    )
                    with open(speech_filepath, "w", encoding="utf-8") as f:
                        json.dump(speeches, f, ensure_ascii=False, indent=2)
                    print(
                        f"[SUCCESS] Processed {protocol_id} ({datum}): {len(speeches)} speeches extracted."
                    )
                else:
                    print(
                        f"[WARNING] {protocol_id}: XML downloaded but no speeches extracted."
                    )

            except Exception as e:
                print(f"[ERROR] Pipeline failed for {protocol_id}: {e}")
        else:
            print(f"[WARNING] No XML URL found for {protocol_id}. Metadata saved.")


def get_person_details(firstname, lastname, legislative_period):
    """
    Fetches persons metadata for a specific date range or name, utilizing exponential backoff.
    returns
    [{'id': '8120', 'funktion': ['Bundesmin.'], 'ressort': ['Bundesministerium der Finanzen'], 'nachname': 'Klingbeil', 'vorname': 'Lars', 'typ': 'Person', 'wahlperiode': [15, 17, 18, 19, 20, 21], 'aktualisiert': '2025-05-09T09:45:30+02:00', 'person_roles': [{'fraktion': 'SPD', 'nachname': 'Klingbeil', 'vorname': 'Lars', 'wahlperiode_nummer': [15, 17, 18, 19, 20, 21]}], 'titel': 'Lars Klingbeil, Bundesmin., Bundesministerium der Finanzen', 'datum': '2026-02-26', 'basisdatum': '2005-02-22'}]
    """
    url = f"{BASE_URL}/person"
    persons = []

    params = {
        "f.wahlperiode": legislative_period,
        "apikey": API_KEY,
    }

    if firstname and lastname:
        params["f.person"] = f"{lastname}, {firstname}"

    while True:
        print(f"Fetching protocol list... (Found {len(persons)} so far)")

        try:
            data = fetch_api_json(url, params)
        except Exception as e:
            print(f"Failed to fetch persons after retries: {e}")
            break

        persons.extend(data.get("documents", []))
        persons_found = data.get("numFound", 0)

        if len(persons) >= persons_found:
            break

        cursor = data.get("cursor")
        params = {"cursor": cursor, "apikey": API_KEY}
    return persons


# Example Execution
if __name__ == "__main__":
    # print(get_person_details("Lars", "klingbeil", 21))
    # speaker_registry = SpeakerRegistry()
    # party = speaker_registry.get_party("Dobrindt", "Alexander", 21)
    # print(party)
    START_DATE = "2026-01-01"
    END_DATE = "2026-01-21"
    download_and_organize_protocols(START_DATE, END_DATE)
    END_DATE = "2026-01-21"
    download_and_organize_protocols(START_DATE, END_DATE)
