import requests
from bs4 import BeautifulSoup
import time
import html
import pandas as pd

# TODO: I only got tweet ids, for their text content I need to scrape the data via oembed endpoint, since the API is not available for free anymore.
# First just load the all entries from the dataset which are from an eu country
# second create new csv with only the important data and merge everything together
# Then extract its content


def get_tweet_text_oembed(tweet_id):
    url = f"https://publish.twitter.com/oembed?url=https://x.com/i/status/{tweet_id}&omit_script=true"
    response = requests.get(url)

    if response.status_code == 200:
        html = response.json().get("html", "")
        soup = BeautifulSoup(html, "html.parser")
        # Extract main text from the first paragraph tag
        p = soup.find("p")
        return p.get_text() if p else ""
    elif response.status_code == 404:
        return "[Deleted or Private Tweet]"
    return None


def parse_oembed_response(oembed_data):
    soup = BeautifulSoup(oembed_data.get("html", ""), "html.parser")

    # 1. Tweet text is inside the <p> element
    p_tag = soup.find("p")
    tweet_text = html.unescape(p_tag.get_text()) if p_tag else ""

    # 2. Date is inside the anchor tag <a> inside the blockquote
    date_tag = soup.find("a")
    tweet_date = date_tag.get_text() if date_tag else ""

    return {
        "text": tweet_text,
        "author": oembed_data.get("author_name"),
        "date": tweet_date,
        "url": oembed_data.get("url"),
    }


# Example usage
tweet_ids = ["1352285621033840646"]
for t_id in tweet_ids:
    data = get_tweet_text_oembed(t_id)
    print(data)
    # print(parse_oembed_response(data))
    # time.sleep(0.5)  # Rate limiting safety pause


# parsed = parse_oembed_response(data)
# print(parsed)
# Output:
