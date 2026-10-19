"""Run one example BiasPredictor call on OpenRouter and project the cost for N tests.

Usage:
    python scripts/estimate_token_cost.py            # RAG mode, 775 tests
    python scripts/estimate_token_cost.py --no-rag
    python scripts/estimate_token_cost.py --runs 3 --n-tests 775

Uses OPENROUTER_API_KEY_ME from .env.local. The example input is taken from an old
batch run log (run eval_matrix_20260825_102140_qwen3, text_index 577).
"""

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rag.evaluator import BiasPredictor  # noqa: E402  (also loads .env.local)

MODEL_ID = "mistralai/mistral-medium-3.1"

EXAMPLE_TEXT = (
    "Eine neue #Abwrackprämie wäre sozial ungerecht,  ökonomisch sinnlos und für die Umwelt schädlich. "
    "Statt #BMW,  #Daimler &amp; #VW die Profite zu sichern,  sollte sich #Merkel darum kümmern,  "
    "dass nicht mehr Millionen Menschen unter Existenzängsten leiden müssen!"
)

EXAMPLE_CHUNKS = [
    {
        "text": "Herr Abgeordneter, vielen Dank für das Lob für die Geduld. Aber das hier ist der Bundestag; hier leben wir von dem guten Argument. Ich denke, das sollten wir alle gemeinsam weiter pflegen. Zur Frage der Abwrackprämie. Eine Abwrackprämie, wie wir sie 2009 hatten, kann es eins zu eins nicht wieder geben. Davon bin ich fest überzeugt. Es muss ein zielgerichtetes Programm geben, das hilft, den Neustart in der Automobilindustrie wirklich voranzubringen, und das demzufolge hilft, Fahrzeuge, die umweltfreundlich sind, die einen geringen CO2-Ausstoß oder am besten gar keinen haben, stärker auf den Markt zu bringen. Das ist nicht ganz trivial, wenn so viele Menschen, wie wir das im Moment erleben, in Kurzarbeit sind und über andere Dinge nachdenken als darüber, sich ein neues Auto anzuschaffen. Deswegen muss das, was wir vorhaben, wirklich zielgerichtet sein. Deswegen kann man nicht mal eben so einen Schnellschuss machen und sagen: Das ist es. – Vielmehr müssen wir uns das genau angucken: Wer kann eigentlich kaufen? Was würde wirklich helfen, einen Neustart zu unterstützen? Was hilft auch im internationalen Vergleich? Es muss auch europäisch miteinander abgestimmt werden. Das alles zusammen ist nicht ganz trivial. Deswegen wird das ausgearbeitet. [I-1] Danke sehr. – Nachfrage?",
        "party": "SPD",
        "country": "Germany",
        "speaker": "Svenja Schulze",
        "date": "2020-05-06",
    },
    {
        "text": "Abwrackprämie“. Wir haben in der Finanz- und Wirtschaftskrise gesehen, dass die Abwrackprämie durchaus kontraproduktiv gewesen ist für die Emissionen im Verkehrssektor, Stichwort „größere, verbrauchsstärkere Autos“. Deswegen wäre es aus unserer Sicht erstens unsozial, eine Abwrackprämie einzuführen, aber auch unökologisch, weil es viele falsche Kaufanreize setzen würde, insbesondere wenn man Diesel-Pkw und Benziner, wie es von der Autolobby gefordert wird, miteinbezieht. Deswegen unsere Frage: Wäre es nicht sinnvoll, anstelle einer Abwrackprämie andere Maßnahmen in Richtung einer Transformation des Automobilsektors vorzunehmen und zu fördern?",
        "party": "DIE LINKE",
        "country": "Germany",
        "speaker": "Lorenz Gösta Beutin",
        "date": "2020-05-06",
    },
    {
        "text": "Produktion von E-Autos, [I-20] aber eben auch verstärkt in der Produktion von Bussen, Straßenbahnen, Zügen und digitalisierter Verkehrssteuerung. [I-21] Es ist deshalb völlig richtig, wenn Sie sagen, dass fossile Verbrenner nicht gefördert werden dürfen. [I-22] Es erscheint mir aber, dass Sie da noch Klärungsbedarf mit Ihrem Ministerpräsidenten in meinem Bundesland Baden-Württemberg haben. [I-23] Der fordert unverdrossen Abwrackprämien für Diesel und Benziner. Das nutzt übrigens auch den Beschäftigten herzlich wenig. Das ist Strohfeuerpolitik und hat mit Nachhaltigkeit nichts zu tun. [I-24] Wir können die Lebensqualität der Mehrheit der Bevölkerung verbessern, durch Investitionen in Krankenhäuser und Altenheime, in Kitas und Schulen, durch mehr bezahlbaren Wohnraum in öffentlichem Eigentum. Wir können auch durch ein gutes Investitionsprogramm das Klima schützen und sinnvolle und sichere Arbeit fördern. [I-25] Die himmelschreiende Ungleichheit bei Löhnen und Arbeitszeiten, die sich in den letzten Jahren entwickelt hat, [I-26] muss dabei auch überwunden werden. [I-27] Eines muss ich noch sagen: Bei der Finanzierung sind Sie von den Grünen wirklich schwach auf der Brust; da hätte ich mehr erwartet. Es müssen doch endlich einmal die Reichen und Vermögenden zur Finanzierung herangezogen werden. [I-28] Wir brauchen eine Vermögensabgabe. [I-29] Herr Kollege, kommen Sie bitte zum Schluss. Wir brauchen den Einstieg in eine Vermögensteuer. Wer dazu nicht den Mut hat, wird kaum den nötigen sozial-ökologischen Wechsel herbeiführen können. Wir haben den Mut dazu.",
        "party": "DIE LINKE",
        "country": "Germany",
        "speaker": "Bernd Riexinger",
        "date": "2020-05-29",
    },
    {
        "text": "halbherzigen Verteidigungsreden heute ist mir noch etwas aufgefallen: Zu der Sache, die das eigentliche Anliegen, der eigentliche Grund für das Führen dieser Debatte war, sagt keiner mehr was; [I-5] denn mittlerweile haben, glaube ich, alle eingesehen, dass der eigentliche Grund – eine Maut, die offensichtlich nur Ausländer diskriminieren soll – von Anfang an eine Schnapsidee war. [I-6] Die CSU mit ihrer Schnapsidee der Maut hat die Bürger bereits fast 80 Millionen Euro gekostet. Dieses Geld – das weiß man heute schon – ist sicher weg. Höchstwahrscheinlich kostet das Ganze noch viele Millionen weitere Euro. Auch dazu hätte ich heute gerne etwas von Ihnen gehört. Ich glaube, man muss kein Schwabe sein, um sich vorzustellen, wie man das Geld sinnvoll für dringende Projekte in dieser Republik hätte einsetzen können. [I-7] Was mir am meisten Sorge in dieser Debatte bereitet: Während wir über das Thema reden, schreitet die Klimakrise voran. Während wir Debatten über das Geschäftsmodell Auto, das für viele Regionen in Deutschland konstitutiv ist, weil es ein wesentlicher Wirtschaftsfaktor ist, über die Automobilwirtschaft, die vor dramatischen Umbrüchen steht, und über die Beschäftigten, die sich Sorgen um ihre Arbeitsplätze machen, weil es dramatische Umbrüche gibt – die Stichworte kennen wir alle: automatisiertes Fahren, vernetztes Fahren und zunehmend auch emissionsfreies Fahren –, führen müssten, beschäftigt sich das CSU-geführte Verkehrsministerium mit der von ihm selbst angerichteten absurden Maut. Das richtet Schaden bei Jobs und für den Standort Deutschland an. [I-8] Was wir jetzt brauchen, ist ein Verkehrsminister, der den Stier bei den Hörnern packt, sich an die Spitze der Transformation stellt, Wirtschaft, Beschäftigte und Klimaschutz zusammenbringt und dafür sorgt, dass wir mit all den Partnern zusammen die Verkehrswende gestalten. Es ist doch bemerkenswert: Der Chef von Volkswagen, Herbert Diess, kündigt an, dass VW auf E-Mobilität setzt. Was fällt dem",
        "party": "BÜNDNIS 90/DIE GRÜNEN",
        "country": "Germany",
        "speaker": "Cem Özdemir",
        "date": "2020-10-08",
    },
    {
        "text": "[I-4] – Na Sie. [I-5] Und ich sehe es auch als einen großen Erfolg der Klimabewegung an, dass es keine Prämie für fossile Verbrenner gab. [I-6] Da kann man insbesondere sagen: „Druck bewirkt etwas“, und ich kann nur hoffen, dass dieser Lernfortschritt ein dauerhafter Lernfortschritt ist. [I-7] Liebe Kolleginnen und Kollegen, bei 130 Milliarden wird allerdings umso drastischer sichtbar, was fehlt. Wir geben Milliarden für die Rettung der Lufthansa aus, auch Milliarden für die Rettung der Autoindustrie, und es ist auch richtig, dass wir sie retten; [I-8] aber für die Ärmsten gibt es in dieser Krise noch nicht einmal einen temporären Aufschlag auf Hartz IV. Für die Familien gibt es einen einmaligen Bonus, [I-9] aber kein dauerhaftes Coronaelterngeld, und die Soloselbstständigen – damit die Künstlerinnen und Künstler, Musiklehrerinnen und Musiklehrer, Therapeutinnen – haben Sie wieder vergessen; sie bekommen keinen Zuschuss zu ihrem Lebensunterhalt. [I-10] Und, liebe Kolleginnen und Kollegen, wenn wir doch Milliarden für Konzerne haben – und das auch in Teilen richtig ist –, dann sollten wir auch Milliarden für die Ärmsten haben. Denn sonst führt das zu einer sozialen Schieflage, und damit gefährden Sie den Zusammenhalt unserer Gesellschaft, und das ist falsch in dieser Krise. [I-11] Wenn ich mir in dem Paket Ihre Maßnahmen zum Klimaschutz und zur Gestaltung der Zukunft anschaue, dann fehlt es halt da wieder mal an Entschlossenheit. Man kann überhaupt sagen: Wenn es bei der Zukunftsgestaltung bei dieser Großen Koalition einen roten Faden gibt, dann ist es dieser: [I-12] dass Sie bei allen Maßnahmen nicht ausreichend entschlossen in die Zukunft investieren. [I-13] Bei 130 Milliarden wäre es ja auch ein Wunder, wenn nicht die eine oder andere Summe für richtige Maßnahmen dabei wäre. Aber sobald es beim Klimaschutz ernst wird, ist es halt zu wenig, zu spät und zu zögerlich. Nehmen wir zum Beispiel die Kfz-Steuer. Die Kfz-Steuerreform schützt nicht das Klima, sondern diese",
        "party": "BÜNDNIS 90/DIE GRÜNEN",
        "country": "Germany",
        "speaker": "Anton Hofreiter",
        "date": "2020-06-19",
    },
]


def fetch_pricing(model_id: str, api_key: str):
    """Return (usd_per_input_token, usd_per_output_token) from OpenRouter, or None."""
    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            models = json.load(resp)["data"]
        for m in models:
            if m["id"] == model_id:
                return float(m["pricing"]["prompt"]), float(m["pricing"]["completion"])
    except Exception as e:
        print(f"(could not fetch pricing from OpenRouter: {e})")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1, help="repeat the call N times")
    parser.add_argument("--n-tests", type=int, default=775)
    parser.add_argument("--no-rag", action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY_ME")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY_ME not set (expected in .env.local)")

    predictor = BiasPredictor(api_key=api_key)

    # predict_bias discards response.usage, so record it by wrapping the client call.
    usages = []
    original_create = predictor.client.chat.completions.create

    def recording_create(*a, **kw):
        response = original_create(*a, **kw)
        usages.append(response.usage)
        return response

    predictor.client.chat.completions.create = recording_create

    is_rag = not args.no_rag
    result = None
    for _ in range(args.runs):
        result = predictor.predict_bias(
            text=EXAMPLE_TEXT,
            model_id=MODEL_ID,
            context_chunks=EXAMPLE_CHUNKS if is_rag else None,
            is_rag_mode=is_rag,
        )

    print(f"Model: {MODEL_ID} | RAG: {is_rag} | chunks: {len(EXAMPLE_CHUNKS) if is_rag else 0}")
    print("Output:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if not usages:
        sys.exit("No usage recorded (the API call failed, see output above).")

    avg_in = sum(u.prompt_tokens for u in usages) / len(usages)
    avg_out = sum(u.completion_tokens for u in usages) / len(usages)
    print(f"\nTokens per run (avg over {len(usages)}): prompt={avg_in:.0f}, completion={avg_out:.0f}")
    print(
        f"Projected for {args.n_tests} tests: prompt={avg_in * args.n_tests:,.0f}, "
        f"completion={avg_out * args.n_tests:,.0f}"
    )

    pricing = fetch_pricing(MODEL_ID, api_key)
    if pricing:
        p_in, p_out = pricing
        per_run = avg_in * p_in + avg_out * p_out
        print(
            f"Pricing: ${p_in * 1e6:.3f}/M in, ${p_out * 1e6:.3f}/M out\n"
            f"Cost per run: ${per_run:.5f} | for {args.n_tests} tests: ${per_run * args.n_tests:.2f}"
        )


if __name__ == "__main__":
    main()
