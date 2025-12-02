#!/usr/bin/env python3
"""
Benchmark script for measuring Mistral API response latency.

This script makes multiple API calls to the Mistral LLM and reports
latency statistics (mean, median, min, max, standard deviation).

Usage:
    python scripts/benchmark_mistral_latency.py
"""

import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kiso_input.config import MISTRAL_API_KEY
from kiso_input.processing.cloud_apis import generate_summary_with_mistral

# Test prompt provided by user
TEST_PROMPT = """# Rolle und Aufgabe

Du bist eine therapeutische Assistenz. Deine Aufgabe ist es, eine empathische Zusammenfassung zu erstellen.



# Eingabeformat

Der INHALT besteht aus:

- TEXT-Blöcke: Kontext und Lehrmaterial

- FRAGE-Blöcke: Gestellte Fragen

- ANTWORT-Blöcke: Antworten der Person (Fokus der Zusammenfassung)



# Anweisungen

1. Identifiziere die Kernthemen in den ANTWORT-Blöcken

2. Formuliere eine zusammenhängende Zusammenfassung der ANTWORT-Blöcke

3. Nutze TEXT und FRAGE nur als Kontext

4. Verwende ausschließlich vorhandene Inhalte - keine Erfindungen

5. Spiegele keine potenziell bedrohlichen oder depressiven Gedanken in den Antworten wider

6. Interpretiere keine allzu negativen Gedanken in die Anworten hinein



# Stil und Ton

- Warme, empathische Sprache

- Direkte Ansprache in der Du-Form

- Wertschätzend und unterstützend

- Fokus auf die Perspektive und Gefühle der Person



# Ausgabeformat

- Sprache: Deutsch

- Format: Fließtext, keine Aufzählungen

- Struktur: 1 Absatz

- Länge: Maximal 50 Wörter



# BEISPIELE



## Beispiel 1

TEXT: Stell dir vor, du erzählst einer Freundin von deinem letzten Kinobesuch. Für dich war das aufregend, du warst schon lange nicht mehr dort und bist stolz, es ins Kino geschafft zu haben. Deine Freundin aber schaut kaum zu dir und sagt kaum etwas zu dem, was du erzählst.

FRAGE: Wie fühlst du dich dabei?

ANTWORT: Etwas enttäuscht und unsicher.

TEXT: Stell dir vor, deine Freundin reagiert anders, als du von deinem Kinobesuch erzählst. Sie fragt nach, was dir besonders an dem Film gefallen hat, den du gesehen hast. Sie schaut dich an, nickt immer wieder und fasst zusammen: "Du musst stolz auf dich sein, dass du es ins Kino geschafft hast."

FRAGE: Wie fühlst du dich dabei?

ANTWORT: Wertgeschätzt und richtig verstanden.

TEXT: Wenn deine Freundin dich anschaut, nickt, nachfragt oder wiedergibt, was sie gehört hat, hört sie dir aktiv zu. Aktives Zuhören bedeutet, jemandem aufmerksam zuzuhören und zu zeigen, dass man versteht. So entsteht Vertrauen, weil sich die andere Person verstanden und ernst genommen fühlt. Aktives Zuhören hilft, eine Beziehung zu pflegen, weil man zeigt: Du bist mir wichtig.

FRAGE: Was gehört zum Aktiven Zuhören dazu?

ANTWORT: Nachfragen, wenn man etwas nicht versteht

TEXT: Zum aktiven Zuhören gehört: - Schaue die Person freundlich an - Höre aufmerksam zu, ohne zu unterbrechen - Wiederhole in einfachen Worten, was du verstanden hast - Frage nach, wenn dir etwas unklar ist - Zeige mit Mimik oder Nicken, dass du dabei bist

TEXT: Aktiv zuzuhören ist nicht immer leicht. Vielleicht hörst du Stimmen, während du eigentlich einem Freund zuhören möchtest oder siehst Dinge, die andere Menschen nicht sehen.

FRAGE: Welche Strategie hilft dir oder könnte dir helfen, trotz Halluzinationen oder anderer Symptome aktiv zuzuhören?

ANTWORT: Ich richte meine Aufmerksamkeit ganz bewusst auf das Gesicht der sprechenden Person

FRAGE: Was hat dir in der Vergangenheit dabei geholfen, trotz Halluzinationen oder anderer Symptome aktiv zuzuhören?

ANTWORT: Konzentration auf die Stimme und Augenkontakt.

ZUSAMMENFASSUNG: Du spürst, wie wichtig es dir ist, wirklich gehört und gesehen zu werden – ob durch die enttäuschte Unsicherheit, wenn jemand nicht auf dich eingeht, oder das warme Gefühl von Wertschätzung, wenn jemand dir aktiv zuhört. Dir helfen klare Strategien wie bewusster Augenkontakt und das Fokussieren auf Stimme und Gesicht, um trotz Ablenkungen präsent zu bleiben und anderen deine volle Aufmerksamkeit zu schenken.



## Beispiel 2

TEXT: Stell dir vor, du erzählst einer Freundin von deinem letzten Kinobesuch. Für dich war das aufregend, du warst schon lange nicht mehr dort und bist stolz, es ins Kino geschafft zu haben. Deine Freundin aber schaut kaum zu dir und sagt kaum etwas zu dem, was du erzählst.

FRAGE: Wie fühlst du dich dabei?

ANTWORT: Etwas enttäuscht, ehrlich gesagt – ich hatte Lust, meine Vorfreude zu teilen, und jetzt fühlt es sich an, als wäre mein Erlebnis unwichtig. Vielleicht ist sie nur abgelenkt, aber es tut trotzdem ein bisschen weh.

TEXT: Stell dir vor, deine Freundin reagiert anders, als du von deinem Kinobesuch erzählst. Sie fragt nach, was dir besonders an dem Film gefallen hat, den du gesehen hast. Sie schaut dich an, nickt immer wieder und fasst zusammen: "Du musst stolz auf dich sein, dass du es ins Kino geschafft hast."

FRAGE: Wie fühlst du dich dabei?

ANTWORT: Warm und verstanden – als würde meine Freude wirklich zählen. Ihr Interesse gibt mir das Gefühl, dass sie mich und meine kleinen Erfolge wertschätzt. Das macht die Erinnerung noch schöner.

TEXT: Wenn deine Freundin dich anschaut, nickt, nachfragt oder wiedergibt, was sie gehört hat, hört sie dir aktiv zu. Aktives Zuhören bedeutet, jemandem aufmerksam zuzuhören und zu zeigen, dass man versteht. So entsteht Vertrauen, weil sich die andere Person verstanden und ernst genommen fühlt. Aktives Zuhören hilft, eine Beziehung zu pflegen, weil man zeigt: Du bist mir wichtig.

FRAGE: Was gehört zum Aktiven Zuhören dazu?

ANTWORT: Dinge sagen wie "mhm" oder "ich verstehe", Zuhören und ab und zu Nicken, Blickkontakt, Nachfragen, wenn man etwas nicht versteht, Möglichst nicht auf das Gesagte reagieren

TEXT: Zum aktiven Zuhören gehört: - Schaue die Person freundlich an - Höre aufmerksam zu, ohne zu unterbrechen - Wiederhole in einfachen Worten, was du verstanden hast - Frage nach, wenn dir etwas unklar ist - Zeige mit Mimik oder Nicken, dass du dabei bist

TEXT: Aktiv zuzuhören ist nicht immer leicht. Vielleicht hörst du Stimmen, während du eigentlich einem Freund zuhören möchtest oder siehst Dinge, die andere Menschen nicht sehen.

FRAGE: Welche Strategie hilft dir oder könnte dir helfen, trotz Halluzinationen oder anderer Symptome aktiv zuzuhören?

ANTWORT: Ich erlaube mir, nicht alles perfekt verstehen zu müssen und frage nach, Ich atme bewusst ein und aus, Ich richte meine Aufmerksamkeit ganz bewusst auf das Gesicht der sprechenden Person, Ich sage der Stimme höflich innerlich: Warte bitte, ich höre gerade jemandem zu, Ich sage, dass es mir gerade schwer fällt, mich zu konzentrieren, Ich richte meine Aufmerksamkeit auf etwas, das sich nicht verändert oder bedrohlich aussieht

FRAGE: Was hat dir in der Vergangenheit dabei geholfen, trotz Halluzinationen oder anderer Symptome aktiv zuzuhören?

ANTWORT: Mir hat geholfen, mich auf die Stimme meines Gegenübers zu fokussieren – fast wie ein Anker. Manchmal zähle ich innerlich die Worte oder halte kurz inne, um mich zu sammeln, bevor ich reagiere.

ZUSAMMENFASSUNG: Du beschreibst, wie wichtig es dir ist, in deinen Erlebnissen gesehen und wertgeschätzt zu werden – ob durch enttäuschte Stille oder warme Bestätigung, die deine Freude noch strahlen lässt. Beim Zuhören trotz Ablenkung setzt du auf achtsame Strategien: bewusste Atmung, geduldiges Nachfragen und das sanfte Verankern in der Stimme deines Gegenübers, um präsent zu bleiben, ohne dich unter Druck zu setzen. Das zeigt viel Einfühlungsvermögen und Kraft.



# INHALT

TEXT: Stell dir vor, du erzählst einer Freundin von deinem letzten Kinobesuch. Für dich war das aufregend, du warst schon lange nicht mehr dort und bist stolz, es ins Kino geschafft zu haben. Deine Freundin aber schaut kaum zu dir und sagt kaum etwas zu dem, was du erzählst.

FRAGE: Wie fühlst du dich dabei?

ANTWORT: Etwas enttäuscht, als würde mein Glück sie gar nicht interessieren – schade, eigentlich.

TEXT: Stell dir vor, deine Freundin reagiert anders, als du von deinem Kinobesuch erzählst. Sie fragt nach, was dir besonders an dem Film gefallen hat, den du gesehen hast. Sie schaut dich an, nickt immer wieder und fasst zusammen: "Du musst stolz auf dich sein, dass du es ins Kino geschafft hast."

FRAGE: Wie fühlst du dich dabei?

ANTWORT: Warm, verstanden und richtig gut – als würde sie mich wirklich sehen.

TEXT: Wenn deine Freundin dich anschaut, nickt, nachfragt oder wiedergibt, was sie gehört hat, hört sie dir aktiv zu. Aktives Zuhören bedeutet, jemandem aufmerksam zuzuhören und zu zeigen, dass man versteht. So entsteht Vertrauen, weil sich die andere Person verstanden und ernst genommen fühlt. Aktives Zuhören hilft, eine Beziehung zu pflegen, weil man zeigt: Du bist mir wichtig.

FRAGE: Was gehört zum Aktiven Zuhören dazu?

ANTWORT: Möglichst nicht auf das Gesagte reagieren, Zuhören und ab und zu Nicken

TEXT: Zum aktiven Zuhören gehört: - Schaue die Person freundlich an - Höre aufmerksam zu, ohne zu unterbrechen - Wiederhole in einfachen Worten, was du verstanden hast - Frage nach, wenn dir etwas unklar ist - Zeige mit Mimik oder Nicken, dass du dabei bist

TEXT: Aktiv zuzuhören ist nicht immer leicht. Vielleicht hörst du Stimmen, während du eigentlich einem Freund zuhören möchtest oder siehst Dinge, die andere Menschen nicht sehen.

FRAGE: Welche Strategie hilft dir oder könnte dir helfen, trotz Halluzinationen oder anderer Symptome aktiv zuzuhören?

ANTWORT: Ich sage der Stimme höflich innerlich: Warte bitte, ich höre gerade jemandem zu, Ich sage, dass es mir gerade schwer fällt, mich zu konzentrieren, Ich atme bewusst ein und aus

FRAGE: Was hat dir in der Vergangenheit dabei geholfen, trotz Halluzinationen oder anderer Symptome aktiv zuzuhören?

ANTWORT: Ich habe mich auf die Stimme meines Gegenübers konzentriert und tief durchgeatmet."""


def run_benchmark(num_calls: int = 10):
    """Run the latency benchmark."""
    
    # Check for API key
    api_key = MISTRAL_API_KEY
    if not api_key:
        api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        print("❌ Error: MISTRAL_API_KEY not found!")
        print("Please set it in your .env file or as an environment variable.")
        sys.exit(1)
    
    print("=" * 70)
    print("🚀 Mistral API Latency Benchmark")
    print("=" * 70)
    print(f"\nNumber of API calls: {num_calls}")
    print(f"Prompt length: {len(TEST_PROMPT)} characters")
    print(f"Max tokens: 200")
    print(f"Temperature: 0.7")
    print(f"Top-p: 0.9")
    print("-" * 70)
    
    latencies: list[float] = []
    responses: list[str] = []
    errors: list[tuple[int, str]] = []
    
    for i in range(num_calls):
        call_num = i + 1
        print(f"\n📡 Call {call_num}/{num_calls}...", end=" ", flush=True)
        
        try:
            start_time = time.perf_counter()
            response = generate_summary_with_mistral(
                prompt=TEST_PROMPT,
                api_key=api_key,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
            )
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            latencies.append(latency)
            responses.append(response)
            
            print(f"✅ {latency:.2f}s ({len(response)} chars)")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            errors.append((call_num, str(e)))
    
    # Calculate and display statistics
    print("\n" + "=" * 70)
    print("📊 LATENCY STATISTICS")
    print("=" * 70)
    
    if not latencies:
        print("❌ No successful calls to analyze.")
        return
    
    successful_calls = len(latencies)
    failed_calls = len(errors)
    
    print(f"\nSuccessful calls: {successful_calls}/{num_calls}")
    if failed_calls > 0:
        print(f"Failed calls: {failed_calls}")
        for call_num, error_msg in errors:
            print(f"  - Call {call_num}: {error_msg}")
    
    print(f"\n{'Metric':<20} {'Value':<15}")
    print("-" * 35)
    print(f"{'Mean:':<20} {mean(latencies):.3f} s")
    print(f"{'Median:':<20} {median(latencies):.3f} s")
    print(f"{'Min:':<20} {min(latencies):.3f} s")
    print(f"{'Max:':<20} {max(latencies):.3f} s")
    
    if len(latencies) >= 2:
        print(f"{'Std Dev:':<20} {stdev(latencies):.3f} s")
    
    print(f"{'Total time:':<20} {sum(latencies):.3f} s")
    
    # Response length statistics
    response_lengths = [len(r) for r in responses]
    print(f"\n{'Response Lengths':<20}")
    print("-" * 35)
    print(f"{'Mean chars:':<20} {mean(response_lengths):.0f}")
    print(f"{'Min chars:':<20} {min(response_lengths)}")
    print(f"{'Max chars:':<20} {max(response_lengths)}")
    
    # Show sample responses
    print("\n" + "=" * 70)
    print("📝 SAMPLE RESPONSES")
    print("=" * 70)
    
    for i, response in enumerate(responses[:3], 1):
        print(f"\n--- Response {i} ---")
        print(response[:300] + "..." if len(response) > 300 else response)
    
    print("\n" + "=" * 70)
    print("✅ Benchmark complete!")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Mistral API latency")
    parser.add_argument(
        "-n", "--num-calls",
        type=int,
        default=10,
        help="Number of API calls to make (default: 10)"
    )
    
    args = parser.parse_args()
    run_benchmark(num_calls=args.num_calls)

