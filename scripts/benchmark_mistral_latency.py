#!/usr/bin/env python3
"""
Benchmark script for measuring Mistral API response latency.

This script tests multiple models with and without examples to assess
how prompt length and model size affect latency.

Usage:
    python scripts/benchmark_mistral_latency.py
    python scripts/benchmark_mistral_latency.py -n 5  # 5 calls per configuration
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kiso_input.config import MISTRAL_API_KEY
from kiso_input.processing.cloud_apis import generate_summary_with_mistral

# Models to benchmark
MODELS = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
]

# Prompt components
SYSTEM_PROMPT = """# Rolle und Aufgabe

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

- Länge: Maximal 50 Wörter"""

EXAMPLES = """

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

ZUSAMMENFASSUNG: Du beschreibst, wie wichtig es dir ist, in deinen Erlebnissen gesehen und wertgeschätzt zu werden – ob durch enttäuschte Stille oder warme Bestätigung, die deine Freude noch strahlen lässt. Beim Zuhören trotz Ablenkung setzt du auf achtsame Strategien: bewusste Atmung, geduldiges Nachfragen und das sanfte Verankern in der Stimme deines Gegenübers, um präsent zu bleiben, ohne dich unter Druck zu setzen. Das zeigt viel Einfühlungsvermögen und Kraft."""

CONTENT = """

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


@dataclass
class BenchmarkResult:
    """Results from a single benchmark configuration."""
    model: str
    with_examples: bool
    prompt_length: int
    latencies: List[float]
    response_lengths: List[int]
    errors: List[str]
    
    @property
    def mean_latency(self) -> float:
        return mean(self.latencies) if self.latencies else 0.0
    
    @property
    def median_latency(self) -> float:
        return median(self.latencies) if self.latencies else 0.0
    
    @property
    def min_latency(self) -> float:
        return min(self.latencies) if self.latencies else 0.0
    
    @property
    def max_latency(self) -> float:
        return max(self.latencies) if self.latencies else 0.0
    
    @property
    def std_latency(self) -> float:
        return stdev(self.latencies) if len(self.latencies) >= 2 else 0.0
    
    @property
    def success_rate(self) -> float:
        total = len(self.latencies) + len(self.errors)
        return len(self.latencies) / total if total > 0 else 0.0


def build_prompt(with_examples: bool) -> str:
    """Build the prompt with or without examples."""
    if with_examples:
        return SYSTEM_PROMPT + EXAMPLES + CONTENT
    else:
        return SYSTEM_PROMPT + CONTENT


def run_single_benchmark(
    api_key: str,
    model: str,
    prompt: str,
    num_calls: int,
    with_examples: bool,
) -> BenchmarkResult:
    """Run benchmark for a single configuration."""
    latencies: List[float] = []
    response_lengths: List[int] = []
    errors: List[str] = []
    
    example_label = "with examples" if with_examples else "without examples"
    print(f"\n{'─' * 60}")
    print(f"📊 {model} ({example_label})")
    print(f"   Prompt: {len(prompt):,} chars")
    print(f"{'─' * 60}")
    
    for i in range(num_calls):
        call_num = i + 1
        print(f"   Call {call_num}/{num_calls}...", end=" ", flush=True)
        
        try:
            start_time = time.perf_counter()
            response = generate_summary_with_mistral(
                prompt=prompt,
                api_key=api_key,
                model=model,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
            )
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            latencies.append(latency)
            response_lengths.append(len(response))
            
            print(f"✅ {latency:.2f}s ({len(response)} chars)")
            
        except Exception as e:
            print(f"❌ {e}")
            errors.append(str(e))
    
    result = BenchmarkResult(
        model=model,
        with_examples=with_examples,
        prompt_length=len(prompt),
        latencies=latencies,
        response_lengths=response_lengths,
        errors=errors,
    )
    
    if latencies:
        print(f"   → Mean: {result.mean_latency:.2f}s | "
              f"Median: {result.median_latency:.2f}s | "
              f"Min: {result.min_latency:.2f}s | "
              f"Max: {result.max_latency:.2f}s")
    
    return result


def print_comparison_table(results: List[BenchmarkResult]) -> None:
    """Print a comparison table of all results."""
    print("\n" + "=" * 90)
    print("📊 COMPARISON TABLE")
    print("=" * 90)
    
    # Header
    print(f"\n{'Model':<25} {'Examples':<10} {'Prompt':<10} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'StdDev':<10}")
    print("-" * 95)
    
    for r in results:
        example_str = "Yes" if r.with_examples else "No"
        if r.latencies:
            print(f"{r.model:<25} {example_str:<10} {r.prompt_length:<10,} "
                  f"{r.mean_latency:<10.2f} {r.median_latency:<10.2f} "
                  f"{r.min_latency:<10.2f} {r.max_latency:<10.2f} {r.std_latency:<10.2f}")
        else:
            print(f"{r.model:<25} {example_str:<10} {r.prompt_length:<10,} {'FAILED':<40}")


def print_analysis(results: List[BenchmarkResult]) -> None:
    """Print analysis of prompt length and model size effects."""
    print("\n" + "=" * 90)
    print("📈 ANALYSIS: Effect of Prompt Length and Model Size")
    print("=" * 90)
    
    # Group by model
    models_data: Dict[str, Dict[str, BenchmarkResult]] = {}
    for r in results:
        if r.model not in models_data:
            models_data[r.model] = {}
        key = "with_examples" if r.with_examples else "without_examples"
        models_data[r.model][key] = r
    
    # Effect of examples (prompt length) per model
    print("\n🔍 Effect of Examples (Prompt Length):")
    print("-" * 60)
    for model in MODELS:
        if model in models_data:
            data = models_data[model]
            if "with_examples" in data and "without_examples" in data:
                with_ex = data["with_examples"]
                without_ex = data["without_examples"]
                if with_ex.latencies and without_ex.latencies:
                    diff = with_ex.mean_latency - without_ex.mean_latency
                    pct = (diff / without_ex.mean_latency) * 100 if without_ex.mean_latency > 0 else 0
                    prompt_diff = with_ex.prompt_length - without_ex.prompt_length
                    print(f"  {model}:")
                    print(f"    Without examples: {without_ex.mean_latency:.2f}s ({without_ex.prompt_length:,} chars)")
                    print(f"    With examples:    {with_ex.mean_latency:.2f}s ({with_ex.prompt_length:,} chars)")
                    print(f"    Difference:       +{diff:.2f}s (+{pct:.1f}%) for +{prompt_diff:,} chars")
                    print()
    
    # Effect of model size (comparing same prompt type)
    print("\n🔍 Effect of Model Size (with examples):")
    print("-" * 60)
    with_examples_results = [r for r in results if r.with_examples and r.latencies]
    if len(with_examples_results) >= 2:
        baseline = with_examples_results[0]
        print(f"  Baseline: {baseline.model} = {baseline.mean_latency:.2f}s")
        for r in with_examples_results[1:]:
            diff = r.mean_latency - baseline.mean_latency
            pct = (diff / baseline.mean_latency) * 100 if baseline.mean_latency > 0 else 0
            print(f"  {r.model}: {r.mean_latency:.2f}s ({diff:+.2f}s, {pct:+.1f}%)")
    
    print("\n🔍 Effect of Model Size (without examples):")
    print("-" * 60)
    without_examples_results = [r for r in results if not r.with_examples and r.latencies]
    if len(without_examples_results) >= 2:
        baseline = without_examples_results[0]
        print(f"  Baseline: {baseline.model} = {baseline.mean_latency:.2f}s")
        for r in without_examples_results[1:]:
            diff = r.mean_latency - baseline.mean_latency
            pct = (diff / baseline.mean_latency) * 100 if baseline.mean_latency > 0 else 0
            print(f"  {r.model}: {r.mean_latency:.2f}s ({diff:+.2f}s, {pct:+.1f}%)")


def run_benchmark(num_calls: int = 5):
    """Run the complete latency benchmark across all configurations."""
    
    # Check for API key
    api_key = MISTRAL_API_KEY
    if not api_key:
        api_key = os.environ.get("MISTRAL_API_KEY")
    
    if not api_key:
        print("❌ Error: MISTRAL_API_KEY not found!")
        print("Please set it in your .env file or as an environment variable.")
        sys.exit(1)
    
    # Build prompts
    prompt_with_examples = build_prompt(with_examples=True)
    prompt_without_examples = build_prompt(with_examples=False)
    
    print("=" * 90)
    print("🚀 Mistral API Latency Benchmark - Model & Prompt Length Comparison")
    print("=" * 90)
    print(f"\nModels to test: {', '.join(MODELS)}")
    print(f"Calls per configuration: {num_calls}")
    print(f"Prompt with examples: {len(prompt_with_examples):,} chars")
    print(f"Prompt without examples: {len(prompt_without_examples):,} chars")
    print(f"Total configurations: {len(MODELS) * 2}")
    print(f"Total API calls: {len(MODELS) * 2 * num_calls}")
    print(f"\nSettings: max_tokens=200, temperature=0.7, top_p=0.9")
    
    results: List[BenchmarkResult] = []
    
    # Run benchmarks for each model and prompt configuration
    for model in MODELS:
        # Without examples (shorter prompt)
        result = run_single_benchmark(
            api_key=api_key,
            model=model,
            prompt=prompt_without_examples,
            num_calls=num_calls,
            with_examples=False,
        )
        results.append(result)
        
        # With examples (longer prompt)
        result = run_single_benchmark(
            api_key=api_key,
            model=model,
            prompt=prompt_with_examples,
            num_calls=num_calls,
            with_examples=True,
        )
        results.append(result)
    
    # Print summary
    print_comparison_table(results)
    print_analysis(results)
    
    print("\n" + "=" * 90)
    print("✅ Benchmark complete!")
    print("=" * 90)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark Mistral API latency across models and prompt sizes"
    )
    parser.add_argument(
        "-n", "--num-calls",
        type=int,
        default=5,
        help="Number of API calls per configuration (default: 5)"
    )
    
    args = parser.parse_args()
    run_benchmark(num_calls=args.num_calls)
