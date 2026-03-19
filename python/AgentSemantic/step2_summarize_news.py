#!/usr/bin/env python3
"""
Step 2 of the pipeline:
Read the combined price/news CSV from step 1 and summarize the news text for
every trading day. Summaries are created with Sumy's LSA-based summarizer plus
an extra keyword weighting heuristic. The output CSV contains the original
columns plus the summary information.
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set

from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer
from sumy.utils import get_stop_words

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "step1_recent_price_and_news.csv"
DEFAULT_OUTPUT = REPO_ROOT / "step2_price_news_summaries.csv"
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
DEFAULT_LANGUAGE = "english"

@dataclass(frozen=True)
class SummaryResult:
    summary: str
    sentence_count: int


@dataclass(frozen=True)
class LsaContext:
    summarizer: LsaSummarizer
    tokenizer: Tokenizer


def sentence_tokenize(text: str) -> List[str]:
    sentences: List[str] = []
    for raw in SENTENCE_SPLIT.split(text):
        chunk = raw.strip()
        if chunk:
            sentences.append(chunk)
    if not sentences and text:
        sentences.append(text.strip())
    return sentences


def build_lsa_context(language: str) -> LsaContext:
    stemmer = Stemmer(language)
    summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(language)
    tokenizer = create_tokenizer(language)
    return LsaContext(summarizer=summarizer, tokenizer=tokenizer)


def create_tokenizer(language: str) -> Tokenizer:
    try:
        return Tokenizer(language)
    except LookupError:
        ensure_nltk_resources(language)
        return Tokenizer(language)


def ensure_nltk_resources(language: str) -> None:
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(
            "nltk is required for Sumy tokenization. Please install nltk."
        ) from exc

    resources = ["punkt", "punkt_tab"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            continue


def increase_weight_for_keywords(sentences: Sequence, keywords: Set[str]):
    weights = defaultdict(float)
    lowered_keywords = {kw.lower() for kw in keywords if kw}
    if not lowered_keywords:
        return weights
    for sentence in sentences:
        sentence_text = str(sentence).lower()
        for word in lowered_keywords:
            if word in sentence_text:
                weights[sentence] += 1.0
    return weights


def build_summary(
    text: str,
    num_sentences: int,
    keywords: Set[str],
    context: LsaContext,
) -> SummaryResult:
    normalized = " ".join(part.strip() for part in text.splitlines()).strip()
    if not normalized:
        return SummaryResult("", 0)

    parser = PlaintextParser.from_string(normalized, context.tokenizer)
    sentences = list(parser.document.sentences)
    if not sentences:
        return SummaryResult("", 0)

    initial_summary = list(context.summarizer(parser.document, num_sentences))
    weights = increase_weight_for_keywords(sentences, keywords)
    for sentence in initial_summary:
        weights[sentence] += 1.0

    ordered_sentences = sentences_with_weights(sentences, weights, num_sentences)
    if not ordered_sentences:
        fallback = [str(s).strip() for s in sentences[:num_sentences] if str(s).strip()]
        return SummaryResult(" ".join(fallback), len(fallback))

    chosen_text = [" ".join(str(sentence).split()) for sentence in ordered_sentences]
    chosen_text = [text for text in chosen_text if text]
    return SummaryResult(" ".join(chosen_text), len(chosen_text))


def sentences_with_weights(
    sentences: Sequence, weights: defaultdict, num_sentences: int
) -> List:
    if not weights:
        return []
    indexed = {sentence: idx for idx, sentence in enumerate(sentences)}
    ranked = sorted(
        weights.items(),
        key=lambda item: (-item[1], indexed.get(item[0], len(sentences))),
    )
    chosen = [item[0] for item in ranked[:num_sentences]]
    chosen.sort(key=lambda sentence: indexed.get(sentence, 0))
    return chosen


def combine_fields(parts: Sequence[str]) -> str:
    cleaned: List[str] = []
    for part in parts:
        if not part:
            continue
        fragments = [frag.strip() for frag in part.split("||")]
        cleaned.extend(fragment for fragment in fragments if fragment)
    return " ".join(cleaned)


def summarize_row(
    row: dict,
    num_sentences: int,
    keywords: Set[str],
    context: LsaContext,
) -> SummaryResult:
    text = combine_fields([row.get("news_titles", ""), row.get("news_bodies", "")])
    return build_summary(
        text,
        num_sentences=num_sentences,
        keywords=keywords,
        context=context,
    )


def process_rows(
    rows: Iterable[dict],
    num_sentences: int,
    base_keywords: Sequence[str],
    context: LsaContext,
    show_lengths: bool = False,
) -> List[dict]:
    updated: List[dict] = []
    for row in rows:
        ticker = (row.get("ticker") or "").strip().lower()
        keywords = set()
        if len(ticker) >= 2:
            keywords.add(ticker)
        keywords.update(kw.lower() for kw in base_keywords if kw)
        summary = summarize_row(
            row,
            num_sentences=num_sentences,
            keywords=keywords,
            context=context,
        )
        if show_lengths:
            original_text = combine_fields(
                [row.get("news_titles", ""), row.get("news_bodies", "")]
            )
            orig_words = len(original_text.split())
            summary_words = len(summary.summary.split())
            print(
                f"{row.get('ticker','').strip()} {row.get('date','')}: "
                f"{orig_words} words -> {summary_words} words"
            )
        new_row = dict(row)
        new_row["news_summary"] = summary.summary
        new_row["summary_sentence_count"] = str(summary.sentence_count)
        updated.append(new_row)
    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize news text for each row.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the CSV generated by step 1.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination CSV for the summaries.",
    )
    parser.add_argument(
        "--sentences",
        type=int,
        default=3,
        help="Number of sentences to keep for each summary.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=DEFAULT_LANGUAGE,
        help="Language used for tokenization/stemming (default: english).",
    )
    parser.add_argument(
        "--keyword",
        action="append",
        default=[],
        help="Extra keyword to upweight in summaries (repeatable).",
    )
    parser.add_argument(
        "--show-lengths",
        action="store_true",
        help="Print original/summary word counts for debugging compression.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} was not found. Run step 1 first.")

    with args.input.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    num_sentences = max(args.sentences, 1)
    keywords = [kw.strip().lower() for kw in args.keyword if kw]
    context = build_lsa_context(args.language)
    updated_rows = process_rows(
        rows,
        num_sentences=num_sentences,
        base_keywords=keywords,
        context=context,
        show_lengths=args.show_lengths,
    )

    fieldnames: List[str] = list(reader.fieldnames or [])
    if "news_summary" not in fieldnames:
        fieldnames.append("news_summary")
    if "summary_sentence_count" not in fieldnames:
        fieldnames.append("summary_sentence_count")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)

    print(f"Wrote {len(updated_rows)} rows with summaries to {args.output}")


if __name__ == "__main__":
    main()
