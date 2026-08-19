from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

MAIN_COUNT = 5
MAIN_MAX = 43
BONUS_MAX = 16

_DRAW_LINE = re.compile(r"Date:\s*(\d{4}-\d{2}-\d{2}).*?Numbers:\s*([\d\s\-]+)")


@dataclass(frozen=True)
class Draw:
    date: str
    main: tuple[int, ...]
    bonus: int


def load_draws(vectorstore: Any) -> list[Draw]:
    """Parse every draw out of a Baloto vector store, newest first.

    The store is a parameter so this package never touches the app's Chroma wiring.
    """
    documents = vectorstore.get(include=["documents"]).get("documents", [])

    seen: set[tuple[str, tuple[int, ...]]] = set()
    draws: list[Draw] = []
    for page in documents:
        for date, raw in _DRAW_LINE.findall(page):
            numbers = tuple(int(n) for n in raw.split("-") if n.strip())
            if len(numbers) != MAIN_COUNT + 1 or (date, numbers) in seen:
                continue  # the dataset contains duplicated rows
            seen.add((date, numbers))
            draws.append(Draw(date=date, main=tuple(sorted(numbers[:MAIN_COUNT])), bonus=numbers[MAIN_COUNT]))

    draws.sort(key=lambda draw: draw.date, reverse=True)
    return draws


def _rank(counts: dict[int, float]) -> list[tuple[int, float]]:
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def count_frequencies(draws: list[Draw], half_life: float) -> dict[str, Any]:
    """Recency-weighted counts. weight = 0.5 ** (draws_ago / half_life); half_life <= 0 disables weighting."""
    positional: list[dict[int, float]] = [defaultdict(float) for _ in range(MAIN_COUNT)]
    overall: dict[int, float] = defaultdict(float)
    bonus: dict[int, float] = defaultdict(float)

    for index, draw in enumerate(draws):
        weight = 0.5 ** (index / half_life) if half_life > 0 else 1.0
        for position, number in enumerate(draw.main):
            positional[position][number] += weight
            overall[number] += weight
        bonus[draw.bonus] += weight

    return {
        "positional": [_rank(slot) for slot in positional],
        "overall": _rank(overall),
        "bonus": _rank(bonus),
    }


def count_gaps(draws: list[Draw]) -> dict[str, Any]:
    """Draws since each number last appeared. Numbers never seen in a slot are left out, not treated as overdue."""
    positional: list[dict[int, float]] = [{} for _ in range(MAIN_COUNT)]
    overall: dict[int, float] = {}
    bonus: dict[int, float] = {}

    for index, draw in enumerate(draws):  # index 0 is the most recent draw
        for position, number in enumerate(draw.main):
            positional[position].setdefault(number, float(index))
            overall.setdefault(number, float(index))
        bonus.setdefault(draw.bonus, float(index))

    return {
        "positional": [_rank(slot) for slot in positional],
        "overall": _rank(overall),
        "bonus": _rank(bonus),
    }


def _blend(hot: list[tuple[int, float]], cold: list[tuple[int, float]], cold_weight: float) -> list[tuple[int, float]]:
    """Mix two rankings after scaling both to 0-1; weighted counts and draw gaps are not comparable otherwise."""
    def normalised(ranked: list[tuple[int, float]]) -> dict[int, float]:
        scores = dict(ranked)
        if not scores:
            return {}
        low, high = min(scores.values()), max(scores.values())
        span = high - low or 1.0
        return {number: (score - low) / span for number, score in scores.items()}

    hot_scores, cold_scores = normalised(hot), normalised(cold)
    merged = {
        number: (1 - cold_weight) * hot_scores.get(number, 0.0) + cold_weight * cold_scores.get(number, 0.0)
        for number in hot_scores.keys() | cold_scores.keys()
    }
    return _rank(merged)


def _positional_ticket(positional: list[list[tuple[int, float]]]) -> list[int]:
    """Highest-ranked number per slot, kept unique and ascending."""
    picked: list[int] = []
    for slot in positional:
        floor = picked[-1] if picked else 0
        chosen = next((number for number, _ in slot if number > floor), None)
        if chosen is None:  # slot exhausted, fall back to the next free number
            chosen = next(number for number in range(floor + 1, MAIN_MAX + 1) if number not in picked)
        picked.append(chosen)
    return picked


def suggest_numbers(
    vectorstore: Any,
    window: int = 100,
    half_life: float = 50.0,
    strategy: str = "positional",
    pool_size: int = 3,
    cold_weight: float = 0.5,
) -> dict[str, Any]:
    """Deterministic ticket plus the candidate pool the caller may choose from.

    Args:
        vectorstore: Baloto vector store holding the draw history.
        window: How many of the most recent draws to count; 0 uses the whole history.
        half_life: Draws after which a draw's weight halves; 0 disables recency weighting.
        strategy: "positional" and "overall" rank by frequency, "cold" by draws since last seen, "hybrid" blends both.
        pool_size: How many alternatives to expose per slot.
        cold_weight: Share of the hybrid score taken from the cold ranking; 0 is pure hot, 1 is pure cold.
    """
    logger.info(
        "Baloto calculator | window=%s half_life=%s strategy=%s pool_size=%s cold_weight=%s",
        window,
        half_life,
        strategy,
        pool_size,
        cold_weight,
    )

    draws = load_draws(vectorstore)
    if not draws:
        logger.warning("Baloto calculator | no draws found in the vector store")
        return {"error": "No Baloto draws available in the vector store."}

    selected = draws[:window] if window > 0 else draws
    hot = count_frequencies(selected, half_life)

    if strategy == "cold":
        stats = count_gaps(selected)
        score_meaning = "draws since the number last appeared (higher is more overdue)"
    elif strategy == "hybrid":
        cold = count_gaps(selected)
        stats = {
            "positional": [_blend(h, c, cold_weight) for h, c in zip(hot["positional"], cold["positional"])],
            "overall": _blend(hot["overall"], cold["overall"], cold_weight),
            "bonus": _blend(hot["bonus"], cold["bonus"], cold_weight),
        }
        score_meaning = f"blended 0-1 score, {cold_weight:.0%} overdue and {1 - cold_weight:.0%} frequency"
    else:
        stats = hot
        score_meaning = "recency-weighted number of appearances (higher is hotter)"

    if strategy == "overall":
        main = sorted(number for number, _ in stats["overall"][:MAIN_COUNT])
        candidates: dict[str, list] = {
            "main": [[n, round(w, 3)] for n, w in stats["overall"][: MAIN_COUNT + pool_size]]
        }
    else:
        main = _positional_ticket(stats["positional"])
        candidates = {
            f"position_{position + 1}": [[n, round(w, 3)] for n, w in slot[:pool_size]]
            for position, slot in enumerate(stats["positional"])
        }

    bonus_ranked = stats["bonus"]
    candidates["bonus"] = [[n, round(w, 3)] for n, w in bonus_ranked[:pool_size]]

    logger.info(
        "Baloto calculator | analysed %d draw(s) %s..%s -> main=%s bonus=%s (%s)",
        len(selected),
        selected[-1].date,
        selected[0].date,
        main,
        bonus_ranked[0][0],
        score_meaning,
    )

    return {
        "draws_analyzed": len(selected),
        "date_range": [selected[-1].date, selected[0].date],
        "window": window,
        "half_life": half_life,
        "strategy": strategy,
        "score_meaning": score_meaning,
        "deterministic_ticket": {"main": main, "bonus": bonus_ranked[0][0]},
        "candidates": candidates,
        "rules": (
            f"The five main numbers must be distinct and between 1 and {MAIN_MAX}; "
            f"the bonus number is between 1 and {BONUS_MAX}. Any substitution must come from the candidate lists. "
            "Neither past frequency nor a long absence makes any future draw more likely."
        ),
    }
