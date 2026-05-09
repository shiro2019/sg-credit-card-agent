import json
import math

UNLIMITED_SENTINEL = 9999

PRESET_WEIGHTS = {
    "miles":    {"miles": 0.85, "lounge": 0.15},
    "lounge":   {"miles": 0.30, "lounge": 0.70},
    "balanced": {"miles": 0.55, "lounge": 0.45},
}

with open("data/cards_raw.json", "r", encoding="utf-8") as f:
    CARDS_DATA = json.load(f)["cards"]


def _get_base_rate(card: dict) -> float | None:
    """
    Get the fallback multiplier for a card, prioritizing 'others', then 'local', and returning None if no match is found
    """
    for priority_category in ["others", "local"]:
        for rate in card["earn_rates"]:
            if rate["category"] == priority_category:
                return rate["earn_rate"]
    return None


def calculate_monthly_miles(card_name: str, spending: dict) -> dict:
    card = None
    for c in CARDS_DATA:
        if c["card_name"].lower() == card_name.lower():
            card = c
            break

    if not card:
        return {"error": f"Card '{card_name}' not found"}

    # Calculate the fallback multiplier in advance and reuse the same value throughout the function
    base_rate = _get_base_rate(card)

    total_miles = 0
    breakdown = []

    for spend_category, spend_amount in spending.items():
        if spend_amount <= 0:
            continue

        # Precisely match the category
        matched_rate = None
        for rate in card["earn_rates"]:
            if rate["category"] == spend_category:
                matched_rate = rate
                break

        # Use the fallback multiplier if no exact match is found
        if not matched_rate:
            if base_rate is None:
                # Data missing, skip and warn
                print(f"Warning: no base rate found for card '{card_name}', skipping category '{spend_category}'")
                continue
            earn_rate = base_rate
            cap = None
        else:
            earn_rate = matched_rate["earn_rate"]
            cap = matched_rate["monthly_cap_sgd"]

        # Calculate miles within the cap
        if cap:
            effective_spend = min(spend_amount, cap)
        else:
            effective_spend = spend_amount

        miles_earned = effective_spend * earn_rate

        # The overflow portion outside the cap is calculated using the fallback multiplier
        if cap and spend_amount > cap:
            overflow = spend_amount - cap
            if base_rate is None:
                print(f"Warning: no base rate for overflow calculation on card '{card_name}'")
            else:
                miles_earned += overflow * base_rate

        total_miles += miles_earned
        breakdown.append({
            "category": spend_category,
            "spend_sgd": spend_amount,
            "earn_rate_mpd": earn_rate,
            "effective_spend_sgd": effective_spend,
            "miles_earned": round(miles_earned)
        })

    return {
        "card_name": card["card_name"],
        "bank": card["bank"],
        "total_monthly_miles": round(total_miles),
        "annual_miles_estimate": round(total_miles * 12),
        "breakdown": breakdown,
        "annual_fee_sgd": card["annual_fee"],
        "annual_fee_waiver": card["annual_fee_waiver"]
    }


def compare_cards(spending: dict) -> list:
    results = []
    for card in CARDS_DATA:
        result = calculate_monthly_miles(card["card_name"], spending)
        if "error" not in result:
            results.append(result)
    results.sort(key=lambda x: x["total_monthly_miles"], reverse=True)
    return results


def get_card_info(card_name: str) -> dict:
    for card in CARDS_DATA:
        if card["card_name"].lower() == card_name.lower():
            return {
                "card_name": card["card_name"],
                "bank": card["bank"],
                "annual_fee_sgd": card["annual_fee"],
                "annual_fee_waiver": card["annual_fee_waiver"],
                "min_income_sgd": card["min_income_sgd"],
                "sign_up_bonus": card.get("sign_up_bonus_notes", "None"),
                "earn_rates": card["earn_rates"]
            }
    return {"error": f"Card '{card_name}' not found"}

# Multi-dimensional ranking functions

def _parse_lounge_visits(card: dict) -> int:
    """
    Parse lounge_access.free_visits_per_year.
    "unlimited" -> UNLIMITED_SENTINEL (9999)
    null / missing field -> 0
    numeric -> int
    """
    la = card.get("lounge_access")
    if not la:
        return 0
    val = la.get("free_visits_per_year")
    if val is None:
        return 0
    if isinstance(val, str) and val.lower() == "unlimited":
        return UNLIMITED_SENTINEL
    if isinstance(val, (int, float)):
        return int(val)
    return 0


def _score_lounge(card: dict, target_visits: int | None, dynamic_max: int) -> dict:
    """
    Score lounge access for a card. Two modes:
    - target_visits is set: piecewise function (0->0, linear to 85 at target, log bonus beyond, unlimited->100)
    - target_visits is None: pure log scaling (0->0, unlimited->100, finite->log normalized)

    dynamic_max: max finite free_visits_per_year across candidates (excludes unlimited)
    """
    visits = _parse_lounge_visits(card)
    la = card.get("lounge_access") or {}

    if visits == UNLIMITED_SENTINEL:
        score = 100.0
    elif visits == 0:
        score = 0.0
    elif target_visits is not None:
        if visits < target_visits:
            score = (visits / target_visits) * 85.0
        elif visits == target_visits:
            score = 85.0
        else:
            # visits > target: log-scaled small bonus, max 100
            denom = max(dynamic_max - target_visits, 1)
            bonus = math.log(1 + visits - target_visits) / math.log(1 + denom) * 15.0
            score = min(85.0 + bonus, 100.0)
    else:
        # target_visits is None: pure log scaling
        if dynamic_max <= 0:
            score = 0.0
        else:
            score = math.log(visits + 1) / math.log(dynamic_max + 1) * 100.0

    return {
        "score": round(score, 1),
        "free_visits_per_year": "unlimited" if visits == UNLIMITED_SENTINEL else visits,
        "lounge_program": la.get("program"),
        "guest_allowed": la.get("guest_allowed", False),
        "booking_channel": la.get("booking_channel"),
        "guest_fee_usd": la.get("guest_fee_usd")
    }


def _apply_hard_filters(cards: list, hard_filters: dict) -> tuple[list, list]:
    """
    Apply deterministic hard filters to cards.
    Returns (passed_cards, excluded_cards).

    Null sub-fields do NOT exclude when requiring a boolean True (conservative).
    """
    passed = []
    excluded = []

    for card in cards:
        reasons = []

        # requires_lounge: free_visits_per_year > 0
        if hard_filters.get("requires_lounge"):
            if _parse_lounge_visits(card) == 0:
                reasons.append("no complimentary lounge access")

        # requires_travel_insurance: travel_insurance is not null
        if hard_filters.get("requires_travel_insurance"):
            if card.get("travel_insurance") is None:
                reasons.append("no travel insurance coverage")

        # auto_activated_insurance: travel_insurance.auto_activated == True
        if hard_filters.get("auto_activated_insurance"):
            ti = card.get("travel_insurance")
            if ti is None or ti.get("auto_activated") is not True:
                reasons.append("travel insurance not auto-activated")

        # covers_flight_delay: null passes (data gap, not exclusion)
        if hard_filters.get("covers_flight_delay"):
            ti = card.get("travel_insurance")
            if ti is not None and ti.get("covers_flight_delay") is False:
                reasons.append("no flight delay coverage")
            # null -> passes (conservative)

        # max_annual_fee_sgd
        max_fee = hard_filters.get("max_annual_fee_sgd")
        if max_fee is not None:
            if card.get("annual_fee", float("inf")) > max_fee:
                reasons.append(f"annual fee S${card['annual_fee']} exceeds budget S${max_fee}")

        # min_income_sgd: card's min_income must be <= user's income
        user_income = hard_filters.get("min_income_sgd")
        if user_income is not None:
            if card.get("min_income_sgd", float("inf")) > user_income:
                reasons.append(f"min income requirement S${card['min_income_sgd']} exceeds your income S${user_income}")

        # card_tier: exact match
        req_tier = hard_filters.get("card_tier")
        if req_tier is not None:
            if card.get("card_tier") != req_tier:
                reasons.append(f"card tier '{card.get('card_tier')}' != required '{req_tier}'")

        # lounge_program: exact match
        req_program = hard_filters.get("lounge_program")
        if req_program is not None:
            la = card.get("lounge_access") or {}
            if la.get("program") != req_program:
                reasons.append(f"lounge program '{la.get('program')}' != required '{req_program}'")

        # fx_fee_free: fx_fee_rate == 0.0
        if hard_filters.get("fx_fee_free"):
            if card.get("fx_fee_rate", 0.0) != 0.0:
                reasons.append(f"FX fee {card.get('fx_fee_rate')}% applies")

        if reasons:
            excluded.append({
                "card_name": card["card_name"],
                "bank": card["bank"],
                "excluded_reasons": reasons
            })
        else:
            passed.append(card)

    return passed, excluded


def rank_cards_by_needs(
    spending: dict = None,
    priorities: list = None,
    hard_filters: dict = None,
    target_visits: int | None = None
) -> dict:
    """
    Two-stage ranking: hard filter -> weighted scoring.

    Stage 1: Apply hard_filters to exclude cards that don't meet must-have criteria.
    Stage 2: Score remaining cards on miles + lounge, weighted by priorities.

    Sorting:
      - satisfied (lounge_score >= 85): by total_annual_cost ascending
      - partial (0 < lounge_score < 85): by total_score descending
      - unsatisfied (lounge_score == 0): by total_annual_cost ascending

    Returns dict with 'ranked', 'excluded', 'weights_used', 'filter_summary'.
    """
    spending = spending or {}
    hard_filters = hard_filters or {}
    priorities = priorities or []

    # ── Resolve weights from priorities ──
    if not priorities:
        weights = dict(PRESET_WEIGHTS["miles"])
    elif priorities == ["balanced"]:
        weights = dict(PRESET_WEIGHTS["balanced"])
    elif len(priorities) == 1:
        primary = priorities[0]
        if primary in PRESET_WEIGHTS:
            weights = dict(PRESET_WEIGHTS[primary])
        else:
            weights = dict(PRESET_WEIGHTS["miles"])
    else:
        # Ordered list: first -> 0.65, second -> 0.35
        weights = {}
        ordered_keys = [p for p in priorities if p in ("miles", "lounge")]
        if len(ordered_keys) >= 2:
            weights[ordered_keys[0]] = 0.65
            weights[ordered_keys[1]] = 0.35
        elif len(ordered_keys) == 1:
            weights = dict(PRESET_WEIGHTS[ordered_keys[0]])
        else:
            weights = dict(PRESET_WEIGHTS["miles"])

    # ── Stage 1: Hard filter ──
    all_cards = list(CARDS_DATA)
    passed, excluded = _apply_hard_filters(all_cards, hard_filters)

    if not passed:
        return {
            "ranked": [],
            "excluded": excluded,
            "weights_used": weights,
            "filter_summary": "No cards passed all filters."
        }

    # ── Calculate raw miles for each passed card ──
    miles_results = {}
    for card in passed:
        result = calculate_monthly_miles(card["card_name"], spending)
        if "error" not in result:
            miles_results[card["card_name"]] = result

    # ── Normalize miles to 0-100 ──
    max_miles = max(
        (r["total_monthly_miles"] for r in miles_results.values()),
        default=1
    )
    miles_scores = {}
    for name, r in miles_results.items():
        if max_miles > 0:
            miles_scores[name] = round(r["total_monthly_miles"] / max_miles * 100, 1)
        else:
            miles_scores[name] = 0.0

    # ── Calculate lounge dynamic_max (max finite visits, exclude unlimited) ──
    finite_visits = [
        _parse_lounge_visits(c)
        for c in passed
        if _parse_lounge_visits(c) not in (0, UNLIMITED_SENTINEL)
    ]
    dynamic_max = max(finite_visits) if finite_visits else 0

    # ── Score lounge for each passed card ──
    lounge_results = {}
    for card in passed:
        lounge_results[card["card_name"]] = _score_lounge(card, target_visits, dynamic_max)

    # ── Build ranked list ──
    ranked = []
    for card in passed:
        name = card["card_name"]
        ms = miles_scores.get(name, 0.0)
        ls = lounge_results[name]["score"]
        w_m = weights.get("miles", 0.5)
        w_l = weights.get("lounge", 0.5)
        total_score = round(ms * w_m + ls * w_l, 1)

        # Determine bucket
        if ls >= 85:
            bucket = "satisfied"
        elif ls > 0:
            bucket = "partial"
        else:
            bucket = "unsatisfied"

        # total_annual_cost
        fx_rate = card.get("fx_fee_rate", 0.0)
        overseas_spend = spending.get("overseas", 0)
        total_annual_cost = round(
            card["annual_fee"] + overseas_spend * 12 * (fx_rate / 100.0), 2
        )

        # Collect data_gaps from lounge and travel_insurance
        data_gaps = []
        la = card.get("lounge_access") or {}
        if la.get("booking_channel") is None:
            data_gaps.append("lounge_booking_channel")
        ti = card.get("travel_insurance")
        if ti is not None:
            for sub_field in ["covers_flight_delay", "covers_baggage_loss",
                              "min_delay_hours", "flight_delay_payout_sgd", "claims_contact"]:
                if ti.get(sub_field) is None:
                    data_gaps.append(f"travel_insurance.{sub_field}")

        mr = miles_results.get(name, {})
        lr = lounge_results[name]

        ranked.append({
            "card_name": name,
            "bank": card["bank"],
            "total_score": total_score,
            "bucket": bucket,
            "scores": {
                "miles": ms,
                "lounge": ls
            },
            "weights_used": weights,
            "raw": {
                "monthly_miles": mr.get("total_monthly_miles", 0),
                "annual_miles": mr.get("annual_miles_estimate", 0),
                "lounge_free_visits": lr["free_visits_per_year"],
                "lounge_program": lr["lounge_program"],
                "lounge_guest_allowed": lr["guest_allowed"],
                "lounge_booking_channel": lr["booking_channel"],
                "annual_fee_sgd": card["annual_fee"],
                "annual_fee_waiver": card["annual_fee_waiver"],
                "fx_fee_rate": fx_rate,
                "total_annual_cost_sgd": total_annual_cost,
                "travel_insurance_coverage_sgd": ti["coverage_sgd"] if ti else None,
                "travel_insurance_auto_activated": ti["auto_activated"] if ti else None,
                "data_gaps": data_gaps
            }
        })

    # ── Sort: by bucket priority, then within-bucket rule ──
    bucket_order = {"satisfied": 0, "partial": 1, "unsatisfied": 2}

    def sort_key(entry):
        b = bucket_order[entry["bucket"]]
        if b == 0:  # satisfied: cost ascending
            return (b, entry["raw"]["total_annual_cost_sgd"], -entry["total_score"])
        elif b == 1:  # partial: score descending
            return (b, -entry["total_score"], entry["raw"]["total_annual_cost_sgd"])
        else:  # unsatisfied: cost ascending
            return (b, entry["raw"]["total_annual_cost_sgd"], -entry["total_score"])

    ranked.sort(key=sort_key)
    for i, entry in enumerate(ranked):
        entry["rank"] = i + 1

    # ── Build filter_summary ──
    active_filters = []
    if hard_filters.get("requires_lounge"):
        active_filters.append("complimentary lounge access")
    if hard_filters.get("requires_travel_insurance"):
        active_filters.append("travel insurance")
    if hard_filters.get("auto_activated_insurance"):
        active_filters.append("auto-activated insurance")
    if hard_filters.get("covers_flight_delay"):
        active_filters.append("flight delay coverage")
    if hard_filters.get("max_annual_fee_sgd") is not None:
        active_filters.append(f"annual fee <= S${hard_filters['max_annual_fee_sgd']}")
    if hard_filters.get("min_income_sgd") is not None:
        active_filters.append(f"income eligibility")
    if hard_filters.get("card_tier") is not None:
        active_filters.append(f"card tier: {hard_filters['card_tier']}")
    if hard_filters.get("lounge_program") is not None:
        active_filters.append(f"lounge program: {hard_filters['lounge_program']}")
    if hard_filters.get("fx_fee_free"):
        active_filters.append("no FX fee")

    if active_filters:
        filter_summary = f"Filtered to cards with: {', '.join(active_filters)}."
    else:
        filter_summary = "No hard filters applied."

    return {
        "ranked": ranked,
        "excluded": excluded,
        "weights_used": weights,
        "filter_summary": filter_summary
    }


if __name__ == "__main__":
    test_spending = {
        "online_retail": 500,
        "overseas": 300,
        "local": 400
    }

    print("=== Single card calculation ===")
    result = calculate_monthly_miles("Citi Rewards Card", test_spending)
    print(json.dumps(result, indent=2))

    print("\n=== All cards comparison ===")
    comparison = compare_cards(test_spending)
    for i, card in enumerate(comparison):
        print(f"{i+1}. {card['card_name']}: {card['total_monthly_miles']} miles/month")

    print("\n=== Multi-dimensional ranking (lounge priority, no hard filter) ===")
    ranking = rank_cards_by_needs(
        spending=test_spending,
        priorities=["lounge", "miles"],
        hard_filters={},
        target_visits=4
    )
    for entry in ranking["ranked"]:
        print(f"{entry['rank']}. {entry['card_name']} ({entry['bucket']}): "
              f"score={entry['total_score']}, cost=S${entry['raw']['total_annual_cost_sgd']}")

    print("\n=== Multi-dimensional ranking (requires lounge + max fee S$250) ===")
    ranking2 = rank_cards_by_needs(
        spending=test_spending,
        priorities=["lounge", "miles"],
        hard_filters={"requires_lounge": True, "max_annual_fee_sgd": 250},
        target_visits=2
    )
    print(ranking2["filter_summary"])
    for entry in ranking2["ranked"]:
        print(f"{entry['rank']}. {entry['card_name']} ({entry['bucket']}): "
              f"score={entry['total_score']}, lounge={entry['scores']['lounge']}, "
              f"cost=S${entry['raw']['total_annual_cost_sgd']}")
    if ranking2["excluded"]:
        print("Excluded:")
        for ex in ranking2["excluded"]:
            print(f"  - {ex['card_name']}: {ex['excluded_reasons']}")