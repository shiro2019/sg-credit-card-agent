import json

with open("data/cards_raw.json", "r", encoding="utf-8") as f:
    CARDS_DATA = json.load(f)["cards"]


def _get_base_rate(card: dict) -> float | None:
    """
    获取一张卡的兜底倍率，优先选 'others'，其次 'local'，找不到返回 None
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

    # 提前计算好兜底倍率，整个函数复用同一个值
    base_rate = _get_base_rate(card)

    total_miles = 0
    breakdown = []

    for spend_category, spend_amount in spending.items():
        if spend_amount <= 0:
            continue

        # 精确匹配该消费类别
        matched_rate = None
        for rate in card["earn_rates"]:
            if rate["category"] == spend_category:
                matched_rate = rate
                break

        # 没有精确匹配则用兜底倍率
        if not matched_rate:
            if base_rate is None:
                # 数据缺失，跳过并警告，不要用魔法数字兜底
                print(f"Warning: no base rate found for card '{card_name}', skipping category '{spend_category}'")
                continue
            earn_rate = base_rate
            cap = None
        else:
            earn_rate = matched_rate["earn_rate"]
            cap = matched_rate["monthly_cap_sgd"]

        # 计算 cap 内的 miles
        if cap:
            effective_spend = min(spend_amount, cap)
        else:
            effective_spend = spend_amount

        miles_earned = effective_spend * earn_rate

        # cap 外的溢出部分，用兜底倍率计算
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


if __name__ == "__main__":
    test_spending = {
        "online_retail": 500,
        "overseas": 300,
        "local": 400
    }

    print("=== 单张卡计算 ===")
    result = calculate_monthly_miles("Citi Rewards Card", test_spending)
    print(json.dumps(result, indent=2))

    print("\n=== 所有卡对比 ===")
    comparison = compare_cards(test_spending)
    for i, card in enumerate(comparison):
        print(f"{i+1}. {card['card_name']}: {card['total_monthly_miles']} miles/month")