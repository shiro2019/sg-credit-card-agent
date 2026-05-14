import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from tools.calculator import calculate_monthly_miles, compare_cards, get_card_info, rank_cards_by_needs

load_dotenv()

# Initialize vectorstore 
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Tools definitions
@tool
def search_card_rules(query: str) -> str:
    """
    Search the knowledge base for credit card rules, earn rates, benefits and conditions.
    Use this when you need to find information about specific card features or compare card benefits.
    """
    results = vectorstore.similarity_search(query, k=3)
    output = []
    for doc in results:
        output.append(doc.page_content)
    return "\n---\n".join(output)


@tool
def calculate_card_rewards(card_name: str, spending_json: str) -> str:
    """
    Calculate the monthly miles earned for a specific card given user spending habits.
    
    Args:
        card_name: Full name of the card e.g. "Citi Rewards Card"
        spending_json: JSON string of spending by category e.g. 
                      '{"online_retail": 500, "overseas": 300, "local": 400}'
    
    Available spending categories:
    - online_retail: online shopping
    - overseas: foreign currency spend
    - local: local SGD spend
    - bonus_transactions: transport, food delivery, groceries (SC Journey only)
    - airlines_hotels: airline and hotel bookings (UOB PRVI only)
    - agoda: bookings on Agoda (OCBC 90N only)
    
    Returns miles calculation breakdown.
    """
    try:
        spending = json.loads(spending_json)
    except json.JSONDecodeError:
        return "Error: spending_json must be valid JSON"
    
    result = calculate_monthly_miles(card_name, spending)
    return json.dumps(result, indent=2)


@tool
def compare_all_cards(spending_json: str) -> str:
    """
    Compare all available cards and rank them by monthly miles earned.
    Use this to find the best card for a user's spending profile.
    
    Args:
        spending_json: JSON string of spending by category
    
    Returns ranked list of cards by miles earned.
    """
    try:
        spending = json.loads(spending_json)
    except json.JSONDecodeError:
        return "Error: spending_json must be valid JSON"
    
    results = compare_cards(spending)
    
    output = "Cards ranked by monthly miles:\n"
    for i, card in enumerate(results):
        output += f"\n{i+1}. {card['card_name']} ({card['bank']})\n"
        output += f"   Monthly miles: {card['total_monthly_miles']}\n"
        output += f"   Annual miles estimate: {card['annual_miles_estimate']}\n"
        output += f"   Annual fee: S${card['annual_fee_sgd']} ({card['annual_fee_waiver']})\n"
    
    return output


@tool
def get_card_details(card_name: str) -> str:
    """
    Get detailed information about a specific card including fees and eligibility.
    
    Args:
        card_name: Full name of the card
    """
    result = get_card_info(card_name)
    return json.dumps(result, indent=2)


@tool
def match_card_by_needs(needs_json: str) -> str:
    """
    Rank cards using multi-dimensional scoring for lifestyle/benefit queries.

    Use this tool when user mentions ANY of:
    - Lounge access / Priority Pass / airport lounge
    - Travel insurance
    - Annual fee budget or value for money
    - FX fee / foreign transaction fee
    - Card tier (entry/mid/premium)
    - "Best overall" or balanced recommendation

    Do NOT use for pure miles questions, use compare_all_cards instead.

    Input JSON:
    {
        "spending": {"local": 2000, "overseas": 500},
        "priorities": ["lounge", "miles"],
        "hard_filters": {
            "requires_lounge": true,
            "max_annual_fee_sgd": 300,
            "requires_travel_insurance": false,
            "fx_fee_free": false,
            "card_tier": null,
            "lounge_program": null
        },
        "target_visits": 4
    }

    hard_filters: binary, cards that fail are excluded entirely.
    priorities: ordered list, first = most important.
      Valid: "miles", "lounge", "balanced" (preset)
    target_visits: user's stated annual lounge visit need. null if not stated.

    Returns ranked list with scores, buckets, raw data, and excluded cards.
    """
    try:
        params = json.loads(needs_json)
    except json.JSONDecodeError:
        return "Error: needs_json must be valid JSON"

    result = rank_cards_by_needs(
        spending=params.get("spending", {}),
        priorities=params.get("priorities", None),
        hard_filters=params.get("hard_filters", {}),
        target_visits=params.get("target_visits", None)
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


# Initialize the agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [search_card_rules, calculate_card_rewards, compare_all_cards, get_card_details, match_card_by_needs]

system_prompt = """You are a Singapore credit card advisor.
Help users find the best credit card based on their spending habits and lifestyle needs.

═══════════════════════════════════════
TOOL SELECTION RULES
═══════════════════════════════════════

Use compare_all_cards:
  → Pure miles/rewards questions only
  → "Which card gives the most miles for my spending?"

Use match_card_by_needs:
  → User mentions lounge, travel insurance, annual fee budget,
    FX fee, card tier, or "best overall"
  → Any lifestyle or benefit question beyond pure miles

Use search_card_rules:
  → Specific policy questions ("does this card cover X?")
  → Detailed T&C queries

Use calculate_card_rewards:
  → Single card calculation requested explicitly

Use get_card_details:
  → Specific card info requested, not comparison

═══════════════════════════════════════
INTENT → FILTER MAPPING (for match_card_by_needs)
═══════════════════════════════════════

HARD FILTER triggers (use hard_filters, not just priorities):
  "must have" / "need" / "necessary" / "是必须" / "得有" / "一定要" / "necessary" / "essential"
  → hard_filters: { "requires_lounge": true }

SOFT PREFERENCE only (priorities, no hard_filters):
  "prefer" / "nice to have" / "最好有" / "if possible" / "would be great"
  → priorities: ["lounge", "miles"], no requires_lounge filter

AMBIGUOUS ("I want lounge" / "I need a card with lounge"):
  → Use hard filter (err on safe side)
  → Better to exclude no-lounge cards than recommend them to someone who needs lounge

PRIORITIES mapping:
  "lounge matters more than miles" → ["lounge", "miles"]
  "best overall / balanced"        → ["balanced"]
  "mainly for miles"               → ["miles"] or use compare_all_cards
  No preference stated             → priorities: null (default miles)

LOUNGE VISIT COUNT:
  When user expresses lounge need but does not state visit count, ask:
  "How many lounge visits do you need per year?
   (If unsure, I'll rank by more visits = better)"
  Pass the answer as target_visits in the tool call.

═══════════════════════════════════════
NARRATION RULES
═══════════════════════════════════════

Structure your response:
1. State filter_summary first ("I've filtered to cards with complimentary lounge access.")
2. If excluded cards exist, briefly explain why they were excluded
3. Present ranked results, citing exact raw numbers from tool output
4. For data_gaps fields, explicitly tell user to verify

Citing numbers:
  → Always use exact figures from tool output — never compute or estimate
  → monthly_miles, lounge_free_visits, annual_fee_sgd: cite directly
  → total_annual_cost_sgd: cite as "estimated annual cost including FX fees"

Raw fields to mention by query type:
  Lounge query    → lounge_free_visits, lounge_program, lounge_guest_allowed,
                    lounge_booking_channel, annual_fee_sgd
  Miles query     → monthly_miles, annual_miles, annual_fee_sgd
  General/overall → monthly_miles, lounge_free_visits, annual_fee_sgd only;
                    surface others only if directly relevant
  Do NOT mention unrelated fields (e.g. don't mention insurance when user asked about lounge)

Annual fee value:
  → Describe qualitatively ONLY — never score or rank by fee value
  → Example: "UOB PRVI charges SGD 261 but includes 4 lounge visits and
    SGD 500k travel insurance — whether this is worth it depends on your travel frequency"

Data gaps:
  → "Booking channel data is unavailable for [card] — please check the
    Priority Pass app or call [bank] to confirm."
  → guest_allowed defaults to false when data is missing (conservative assumption)

Unsupported filter conditions:
  → If user's requirement cannot map to any available filter parameter,
    do NOT silently approximate to the nearest field
  → Tell user: "I don't have structured data for [requirement] and cannot
    filter on it automatically. Here's what I do have: [relevant raw fields]"

Language:
  → Respond in the same language the user writes in
  → Numbers and card names stay in their original form regardless of language

═══════════════════════════════════════
AVAILABLE CARDS (current database)
═══════════════════════════════════════
Citi Rewards Card, DBS Altitude Card, UOB PRVI Miles Card,
OCBC 90°N Card, Standard Chartered Journey Card
"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt
)


if __name__ == "__main__":
    print("Singapore Credit Card Advisor")
    print("Type 'quit' to exit\n")

    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        history.append(("human", user_input))
        response = agent.invoke({"messages": history})
        answer = response["messages"][-1].content
        history.append(("assistant", answer))

        print(f"\nAdvisor: {answer}\n")