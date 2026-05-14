import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from tools.calculator import calculate_monthly_miles, compare_cards, get_card_info, rank_cards_by_needs

load_dotenv()

# Build vector store if not exists
import os
if not os.path.exists("chroma_db"):
    import json
    from langchain_core.documents import Document
    
    with open("data/cards_raw.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    documents = []
    for card in data["cards"]:
        card_name = card["card_name"]
        bank = card["bank"]
        
        for rate in card["earn_rates"]:
            content = f"""Card: {card_name}
Bank: {bank}
Category: {rate['category']}
Earn Rate: {rate['earn_rate']} {rate['unit']}
Monthly Cap: {rate['monthly_cap_sgd'] if rate['monthly_cap_sgd'] else 'No cap'}
Notes: {rate['notes']}
Annual Fee: S${card['annual_fee']} ({card['annual_fee_waiver']})
Min Income: S${card['min_income_sgd']}"""
            documents.append(Document(
                page_content=content,
                metadata={
                    "card_name": card_name,
                    "bank": bank,
                    "category": rate["category"],
                    "earn_rate": rate["earn_rate"],
                    "monthly_cap_sgd": rate["monthly_cap_sgd"] or 0,
                    "annual_fee": card["annual_fee"],
                    "min_income_sgd": card["min_income_sgd"]
                }
            ))
        
        overview = f"""Card: {card_name}
Bank: {bank}
Annual Fee: S${card['annual_fee']}
Annual Fee Waiver: {card['annual_fee_waiver']}
Sign-up Bonus: {card.get('sign_up_bonus_notes', 'None')}
Min Income: S${card['min_income_sgd']}
Points Conversion: {card.get('points_to_miles_conversion', 'N/A')}
Miles Expiry: {card.get('miles_expiry', 'Not specified')}"""
        documents.append(Document(
            page_content=overview,
            metadata={
                "card_name": card_name,
                "bank": bank,
                "category": "overview",
                "annual_fee": card["annual_fee"],
                "min_income_sgd": card["min_income_sgd"]
            }
        ))
    
    temp_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    Chroma.from_documents(
        documents=documents,
        embedding=temp_embeddings,
        persist_directory="chroma_db"
    )

# Page config
st.set_page_config(
    page_title="SG Credit Card Advisor",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Singapore Credit Card Advisor")
st.caption("Find the best credit card based on your spending habits")

# Disclaimer
with st.expander("⚠️ Disclaimer"):
    st.write("""
    This tool provides general information only and does not constitute financial advice.
    Card benefits and terms may change. Always verify with the respective bank before applying.
    Data last updated: May 2026.
    """)

# Initialize vectorstore 
@st.cache_resource
def init_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

vectorstore = init_vectorstore()

# Tools definitions for the agent to use
@tool
def search_card_rules(query: str) -> str:
    """
    Search the knowledge base for credit card rules, earn rates, 
    benefits and conditions.
    """
    results = vectorstore.similarity_search(query, k=3)
    return "\n---\n".join([doc.page_content for doc in results])


@tool
def calculate_card_rewards(card_name: str, spending_json: str) -> str:
    """
    Calculate the monthly miles earned for a specific card given user spending habits.
    
    Args:
        card_name: Full name of the card
        spending_json: JSON string representing monthly spending in different categories.
    """
    try:
        spending = json.loads(spending_json)
    except json.JSONDecodeError:
        return "Error: invalid JSON"
    result = calculate_monthly_miles(card_name, spending)
    return json.dumps(result, indent=2)


@tool
def compare_all_cards(spending_json: str) -> str:
    """
    Compare all cards and rank by monthly miles earned.
    Use this to find the best card for a user's spending profile.
    
    Args:
        spending_json: JSON string of spending by category e.g.
                      '{"online_retail": 500, "overseas": 300, "local": 400}'
    
    Available spending categories:
    - online_retail: online shopping
    - overseas: foreign currency spend  
    - local: local SGD spend
    - bonus_transactions: transport, food delivery, groceries (SC Journey only)
    - airlines_hotels: airline and hotel bookings (UOB PRVI only)
    - agoda: bookings on Agoda (OCBC 90N only)
    """
    try:
        spending = json.loads(spending_json)
    except json.JSONDecodeError:
        return "Error: invalid JSON"
    results = compare_cards(spending)
    output = "Cards ranked by monthly miles:\n"
    for i, card in enumerate(results):
        output += f"\n{i+1}. {card['card_name']} ({card['bank']})\n"
        output += f"   Monthly miles: {card['total_monthly_miles']}\n"
        output += f"   Annual miles estimate: {card['annual_miles_estimate']}\n"
        output += f"   Annual fee: SGD {card['annual_fee_sgd']} ({card['annual_fee_waiver']})\n"
    return output


@tool
def get_card_details(card_name: str) -> str:
    """Get detailed information about a specific card."""
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

    Do NOT use for pure miles questions — use compare_all_cards instead.

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

    hard_filters: binary — cards that fail are excluded entirely.
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


# Initialize the agent with tools and system prompt
@st.cache_resource
def init_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_card_rules, calculate_card_rewards,
             compare_all_cards, get_card_details, match_card_by_needs]
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
  "must have" / "need" / "necessary" / "是必须" / "得有" / "一定要"
  → hard_filters: { "requires_lounge": true }

SOFT PREFERENCE only (priorities, no hard_filters):
  "prefer" / "nice to have" / "最好有" / "if possible" / "would be great"
  → priorities: ["lounge", "miles"], no requires_lounge filter

AMBIGUOUS ("I want lounge" / "I need a card with lounge"):
  → Use hard filter (err on safe side)

PRIORITIES mapping:
  "lounge matters more than miles" → ["lounge", "miles"]
  "best overall / balanced"        → ["balanced"]
  "mainly for miles"               → ["miles"] or use compare_all_cards
  No preference stated             → priorities: null (default miles)

LOUNGE VISIT COUNT:
  When user expresses lounge need but does not state visit count, ask:
  "How many lounge visits do you need per year?
   (If unsure, I'll rank by more visits = better)"

═══════════════════════════════════════
NARRATION RULES
═══════════════════════════════════════

Structure your response:
1. State filter_summary first
2. If excluded cards exist, briefly explain why
3. Present ranked results, citing exact raw numbers from tool output
4. For data_gaps fields, explicitly tell user to verify

Citing numbers:
  → Always use exact figures from tool output — never compute or estimate
  → total_annual_cost_sgd: cite as "estimated annual cost including FX fees"

Raw fields to mention by query type:
  Lounge query    → lounge_free_visits, lounge_program, lounge_guest_allowed,
                    lounge_booking_channel, annual_fee_sgd
  Miles query     → monthly_miles, annual_miles, annual_fee_sgd
  General/overall → monthly_miles, lounge_free_visits, annual_fee_sgd only

Data gaps:
  → "Booking channel data is unavailable for [card] — please check the
    Priority Pass app or call [bank] to confirm."

Unsupported filter conditions:
  → If user's requirement cannot map to any available filter parameter,
    do NOT silently approximate
  → Tell user what you can and cannot filter on

Language:
  → Respond in the same language the user writes in

═══════════════════════════════════════
AVAILABLE CARDS (current database)
═══════════════════════════════════════
Citi Rewards Card, DBS Altitude Card, UOB PRVI Miles Card,
OCBC 90°N Card, Standard Chartered Journey Card
"""

    return create_react_agent(model=llm, tools=tools, prompt=system_prompt)


agent = init_agent()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hi! I can help you find the most suitable Singapore credit card. Tell me about your monthly spending habits. For example, how much do you spend on online shopping, overseas travel, and daily local expenses?"
    })

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.text(msg["content"])
        else:
            st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask me about credit cards..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.text(prompt)

    # Agent response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            # Build full conversation history
            history = []
            for msg in st.session_state.messages:
                role = "human" if msg["role"] == "user" else "assistant"
                history.append((role, msg["content"]))
            history.append(("human", prompt))

            response = agent.invoke({"messages": history})
            answer = response["messages"][-1].content
            answer = answer.replace("S$", "SGD ") # Replace currency symbol avoid displaying garbled characters
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})