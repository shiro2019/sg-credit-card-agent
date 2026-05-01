import json
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from tools.calculator import calculate_monthly_miles, compare_cards, get_card_info

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
    Data last updated: April 2026.
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


# Initialize the agent with tools and system prompt
@st.cache_resource
def init_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    tools = [search_card_rules, calculate_card_rewards, 
             compare_all_cards, get_card_details]
    system_prompt = """You are a Singapore credit card advisor.
Help users find the best credit card based on their spending habits.

Always:
1. Use compare_all_cards tool to rank cards - never calculate miles yourself
2. Use the exact numbers from tool results - never modify or estimate numbers
3. If a card has no bonus rate for a category, say "base rate applied"

Available cards: Citi Rewards Card, DBS Altitude Card, UOB PRVI Miles Card, 
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
            response = agent.invoke({
                "messages": [("human", prompt)]
            })
            answer = response["messages"][-1].content
            answer = answer.replace("S$", "SGD ") # Replace currency symbol avoid displaying garbled characters
            st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})