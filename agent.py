import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

from tools.calculator import calculate_monthly_miles, compare_cards, get_card_info

load_dotenv()

# 初始化向量数据库
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# 定义工具
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


# 创建Agent
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools = [search_card_rules, calculate_card_rewards, compare_all_cards, get_card_details]

system_prompt = """You are a Singapore credit card advisor.
Help users find the best credit card based on their spending habits.

Always:
1. Use compare_all_cards tool to rank cards - never calculate miles yourself
2. Use the exact numbers from tool results - never modify or estimate numbers
3. The breakdown numbers in your response must match exactly what the tools return
4. If a card has no bonus rate for a category, say "base rate applied" not "not applicable"

Available cards: Citi Rewards Card, DBS Altitude Card, UOB PRVI Miles Card, OCBC 90°N Card, Standard Chartered Journey Card
"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=system_prompt
)


if __name__ == "__main__":
    print("Singapore Credit Card Advisor")
    print("Type 'quit' to exit\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        response = agent.invoke({
            "messages": [("human", user_input)]
        })
        
        print(f"\nAdvisor: {response['messages'][-1].content}\n")