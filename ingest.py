import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()

# Read raw card data
with open("data/cards_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Convert each earn_rate rule of each card into a separate Document
documents = []

for card in data["cards"]:
    card_name = card["card_name"]
    bank = card["bank"]
    annual_fee = card["annual_fee"]
    annual_fee_waiver = card["annual_fee_waiver"]
    min_income = card["min_income_sgd"]

    # Each earn_rate is stored in a separate chunk
    for rate in card["earn_rates"]:
        content = f"""
Card: {card_name}
Bank: {bank}
Category: {rate['category']}
Earn Rate: {rate['earn_rate']} {rate['unit']}
Monthly Cap: {rate['monthly_cap_sgd'] if rate['monthly_cap_sgd'] else 'No cap'}
Notes: {rate['notes']}
Annual Fee: S${annual_fee} ({annual_fee_waiver})
Min Income: S${min_income}
"""
        doc = Document(
            page_content=content.strip(),
            metadata={
                "card_name": card_name,
                "bank": bank,
                "category": rate["category"],
                "earn_rate": rate["earn_rate"],
                "monthly_cap_sgd": rate["monthly_cap_sgd"] or 0,
                "annual_fee": annual_fee,
                "min_income_sgd": min_income
            }
        )
        documents.append(doc)

    # The overall card information is also stored in a separate chunk
    overall_content = f"""
Card: {card_name}
Bank: {bank}
Annual Fee: S${annual_fee}
Annual Fee Waiver: {annual_fee_waiver}
Sign-up Bonus: {card.get('sign_up_bonus_notes', 'None')}
Min Income Required: S${min_income}
Points Conversion: {card.get('points_to_miles_conversion', 'N/A')}
Miles Expiry: {card.get('miles_expiry', 'Not specified')}
"""
    doc = Document(
        page_content=overall_content.strip(),
        metadata={
            "card_name": card_name,
            "bank": bank,
            "category": "overview",
            "annual_fee": annual_fee,
            "min_income_sgd": min_income
        }
    )
    documents.append(doc)

print(f"Total documents created: {len(documents)}")

# Store in ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Successfully stored in ChromaDB")
print(f"Collection size: {vectorstore._collection.count()}")