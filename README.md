# 💳 Singapore Credit Card Advisor

An agentic RAG system that recommends the optimal Singapore credit card based on user spending habits. Built with LangChain, LangGraph, and ChromaDB.

## Demo

> Demo available upon request (contact for access credentials)

## Features

- **Agentic RAG**: LangGraph ReAct agent autonomously decides which tools to call based on user queries
- **Deterministic calculation**: Miles/cashback computed via Python functions but not by LLM to eliminate hallucination
- **Hybrid knowledge base**: ChromaDB vector store with semantic search over card rules and T&C documents
- **Multi-card comparison**: Ranks all available cards by projected monthly miles given user spending profile

## Architecture

User Input
    ↓
ReAct Agent (gpt-4o-mini)
    ↓
[search_card_rules] [compare_all_cards] [calculate_card_rewards] [get_card_details]
    ↓
Deterministic Calculator (Python)
    ↓
Natural Language Response

## Tech Stack

- **Agent framework**: LangGraph (ReAct)
- **LLM**: GPT-4o-mini
- **Vector store**: ChromaDB
- **Embeddings**: OpenAI text-embedding-3-small
- **Frontend**: Streamlit
- **Data**: major Singapore bank cards with structured earn rate schemas

## Quick Start

```bash
pip install -r requirements.txt
python ingest.py    
streamlit run app.py
```

> Requires OpenAI API key in `.env`