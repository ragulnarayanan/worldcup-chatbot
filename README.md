# 🏆 FIFA World Cup Analyst Chatbot

A conversational AI chatbot that answers natural language questions about FIFA World Cup history, team statistics, and match predictions using a **Retrieval-Augmented Generation (RAG)** pipeline.

Instead of relying on the LLM's training data alone, the chatbot retrieves real match records from a Pinecone vector database — enriched with goalscorer names, penalty shootout results, and team statistics computed across 49,016 international matches dating back to 1872. A LangGraph ReAct agent dynamically decides which of 5 specialized tools to call based on the user's question, chains tools together when needed (e.g. head-to-head lookup → prediction generation), and maintains conversation memory so follow-up questions work naturally.

Users can ask about specific match results, compare teams, get all-time statistics, predict upcoming fixtures with visual H2H charts, and explore the 2026 World Cup — all in natural language, in any language they prefer.

Built with LangChain · LangGraph · Pinecone · OpenAI · Streamlit

🌐 **Live App:** [worldcup-chatbot.streamlit.app](https://worldcup-chatbot-h65uaprgzj9fg6f7wchze2.streamlit.app/)

---

## 🏗️ Architecture

```
User Question
      ↓
LangGraph ReAct Agent (GPT-4o-mini)
      ↓ decides which tool to call
      ├── match_retrieval_tool    →  worldcup-matches    (Pinecone)
      ├── team_stats_tool         →  worldcup-team-stats (Pinecone)
      ├── head_to_head_tool       →  international-matches (Pinecone)
      ├── match_prediction_tool   →  GPT-4o-mini + Plotly chart
      └── wc2026_tool             →  worldcup-2026       (Pinecone)
                ↓
      LLM reads retrieved context → generates grounded answer
                ↓
      User sees response + optional H2H chart
```

---

## 📦 Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | OpenAI GPT-4o-mini |
| Embeddings | OpenAI text-embedding-3-small (1536 dims) |
| Vector Database | Pinecone (4 serverless indexes, AWS us-east-1) |
| Agent Framework | LangGraph ReAct agent |
| Memory | LangGraph MemorySaver + thread_id |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Notebook UI | ipywidgets |
| Web UI | Streamlit |
| Scraping | BeautifulSoup4 + requests (rate-limited + cached) |
| Environment | Google Colab |

---

## 📊 Data Sources

| File | Source | Records | License |
|------|--------|---------|---------|
| results.csv | [Kaggle — International Football Results 1872–2025](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 49,016 matches | Open |
| goalscorers.csv | [Kaggle — International Football Results 1872–2025](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Every goal + scorer name | Open |
| shootouts.csv | [Kaggle — International Football Results 1872–2025](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | 665 penalty shootout records | Open |
| former_names.csv | [Kaggle — International Football Results 1872–2025](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017) | Historical team name changes | Open |
| 2026 FIFA World Cup | [Wikipedia](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup) | Qualified teams + host info | Open (scraped responsibly) |

> **Scraping policy:** Wikipedia page is fetched once with a 2-second rate limit and cached locally as HTML. No repeated requests are made.

---

## 🗄️ Pinecone Indexes

| Index | Documents | Content |
|-------|-----------|---------|
| worldcup-matches | ~850 | WC match results enriched with goalscorers + shootouts |
| international-matches | ~49,016 | All international matches 1872–2025 |
| worldcup-team-stats | ~80 | Per-team stats, top scorers, penalty record |
| worldcup-2026 | ~50 | Qualified teams, host info, tournament facts |

Each document uses **hybrid conversion**:
- `page_content` → natural language sentence → embedded into 1536-dim vector
- `metadata` → structured dictionary → used for exact filtering

---

## 🛠️ The 5 Pipeline Tools

```
Tool 1: match_retrieval_tool
  Stage: Retrieval
  Index: worldcup-matches
  Use:   Historical match results, scorers, shootouts

Tool 2: team_stats_tool
  Stage: Retrieval + Aggregation
  Index: worldcup-team-stats
  Use:   Win rates, appearances, top scorers, records

Tool 3: head_to_head_tool
  Stage: Reasoning + Aggregation
  Index: international-matches
  Use:   H2H record + recent form (always before prediction)

Tool 4: match_prediction_tool
  Stage: Report Generation
  Uses:  head_to_head data + GPT-4o-mini + Plotly
  Use:   3-paragraph match preview + predicted scoreline + chart

Tool 5: wc2026_tool
  Stage: Retrieval
  Index: worldcup-2026
  Use:   2026 qualified teams, host cities, tournament info
```

---

## 💬 Features

- **Conversational memory** — follow-up questions work without repeating context
- **4 persistent user preferences** — language, favorite team, detail level, format
- **Scorer enrichment** — every match document includes goalscorer names and annotations
- **Penalty shootout data** — correctly handles draws that went to penalties
- **Name normalization** — West Germany → Germany, Soviet Union → Russia across all eras
- **Graceful fallbacks** — never hallucinates when data is missing
- **Plotly H2H charts** — visual bar chart for every prediction

---

## 📁 Repository Structure

```
worldcup-chatbot/
├── worldcup_chatbot.ipynb   ← main Colab notebook
├── app.py                   ← Streamlit web UI
├── requirements.txt         ← package dependencies
└── README.md
```

---

## 🌐 Streamlit Deployment

**Live App:** [https://worldcup-chatbot-h65uaprgzj9fg6f7wchze2.streamlit.app/](https://worldcup-chatbot-h65uaprgzj9fg6f7wchze2.streamlit.app/)

```bash
# Run locally
streamlit run app.py
```

---

## 👤 Author

**Ragul Narayanan Magesh**
MS Data Analytics Engineering — Northeastern University
Boston, MA

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/ragul-narayanan-magesh-18013916b/)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/ragulnarayanan)


---

## 📄 License

This project is for educational purposes.
Data sources are publicly available and credited above.
