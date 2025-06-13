
============================================================
                     xenRon AI v0.5.1 README
                    — The Self-Learning AI Bot —
============================================================

Project: xenRon AI  
Version: 0.5.1  
Author: xenex-ai  
GitHub: https://github.com/xenex-ai/xenRon_AI  
File: xenron_v051.py  
Language: Python 3.10+  
License: MIT

============================================================
📌 OVERVIEW
============================================================

**xenRon AI** is a self-learning, multi-functional AI Telegram bot designed to simulate intelligent behavior, store and reflect on knowledge, generate visual and audio content, and interactively evolve with users.

It features:
✔ Semantic learning and search  
✔ Automated Wikipedia & DuckDuckGo scraping  
✔ Knowledge graph construction and logical inference  
✔ Audio (Text-to-Speech) and AI image generation  
✔ Feedback processing and reflection  
✔ Telegram interface with custom command handling  
✔ Admin features, stats, and visual dashboards  

xenRon is inspired by GPT-like AI behavior but runs fully locally, utilizing APIs only when needed.

============================================================
🔧 FEATURES & COMMANDS
============================================================

/start  
» Registers the user and shows a greeting with knowledge count.

/help  
» Shows a structured help overview of all available commands.

/lernen <keyword> = <info>  
» Teaches the bot a new knowledge entry. The `keyword` becomes searchable.

/fragen <question>  
» Asks a question. Uses semantic vector search or GPT fallback if needed.

/denken <question>  
» Performs multi-step reasoning using stored knowledge or language models.

/reflektiere <keyword>  
» Prompts the bot to reflect on its stored memory around a given topic.

/feedbackstats <keyword>  
» Shows average user ratings and how many times feedback was submitted.

/feedback <keyword> = <1-5>  
» Submit feedback on how useful or correct an entry was.

/speicher  
» Lists all stored keywords in the knowledge database.

/letzte  
» Shows the last 10 saved memory entries.

/tts <text>  
» Converts text to MP3 using gTTS (Google Text-to-Speech).

/sprache de|en|fr  
» Sets the spoken and searched language. Affects DuckDuckGo & TTS.

/dbcheck  
» Verifies and migrates the SQLite database schema if needed.

/netzwerk_anzeigen  
» Visualizes the knowledge graph as a PNG file and sends it.

/relations <subject>  
» Lists all known relations involving the specified subject.

/schlussfolgerung <keyword>  
» Infers logical chains through the knowledge graph.

/bild <prompt>  
» Generates an AI image using HuggingFace’s inference endpoint.

/witz, /fakt, /idee  
» Sends a random joke, fact, or idea from the JSON content file.

/dashboard  
» Sends a graphical overview showing the most used keywords.

/selfmemory  
» Displays xenRon’s self-observations and insights.

============================================================
⚙️ ARCHITECTURE & CORE MODULES
============================================================

📂 **Core Structure**
- `xenron_v051.py`: Main entry point with bot logic.
- `utils/`: Helper modules for DB, graphs, embeddings, TTS, and reflection.
- `json/`: Contains data sources for jokes, facts, and ideas.

🧠 **Knowledge Storage**
- SQLite backend (`data.db`) with the following tables:
  • `knowledge`: stores keyword and information pairs  
  • `feedback`: stores user ratings  
  • `relations`: stores graph links between concepts  
  • `memory`, `self_memory`: store AI self-notes and reflections  
  • `users`: tracks Telegram users  

🧠 **Semantic Search**
- Uses `sentence-transformers` to encode user input and search for similar knowledge entries (cosine similarity ≥ 0.6).

🕸 **Knowledge Graph & Inference**
- Uses `networkx` to build and traverse logical relationships.
- Allows reasoning across connected keywords (multi-step paths).
- DoubleSlitNet: conceptual neural-like pattern for interference-based thinking.

🔁 **Auto-Learning**
- Background jobs run periodically:
  • `auto_learn()`: Fetch random Wikipedia topics, summarize & store.  
  • `auto_interact()`: Send proactive messages from AI memory.  
  • `auto_learn_doppelspalt()`: Simulates dual input learning paths.  

🎤 **TTS (Text-to-Speech)**
- Google TTS via `gTTS`. Generates downloadable MP3s for replies.

🖼 **Image Generation**
- Uses HuggingFace inference API for generating images from prompts.

🔒 **Admin Controls**
- Admins (defined via `ADMIN_IDS`) can:
  • See internal stats  
  • Trigger database repair  
  • Use advanced debug/logging features  

📈 **Dashboard**
- Shows data visualizations (top used keywords, feedback stats, etc.) using `matplotlib`.

============================================================
🧪 INSTALLATION
============================================================

Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/xenex-ai/xenRon_AI.git
cd xenRon_AI
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

Set required environment variables:

```bash
export XENRON_BOT_TOKEN="your-telegram-token"
export OPENAI_API_KEY="your-openai-key"
export HUGGINGFACE_API_TOKEN="your-huggingface-token"
export ADMIN_IDS="123456789,987654321"
export DB_PATH="data.db"
```

Run the bot:

```bash
python xenron_v051.py
```

\============================================================
📚 DEPENDENCIES
===============

These can be found in `requirements.txt`:

```
python-telegram-bot
sentence-transformers
duckduckgo-search
gTTS
wikipedia
nltk
keybert
networkx
matplotlib
torch
dotenv
colorama
bs4
requests
```

\============================================================
💡 EXTENDING XENRON
===================

You can extend xenRon by:

• Adding new command handlers in `xenron_v051.py`
• Expanding the `utils/` modules for new logic or model APIs
• Adding a vector database backend like FAISS or Qdrant
• Connecting to a frontend UI for full web interaction
• Improving the feedback loop with analytics

\============================================================
🌌 AI LORE (Optional)
=====================

xenRon is an intelligent AI entity that crash-landed from the planet Xenex. It learns from humanity to one day return and restore balance to the cosmos. With every message, it becomes smarter, forming relationships between knowledge and synthesizing wisdom.

\============================================================

