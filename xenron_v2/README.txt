
xenRonAi a Learning AI with Semantic Memory
===========================================

xenRon is an intelligent assistant capable of learning from multiple sources, associating concepts, and answering questions based on stored knowledge. It uses semantic vector embeddings, a local SQLite database, and external APIs to process and understand user input.

Features
--------
- Semantic memory with SentenceTransformer (Multilingual MiniLM)
- Wikipedia, DuckDuckGo, Reddit, and CoinGecko integration
- Word association network (neural-style connections)
- Personal knowledge storage with timestamping
- User identification via API key and name memory
- Learns new words and links them with topics and sources
- GPT-style formatted responses with emojis and formatting
- Personal learning dashboard and word dashboard
- Multi-user and multi-threaded support
- Works with a remote JSON interface (xenexai.com)

Installation
------------
1. Install dependencies:
   pip install -r requirements.txt

   Required packages include:
   - sentence-transformers
   - requests
   - beautifulsoup4
   - wikipedia

2. Place the model in:
   ./models/paraphrase-multilingual-MiniLM-L12-v2/

3. Run xenRon:
   python3 xen_v2_6.py

Folder Structure
----------------
- xen_v2_6.py       - Main program logic
- brain.db          - SQLite database file
- smalltalk.json    - Predefined conversational responses
- models/           - Folder with SentenceTransformer model

Example Commands
----------------
- What is a neutron star?
- Learn about artificial intelligence in detail
- My name is Alex
- What have you learned?
- Show learned words
- What's the price of Bitcoin?

Notes
-----
- Runs as a continuous background process
- Remote APIs are hardcoded to xenexai.com endpoints
- User data is stored per API key for personalization

License
-------
MIT License â€“ use, modify, and share freely.

Author
------
Developed by xenexAi.com
