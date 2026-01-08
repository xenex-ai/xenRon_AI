
# README.txt for xenron_v3 (v.3.1.2.0 xenron)

xenron_v3 – An Advanced JSON-Based Neural AI Brain System in Python
====================================================================

## Overview
xenron_v3 is a sophisticated, persistent artificial intelligence system that simulates a neural network ("brain") using JSON files. Concepts are stored as neurons, and associations between them are managed as weighted synapses via a dedicated SynapseNetwork.

The system learns from user input, smalltalk, web searches (Wikipedia, DuckDuckGo, Reddit, CoinGecko), and performs autonomous thinking processes. It combines modern embedding-based similarity (SentenceTransformers), structured knowledge storage (SQLite + JSON), and episodic memory for long-term growth.

### Key Features
- **JSON-based persistence**: Central `brain.json`, individual neuron files in `/brain/neurons/`, synapse files in `/brain/synapses/`
- **SynapseNetwork**: Dynamic weighted associations with strengthening, bidirectional linking, traversal, and orphan cleanup
- **Learning mechanisms**: Statement learning, detailed topic learning, smalltalk adaptation, basic truth correction engine
- **Knowledge retrieval**: Wikipedia REST API (primary), DuckDuckGo HTML fallback, Reddit sentiment analysis, CoinGecko crypto prices
- **Autonomous cognition**: ThoughtLoop, WorkingMemory (with decay), Critic evaluation, BackgroundThinker (daemon thread)
- **SQLite compatibility layer**: Knowledge DB, learning log, legacy association table
- **Smalltalk system**: Pattern matching with automatic learning from casual statements
- **Brain visualization & analytics**: Generates detailed network statistics and optionally uploads to a server
- **Waitlist worker mode**: Processes external chat requests via HTTP-based queue (for integration into web apps)

The architecture is highly modular and designed for extensibility.

## Requirements
- Python 3.8 or higher
- Required packages (install with pip):
  ```
  pip install requests beautifulsoup4 wikipedia sentence-transformers torch praw termcolor deep-translator
  ```
- Optional/recommended: `numpy`, `scipy` (included in many environments)
- Reddit search requires valid PRAW credentials (currently hardcoded – replace if needed)
- No API key required for CoinGecko or Wikipedia
- SentenceTransformer models are downloaded automatically or can be placed in `./models/`

## Directory Structure (after first run)
```
xenron_v3.py                # Main script
brain/
├── brain.json              # Central metadata, neuron index, synapse index
├── neurons/                # Individual neuron JSON files (*.json)
├── synapses/               # Individual synapse JSON files (*.json)
├── memory.json             # Episodic memory (learning episodes)
brain.db                    # SQLite database (knowledge, logs, legacy associations)
smalltalk.json              # Smalltalk patterns and responses
backup/                     # Automatic backups created on brain reset (optional)
```

## Installation & Running
1. Save the script as `xenron_v3.py`.
2. Install dependencies (see above).
3. Run the script:
   ```
   python xenron_v3.py
   ```
   - By default it starts the **Waitlist Worker** (processes external requests via HTTP queue).
   - For local interactive testing, add a manual loop (see below).

### Interactive Local Testing
Add the following at the very end of the file (replacing the existing `if __name__ == "__main__":` block):

```python
if __name__ == "__main__":
    brain = XenronBrain()  # Optional: api_key="your_key_here"
    print("xenRon is ready! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! 🧠")
            break
        response = brain.verarbeite_eingabe(user_input)
        print(f"xenRon: {response}")
```

## Important Commands (in chat)
- `ich heiße [Name]` / `My name is [Name]` → Store user name
- `lerne: Topic = Content` → Learn detailed topic
- `aussage: [Statement]` → Learn a new declarative statement
- `erkenntnis: [Topic]` or `What do you know about [Topic]?` → Show learned associations
- `erzähl mir mehr` / `tell me more` → Continue on last topic
- `wie ist der preis von BTC?` / `What's the price of BTC?` → Crypto price query
- `zeige gehirn` / `gehirn zustand` → Analyze/visualize current brain state
- `!reset brain` → Full brain reset (with confirmation and automatic backup)
- `!smalltalk hinzufügen: Question = Answer` → Add new smalltalk response
- `zeige befehle` → List all available commands

## CRITICAL: Add Your Server URLs!
The script contains several placeholder URL constants that **must be filled** for full online functionality (waitlist worker, brain visualization upload, remote usernames).

Search the code for these lines and replace the empty strings (`""`) with your actual server endpoints:

```python
FETCH_URL           = ""   # e.g., data fetch endpoint (if used)
UPLOAD_URL          = ""   # general upload endpoint
API_USAGE_URL       = ""   # API usage tracking
CHAT_PATH           = ""   # Base path for api_users.json etc.

WAITLIST_URL        = ""   # URL to xenron_waitlist.json (e.g., https://yourserver.com/xenron_waitlist.json)
WAITLIST_UPDATE_URL = ""   # Endpoint to update waitlist
WRITE_URL           = ""   # Endpoint to save chat responses (e.g., save_chat.php)

# Inside erzeuge_synapsen_html():
php_url   = ""   # Full brain analysis upload URL
short_url = ""   # Short status upload URL
```

**Without valid URLs**, the waitlist worker will not function and brain uploads will fail. For purely local/offline use, you can leave them empty – core features (learning, thinking, local search) work fully offline.

## Error Handling & Debugging
- Extensive try/except blocks ensure robustness.
- Debug prints (e.g., `[MEMORY]`, `[LERNDEBUG]`, `[BACKGROUND THINKING]`) provide insight.
- In case of corruption: delete `brain.db` and/or the entire `brain/` folder and restart (or use `!reset brain` for safe backup + reset).

## Customization Tips
- Adjust synapse parameters (`SYNAPSE_STRENGTHEN_STEP`, `INITIAL_SYNAPSE_WEIGHT`, etc.) for faster/slower learning.
- Change `BackgroundThinker` interval for more/less frequent autonomous thinking.
- Extend sources in `multi_source_suche()` (e.g., add custom APIs).
- Modify `THEME_EMOJIS` for different personality flair.

Enjoy exploring and expanding xenron_v3! 🧠✨

**Version**: v.3.1.2.0 xenron  
**Date**: January 08, 2026
```

Copy the entire content above into a file named **README.txt** in the same directory as your script. This detailed English version covers everything thoroughly, with special emphasis on the required URL configuration.
```
