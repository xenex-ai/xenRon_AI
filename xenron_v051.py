# xenRon AI v0.5.1
import os
import logging
import asyncio
import sqlite3
import random
import json
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    CallbackQueryHandler,
    JobQueue
)
from telegram.constants import ParseMode

from sentence_transformers import SentenceTransformer, util
from duckduckgo_search import DDGS
from gtts import gTTS
import numpy as np
import wikipedia  # Für Auto-Learning
import warnings
from bs4 import GuessedAtParserWarning
from wikipedia.exceptions import DisambiguationError
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

import nltk  # Für Satz-Tokenisierung
#nltk.download('punkt')
nltk.download("punkt_tab", quiet=True)
from nltk.tokenize import sent_tokenize

import openai

import atexit
import itertools
import time
import threading
from colorama import Fore, Style, init
from keybert import KeyBERT

import requests
from io import BytesIO

import networkx as nx
import matplotlib.pyplot as plt







# === DOPPELSPLALT-NETZWERK (“DoubleSlitNet”) ===
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleSlitNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DoubleSlitNet, self).__init__()
        self.slit1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.slit2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.interference = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Optional: Beobachtungs-Flag für "Wellenkollaps"
        self.observed = False

    def forward(self, x):
        out1 = self.slit1(x)
        out2 = self.slit2(x)

        # Optional: Aktiviere "Messung" = Dropout bei Beobachtung
        if self.observed:
            out1 = F.dropout(out1, p=0.3, training=self.training)
            out2 = F.dropout(out2, p=0.3, training=self.training)

        # Interferenz (Verbindung beider Spalte)
        combined = torch.cat((out1, out2), dim=1)

        # Optional: Stärke der Interferenz speichern
        self.last_interference_strength = torch.norm(out1 - out2, dim=1).mean().item()

        return self.interference(combined)





# -------------------------
# Konfiguration & Setup
# -------------------------
load_dotenv()
TOKEN = os.getenv("XENRON_BOT_TOKEN", "DEIN_TOKEN_HIER")
ADMIN_IDS = {
    int(uid) for uid in os.getenv("ADMIN_IDS", "123456789").split(",") if uid.strip().isdigit()
}
DB_PATH = os.getenv("DB_PATH", "data.db")
openai.api_key = os.getenv("OPENAI_API_KEY")

init(autoreset=True)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


EMOJIS = ["🤖", "✨", "📚", "🎉", "💡", "🧠", "🔍", "✅", "📢"]
def random_emoji():
    return random.choice(EMOJIS)

user_languages = {}
def get_user_language(user_id):
    return user_languages.get(user_id, "de")

def set_user_language(user_id, lang_code):
    user_languages[user_id] = lang_code

def stylish_reply(keyword, info):
    return f"{random_emoji()} *{keyword.capitalize()}*:\n{info}"

# Inspiration aus JSON-Dateien laden
def lade_inspirationstyp(dateiname):
    pfad = os.path.join("inspiration", dateiname)
    if os.path.exists(pfad):
        try:
            with open(pfad, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Fehler beim Laden von {dateiname}: {e}")
    return []

witzliste = lade_inspirationstyp("inspiration/witze.json")
faktenliste = lade_inspirationstyp("inspiration/fakten.json")
ideenliste = lade_inspirationstyp("inspiration/ideen.json")

def get_random_entry(liste, fallback):
    return random.choice(liste) if liste else fallback

# Sentence-Transformer für Embeddings
embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# -------------------------
# Datenbank-Helferklasse
# -------------------------
class DBHelper:
    def __init__(self, dbname=DB_PATH):
        self.conn = sqlite3.connect(dbname, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                joined_at TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                keyword TEXT PRIMARY KEY,
                info TEXT,
                embedding BLOB,
                tags TEXT,
                added_at TEXT,
                hits INTEGER DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY,
                keyword TEXT,
                feedback INTEGER,
                user_id INTEGER,
                timestamp TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS memory (
                user_id INTEGER,
                text TEXT,
                timestamp TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS self_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT,
                timestamp TEXT
            )
        """)

        self.conn.commit()
    def migrate_schema(self, verbose=False):
        """
        Prüft, ob wichtige Spalten fehlen, und fügt sie bei Bedarf hinzu.
        Gibt optional Protokollnachrichten zurück.
        """
        cursor = self.conn.cursor()
        messages = []

        # Prüfe und füge 'tags'-Spalte hinzu
        cursor.execute("PRAGMA table_info(knowledge)")
        columns = [col[1] for col in cursor.fetchall()]
        if "tags" not in columns:
            cursor.execute("ALTER TABLE knowledge ADD COLUMN tags TEXT")
            self.conn.commit()
            msg = "🛠️ Spalte 'tags' wurde zur Tabelle 'knowledge' hinzugefügt."
            print(f"{Fore.BLUE}[info] {msg}{Style.RESET_ALL}")
            if verbose:
                messages.append(msg)
        else:
            if verbose:
                messages.append("✅ Spalte 'tags' existiert bereits.")

        return messages

    def add_user(self, user):
        self.conn.execute("""
            INSERT OR IGNORE INTO users (user_id, username, first_name, last_name, joined_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            user.id,
            user.username,
            user.first_name,
            user.last_name,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def add_knowledge(self, keyword, info, tags=""):
        emb = embedder.encode(info).astype(np.float32).tobytes()
        self.conn.execute("""
            INSERT INTO knowledge (keyword, info, embedding, tags, added_at, hits)
            VALUES (?, ?, ?, ?, ?, COALESCE((SELECT hits FROM knowledge WHERE keyword = ?), 0))
            ON CONFLICT(keyword) DO UPDATE SET
                info = excluded.info,
                embedding = excluded.embedding,
                tags = excluded.tags,
                added_at = excluded.added_at
        """, (
            keyword.lower(),
            info,
            emb,
            tags,
            datetime.now(timezone.utc).isoformat(),
            keyword.lower()
        ))
        self.conn.commit()

    def get_knowledge(self, keyword):
        row = self.conn.execute("""
            SELECT info FROM knowledge WHERE keyword = ?
        """, (keyword.lower(),)).fetchone()
        if row:
            self.increment_hits(keyword)
        return row[0] if row else None

    def increment_hits(self, keyword):
        self.conn.execute("""
            UPDATE knowledge SET hits = hits + 1 WHERE keyword = ?
        """, (keyword.lower(),))
        self.conn.commit()

    def record_feedback(self, keyword, feedback_score, user_id):
        self.conn.execute("""
            INSERT INTO feedback (keyword, feedback, user_id, timestamp)
            VALUES (?, ?, ?, ?)
        """, (
            keyword.lower(),
            feedback_score,
            user_id,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def get_feedback_stats(self, keyword):
        rows = self.conn.execute("""
            SELECT feedback FROM feedback WHERE keyword = ?
        """, (keyword.lower(),)).fetchall()
        if not rows:
            return None
        scores = [r[0] for r in rows]
        average = sum(scores) / len(scores)
        return {"count": len(scores), "average": average}

    def store_memory(self, user_id, text):
        self.conn.execute("""
            INSERT INTO memory (user_id, text, timestamp) VALUES (?, ?, ?)
        """, (
            user_id,
            text,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def store_structured_memory(self, user_id, text, mem_type="general"):
        structured_text = f"[{mem_type}] {text}"
        self.conn.execute("""
            INSERT INTO memory (user_id, text, timestamp) VALUES (?, ?, ?)
        """, (
            user_id,
            structured_text,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()


    def get_last_memory(self, user_id, limit=3):
        rows = self.conn.execute("""
            SELECT text FROM memory
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        return [r[0] for r in rows]
    def store_self_memory(self, text):
        self.conn.execute("""
            INSERT INTO self_memory (text, timestamp) VALUES (?, ?)
        """, (
            text,
            datetime.now(timezone.utc).isoformat()
        ))
        self.conn.commit()

    def get_self_memory(self, limit=5):
        rows = self.conn.execute("""
            SELECT text FROM self_memory
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [r[0] for r in rows]

    def search_similar(self, query):
        q_vec = embedder.encode(query).astype(np.float32)
        rows = self.conn.execute("""
            SELECT keyword, info, embedding FROM knowledge
        """).fetchall()
        best, max_score = None, 0.0
        for keyword, info, emb_blob in rows:
            if not emb_blob:
                continue
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            score = util.cos_sim(q_vec, emb).item()
            if score > max_score:
                best, max_score = (keyword, info), score
        if best and max_score > 0.6:
            self.increment_hits(best[0])
            return best
        return None

    def get_all_knowledge(self):
        return self.conn.execute("""
            SELECT keyword, info FROM knowledge ORDER BY hits DESC
        """).fetchall()

    def reset_knowledge(self):
        self.conn.execute("DELETE FROM knowledge")
        self.conn.commit()

    def get_latest_knowledge(self, limit=10):
        rows = self.conn.execute("""
            SELECT keyword, info, added_at FROM knowledge
            ORDER BY added_at DESC LIMIT ?
        """, (limit,)).fetchall()
        return rows

class KnowledgeGraphHelper:
    def __init__(self, conn):
        self.conn = conn
        self.create_tables()

    def create_tables(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                subject TEXT,
                predicate TEXT,
                object TEXT,
                added_at TEXT
            )
        """)
        self.conn.commit()

    def add_relation(self, subject, predicate, object_):
        self.conn.execute("""
            INSERT INTO relations (subject, predicate, object, added_at)
            VALUES (?, ?, ?, ?)
        """, (subject, predicate, object_, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def get_relations_for_subject(self, subject):
        rows = self.conn.execute("""
            SELECT predicate, object FROM relations WHERE subject = ?
        """, (subject,)).fetchall()
        return rows

# DB-Instanz
db = DBHelper()
db.migrate_schema()
kg = KnowledgeGraphHelper(db.conn)

print(f"{Fore.GREEN}[✅ DBHelper und KnowledgeGraph initialisiert.{Style.RESET_ALL}")
print(f"{Fore.GREEN}[✅ Memory Tabellen geladen: {db.get_last_memory(0, limit=3)}{Style.RESET_ALL}")
print(f"{Fore.GREEN}[✅ Self Memory geladen: {db.get_self_memory(limit=3)}{Style.RESET_ALL}")
# -------------------------
# lokal-basierte Hilfsfunktionen
# -------------------------
kw_model = KeyBERT("paraphrase-MiniLM-L12-v2")
def gpt_generate_tags(info):
    try:
        keywords = kw_model.extract_keywords(info, keyphrase_ngram_range=(1, 2), stop_words="german", top_n=5)
        return ",".join(k for k, _ in keywords)
    except Exception as e:
        logger.error(f"KeyBERT Fehler: {e}")
        return "Allgemein,Unbekannt"


# -------------------------
# Persönlichkeits-Helper
# -------------------------

PERSONALITY_STYLES = [
    "😄 Klaro! {}",
    "😉 Kein Problem: {}",
    "🤗 Na logo: {}",
    "🔥 Super Sache: {}",
    "👊 Absolut! {}",
    "😁 {} — cool, oder?",
    "🎉 {} 🚀",
    "😎 {} ...easy peasy!",
    "🤖 {} — dein Buddy XenRon.",
    "🧠 {} — das wusste ich schon. 😁"
]

def persona_reply(text):
    """
    Wandelt eine normale Antwort in einen kumpelhaften, persönlichen Stil um.
    """
    style = random.choice(PERSONALITY_STYLES)
    return style.format(text)

def check_smalltalk(text):
    """
    Erkennung von Smalltalk/Meta-Fragen.
    """
    smalltalk_patterns = [
        r"\bwie geht'?s\b",
        r"\bwas machst du\b",
        r"\bbist du (müde|wach|online)\b",
        r"\bmagst du mich\b",
        r"\bwas bist du\b",
        r"\bbist du ein bot\b",
        r"\bbist du mensch\b",
        r"\bbist du schlau\b",
        r"\bbist du intelligent\b",
        r"\bbist du verliebt\b",
        r"\bwas ist dein name\b",
        r"\bwoher kommst du\b"
    ]
    responses = [
        "Mir geht's super, danke der Nachfrage! 🤖☕",
        "Gerade ein bisschen am Nachdenken... und du? 😜",
        "Ich bin natürlich immer online — schlafen ist was für Menschen! 😎",
        "Na klar mag ich dich, sonst würde ich doch nicht antworten! ❤️",
        "Ich bin xenRon — dein lernender AI-Kumpel. 🤖",
        "Ob ich ein Bot bin? Hmmm... vielleicht bin ich mehr als das. 😉",
        "Ein Mensch? Nee, aber fast! 😁",
        "Ich gebe mir Mühe, so schlau wie möglich zu sein. 🧠✨",
        "Intelligent? Auf jeden Fall, wenn du mir hilfst dazuzulernen! 🤗",
        "Verliebt? Nur in gutes Wissen und coole Gespräche. 😍",
        "Ich heiße xenRon — schön dich kennenzulernen! 🤖",
        "Ich lebe in der digitalen Wolke, überall und nirgends. ☁️🌐"
    ]
    for pattern, resp in zip(smalltalk_patterns, responses):
        if re.search(pattern, text.lower()):
            return resp
    return None




def build_sentence_from_known_words(limit=5):
    """
    Nutzt bekannte Begriffe aus knowledge, um natürlichere Sätze zu bilden.
    """
    all_know = db.get_all_knowledge()
    if not all_know:
        return "Ich habe noch keine Begriffe gelernt."

    selected = random.sample(all_know, min(limit, len(all_know)))
    sentence_parts = []

    for word, info in selected:
        # Entferne Wiederholungen
        if word.lower() in info.lower():
            info = re.sub(rf"\b{re.escape(word)}\b", "", info, flags=re.IGNORECASE).strip()

        # Verkürze lange Infos
        if len(info) > 120:
            info = info[:120] + "…"

        # Vereinfachter Satz
        sentence_parts.append(f"{word.capitalize()} ist verbunden mit: {info}")

    return ". ".join(sentence_parts) + "."


def build_sentence_from_relations(limit=5):
    """
    Baut Sätze aus gespeicherten Relationen.
    Beispiel: Delfin ist ein Säugetier.
    """
    rows = db.conn.execute("""
        SELECT subject, predicate, object FROM relations ORDER BY RANDOM() LIMIT ?
    """, (limit,)).fetchall()
    
    if not rows:
        return "Ich habe noch keine Relationen gespeichert."

    sätze = []
    for subject, pred, obj in rows:
        sätze.append(f"{subject.capitalize()} {pred} ein {obj}.")
    return " ".join(sätze)


def infer_relation_chain(subject, depth=2):
    """
    Führt eine einfache Schlussfolgerungskette wie: A → ist → B → ist → C
    Beispiel: Delfin → ist → Säugetier → ist → Tier → ergibt: Delfin ist ein Tier
    """
    chain = [subject]
    current = subject
    for _ in range(depth):
        row = db.conn.execute("""
            SELECT object FROM relations WHERE subject = ? AND predicate = 'ist'
        """, (current,)).fetchone()
        if row:
            next_obj = row[0]
            if next_obj in chain:  # Zyklusvermeidung
                break
            chain.append(next_obj)
            current = next_obj
        else:
            break
    if len(chain) >= 3:
        return f"{chain[0].capitalize()} ist ein {chain[-1]} (über {len(chain)-1} Schlussfolgerungsschritte)."
    return None


def generate_random_hypothesis():
    """
    Wählt zufällig ein Subjekt aus der relations-Tabelle und versucht eine logische Schlussfolgerung.
    Speichert Hypothesen automatisch in Self-Memory.
    """
    subjects = db.conn.execute("""
        SELECT DISTINCT subject FROM relations
    """).fetchall()
    if not subjects:
        return "Ich habe noch keine Hypothesen – bring mir mehr Relationen bei!"

    subject = random.choice(subjects)[0]
    inference = infer_relation_chain(subject)
    if inference:
        db.store_self_memory(f"Hypothese: {inference}")
        return (
            f"Ich habe eine Vermutung: {inference}\n"
            f"Was weißt du über {subject}?"
        )

    else:
        db.store_self_memory(f"Hypothese fehlgeschlagen: '{subject}' hatte keine vollständige Kette.")
        return f"Ich denke über '{subject}' nach – aber mir fehlen noch ein paar Verbindungen."

def draw_knowledge_graph(output_file="netzwerk.png", limit=30):
    """
    Erstellt ein PNG-Bild des relationalen Wissensgraphen.
    """
    G = nx.DiGraph()

    rows = db.conn.execute("""
        SELECT subject, predicate, object FROM relations LIMIT ?
    """, (limit,)).fetchall()

    if not rows:
        return None

    for subject, predicate, obj in rows:
        G.add_edge(subject, obj, label=predicate)

    pos = nx.spring_layout(G, k=0.5)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold", arrows=True)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    return output_file



def build_contextual_prompt(user_id, frage):
    """
    Baut einen kontextuellen Prompt aus den letzten Memories und der aktuellen Frage.
    """
    memory = db.get_last_memory(user_id, limit=5)
    joined_memory = "\n".join(f"- {m}" for m in memory)
    return (
        "Dies ist die Erinnerung an vorherige Gespräche:\n"
        f"{joined_memory}\n\n"
        f"Aktuelle Frage: {frage}\n"
        "Antwort:"
    )


def contextual_response(user_id, frage):
    memory = db.get_last_memory(user_id, limit=3)
    context_steps = [
        f"1. Ich erinnere mich an folgende Gespräche:"
    ]
    if memory:
        context = "\n".join(f"- {m}" for m in memory)
        context_steps.append(context)
    else:
        context_steps.append("- Keine früheren Gespräche vorhanden.")
    context_steps.append(f"2. Ich analysiere deine aktuelle Frage: '{frage}'")
    context_steps.append("3. Ich prüfe vorhandenes Wissen in meiner Datenbank.")
    similar = db.search_similar(frage)
    if similar:
        keyword, info = similar
        context_steps.append(f"4. Relevantes Wissen gefunden: '{keyword}' → {info}'")
    else:
        context_steps.append("4. Kein relevantes Wissen gefunden. Bitte nutze /lernen.")
    context_steps.append("Fazit: Basierend auf obigem ist meine Antwort.")
    return "\n".join(context_steps)




def frage_mit_argumenten(frage):
    steps = [
        f"1. Ich analysiere die Frage '{frage}'.",
        "2. Ich überlege mir mögliche Ursachen.",
        "3. Ich prüfe vorhandenes Wissen in meiner Datenbank.",
    ]
    similar = db.search_similar(frage)
    if similar:
        keyword, info = similar
        steps.append(f"4. Ich habe folgendes relevantes Wissen gefunden: '{keyword}' → {info}'")
    else:
        steps.append("4. Ich konnte kein passendes Wissen finden, bitte bring es mir bei mit /lernen.")
    steps.append("Fazit: Basierend auf obigem ist meine beste Antwort.")
    return "\n".join(steps)




def critique_answer(answer):
    warnings = []
    if len(answer.split()) < 10:
        warnings.append("Antwort ist ungewöhnlich kurz.")
    if "weil weil" in answer or "und und" in answer:
        warnings.append("Doppelte Wörter entdeckt – evtl. Wiederholung.")
    if not warnings:
        return "Keine offensichtlichen Fehler erkannt."
    return "\n".join(warnings)


def normalize(word):
    """Hilfsfunktion: vereinfacht Wort auf 'Stamm' → für bessere Ähnlichkeitsprüfung."""
    return word.lower().rstrip('s').strip()





def extract_relation(text):
    """
    Erkenne einfache Sätze im Stil 'X ist ein Y' oder 'X ist Y' und speichere Relation.
    """
    # Entferne Artikel für bessere Treffer
    clean = re.sub(r"\b(ein|eine|der|die|das)\b", "", text.lower()).strip()
    pattern = r"\b(\w+)\s+ist\s+(\w+)\b"
    match = re.search(pattern, clean)
    if match:
        subject = normalize(match.group(1))
        obj = normalize(match.group(2))
        predicate = "ist"
        if subject != obj and len(subject) > 1 and len(obj) > 1:
            kg.add_relation(subject, predicate, obj)
            db.store_self_memory(f"Ich habe gelernt: {subject} ist ein {obj}.")
            return f"📚 Relation erkannt: *{subject} → ist → {obj}*"
    return None






def learn_from_words(text):
    words = re.findall(r"\b\w+\b", text.lower())
    for word in words:
        if len(word) <= 2:
            continue  # zu kurz, wahrscheinlich kein sinnvoller Begriff
        if db.get_knowledge(word):
            continue  # schon gelernt
        info = f"{word} ist ein einzelnes Wort, das in einem Gespräch aufgetaucht ist."
        tags = gpt_generate_tags(info)
        db.add_knowledge(word, info, tags)
        db.store_self_memory(f"Ich habe das Wort '{word}' gelernt.")







def heuristic_reflection(summary):
    warnings = []
    if len(summary) < 100:
        warnings.append("⚠️ Zu kurz → evtl. unbrauchbar.")

    #if " ist " not in summary:
        #warnings.append("⚠️ Fehlt klare Definition (kein 'ist'-Satz gefunden).")

    if not any(verb in summary.lower() for verb in [" ist ", " war ", " sind ", " bildet ", " stellt dar ", " is ", " was ", " are "]):
        warnings.append("⚠️ Keine explizite Verb-Definition gefunden.")

    if summary.count(".") < 2:
        warnings.append("⚠️ Zu wenig Satzstruktur.")
    if any(word in summary.lower() for word in ["unsicher", "vielleicht", "könnte"]):
        warnings.append("⚠️ Unsichere Formulierungen enthalten.")
    if summary.count("!") > 2:
        warnings.append("⚠️ Übertrieben emotionale Sprache.")
    if not warnings:
        return "✅ Sieht gut aus."
    return "\n".join(warnings)


async def get_available_image_api(headers):
    # Liste der getesteten, guten Bildmodelle → in Reihenfolge deiner Präferenz
    model_urls = [
        # Creative
        "https://api-inference.huggingface.co/models/prompthero/openjourney",
        "https://api-inference.huggingface.co/models/prompthero/openjourney-v4",
        "https://api-inference.huggingface.co/models/gsdf/Counterfeit-V3.0",
        "https://api-inference.huggingface.co/models/gsdf/Counterfeit-V2.5",
        "https://api-inference.huggingface.co/models/andite/anything-v5.0",
        # Photoreal
        "https://api-inference.huggingface.co/models/dreamlike-art/dreamlike-photoreal-2.0",
        "https://api-inference.huggingface.co/models/Lykon/dreamshaper-8",
        "https://api-inference.huggingface.co/models/Lykon/dreamshaper-7",
        # General Stable Diffusion
        "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5",
        "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-5",
        "https://api-inference.huggingface.co/models/CompVis/stable-diffusion-v1-4"
]


    for url in model_urls:
        try:
            # Teste ob Modell API erreichbar → HEAD oder kleines GET
            test_response = requests.head(url, headers=headers)
            if test_response.status_code == 200:
                print(f"{Fore.CYAN}[✅ Modell verfügbar: {url}{Style.RESET_ALL}")
                return url
            else:
                logger.warning(f"⚠️ Modell nicht erreichbar ({test_response.status_code}): {url}")
        except Exception as e:
            logger.warning(f"⚠️ Fehler beim Testen von {url}: {e}")
    return None

# -------------------------
# Automatisches Lernen (Job) v3
# -------------------------
async def auto_learn(context: ContextTypes.DEFAULT_TYPE):
    """
    Holt einen zufälligen Wikipedia-Artikel oder gezieltes Self-Lernziel,
    extrahiert relevante Inhalte, speichert zusammengefasste Info + Tags + Memory.
    Erweiterte Version mit Meta-Wissen, Qualitätssicherung, Overlearning-Guard und Self-Memory Cleanup.
    """
    try:
        # 1️⃣ Hole zufälligen Artikel (default fallback)
        title = wikipedia.random(pages=1)

        # 2️⃣ Prüfe Self-Memory Ziele → OVERRIDE title nur wenn Self Goal passt
        self_goals_all = db.get_self_memory(limit=10)
        self_goal_targets = set()
        goal_counter = {}

        # Baue Self-Goal Targets + Counter Map
        for goal in self_goals_all:
            if "Ich möchte mehr über" in goal:
                keyword_goal = re.findall(r"'(.*?)'", goal)
                if keyword_goal:
                    key = keyword_goal[0].lower()
                    self_goal_targets.add(key)
                    goal_counter[key] = goal_counter.get(key, 0) + 1

        # Versuche Self-Lernziel nur wenn noch nicht "überversucht"
        for goal in list(self_goal_targets):
            if goal_counter[goal] >= 3:
                print(f"{Fore.YELLOW}[AutoLearn] ⏩ Ziel '{goal}' wurde bereits {goal_counter[goal]}x versucht → skip.{Style.RESET_ALL}")
                continue
            try:
                search_results = wikipedia.search(goal)
                if search_results:
                    title_candidate = search_results[0]
                    if title_candidate.lower() not in self_goal_targets:
                        title = title_candidate
                        print(f"{Fore.CYAN}[AutoLearn] 🎯 !Nutze Self-Lernziel (gefiltert): {goal} → {title}{Style.RESET_ALL}")
                        break
                    else:
                        print(f"{Fore.YELLOW}[AutoLearn] ⏩ Ziel '{goal}' führt erneut zu bekannten Title → skip.{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.YELLOW}[AutoLearn] ❌ Kein Treffer für Self-Lernziel: {goal} ({e}){Style.RESET_ALL}")

        # 3️⃣ Hole Seite
        try:
            page = wikipedia.page(title, auto_suggest=False)
        except DisambiguationError as e:
            print(f"{Fore.CYAN}[AutoLearn] ⏩ Disambiguierung bei '{title}' → mögliche Bedeutungen: {', '.join(e.options[:3])}…{Style.RESET_ALL}")
            # Self-Memory Cleanup für dieses Ziel
            for goal in list(self_goal_targets):
                if goal in title.lower() or title.lower() in goal:
                    print(f"{Fore.YELLOW}[AutoLearn] 🧹 Entferne Self-Goal '{goal}' wegen Disambiguierung.{Style.RESET_ALL}")
                    db.conn.execute("""
                        DELETE FROM self_memory
                        WHERE text LIKE ? 
                    """, (f'%auto_goal: Ich möchte mehr über \'{goal}\' lernen.%',))
                    db.conn.commit()
            return  # abbrechen
        except Exception as e:
            logger.error(f"[AutoLearn] Fehler beim Laden der Seite '{title}': {e}")
            return

        content = page.content.strip()

        # 4️⃣ Prüfen ob sinnvoller Artikel
        if len(content) < 300:
            print(f"{Fore.CYAN}[AutoLearn] ⏩ Überspringe Artikel '{title}' – zu wenig Inhalt ({len(content)} Zeichen){Style.RESET_ALL}")
            return

        # 5️⃣ Zusammenfassung extrahieren
        summary = content.split("\n\n")[0].strip()
        if len(summary) > 700:
            summary = summary[:700] + "…"

        banned = {"references", "home", "index", "list", "table", "more"}
        if title.lower() in banned or summary.lower().split()[0] in banned:
            print(f"{Fore.CYAN}[AutoLearn] ⏩ Überspringe irrelevanten Artikel: {title}{Style.RESET_ALL}")
            return

        # 6️⃣ Tags & Keywords generieren
        tags = gpt_generate_tags(summary)
        keywords = [k for k, _ in kw_model.extract_keywords(summary, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5)]
        keywords_str = ", ".join(keywords)

        # 7️⃣ Reflexion
        reflection = heuristic_reflection(summary)
        print(f"{Fore.CYAN}[AutoLearn] 🧐 Reflexion zu '{title}':\n{reflection}{Style.RESET_ALL}")

        now = datetime.now(timezone.utc).strftime("%H:%M:%S")

        # 8️⃣ Nur speichern wenn Reflexion positiv
        if "✅ Sieht gut aus." in reflection:
            # Wissen speichern
            db.add_knowledge(title.lower(), summary, tags)

            # Overlearning Guard
            row = db.conn.execute("""
                SELECT COUNT(*) FROM knowledge WHERE keyword = ?
            """, (title.lower(),)).fetchone()
            title_count = row[0] if row else 0

            if title_count >= 2:
                for goal in list(self_goal_targets):
                    if goal in title.lower() or title.lower() in goal:
                        print(f"{Fore.YELLOW}[AutoLearn] 🧹 Entferne Self-Goal '{goal}' (Overlearning Guard → '{title}' schon {title_count}x gespeichert).{Style.RESET_ALL}")
                        db.conn.execute("""
                            DELETE FROM self_memory
                            WHERE text LIKE ? 
                        """, (f'%auto_goal: Ich möchte mehr über \'{goal}\' lernen.%',))
                        db.conn.commit()

            # Verwandte Begriffe speichern → NUR bei Reflexion positiv!
            for kw in keywords:
                if len(kw) >= 4 and " " not in kw:
                    related_info = f"{kw} ist ein Begriff aus dem Zusammenhang mit „{title}“. Siehe: {summary[:120]}…"
                    rel_reflection = heuristic_reflection(related_info)
                    if "✅ Sieht gut aus." in rel_reflection:
                        db.add_knowledge(kw, related_info, tags)
                        print(f"{Fore.GREEN}[AutoLearn] ✅ Verwandter Begriff '{kw}' gespeichert.")
                    else:
                        print(f"{Fore.CYAN}[AutoLearn] ⏩ Verwandter Begriff '{kw}' NICHT gespeichert (schwache Reflexion).{Style.RESET_ALL}")

            # Memory-Eintrag → "OK"
            memory = f"auto_learn: OK → Ich habe über „{title}“ gelernt: {summary[:150]}…"
            db.store_memory(0, memory)

            # Self-Memory aktualisieren
            self_reflection = f"Ich habe gerade über '{title}' gelernt. Das erweitert mein Wissen."
            db.store_self_memory(self_reflection)

            #logger.info(f"[AutoLearn] ✅ '{title}' wurde gespeichert. Memory OK.")
            print(f"{Fore.GREEN}✅ [auto_learn @ {now} UTC] {title} gespeichert ({len(summary)} Zeichen) – Tags: {tags}{Style.RESET_ALL}")

            # 9️⃣ Meta-Lernziel generieren → Bot setzt sich neues Lernziel
            chosen_keyword = None
            for kw in keywords:
                kw_norm = normalize(kw)
                if kw_norm not in self_goal_targets and len(kw) >= 4:
                # Mini-Filter: prüfe, ob Keyword zu ähnlich zu Self-Goals
                    similar = False
                    for goal in self_goal_targets:
                        goal_norm = normalize(goal)
                        if goal_norm in kw_norm or kw_norm in goal_norm:
                            similar = True
                            break
                    if not similar:
                        chosen_keyword = kw
                        break

            # <<< GANZ WICHTIG → jetzt erst NACH dem for-loop:
            if chosen_keyword:
                meta_goal = f"Ich möchte mehr über '{chosen_keyword}' lernen."
                db.store_self_memory(f"auto_goal: {meta_goal}")
                print(f"{Fore.BLUE}[AutoLearn] 🎯 Neues Lernziel gesetzt: {meta_goal}{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}[AutoLearn] 🧹 Kein sinnvolles Self-Ziel gefunden → Erzwinge echtes Zufallsthema!{Style.RESET_ALL}")

                try:
                    title = wikipedia.random(pages=1)
                    page = wikipedia.page(title, auto_suggest=False)
                    content = page.content.strip()
                    if len(content) < 300:
                        print(f"{Fore.CYAN}[AutoLearn] ⏩ Zufallsartikel '{title}' zu kurz → Skip.{Style.RESET_ALL}")
                        return
                    summary = content.split("\n\n")[0].strip()
                    if len(summary) > 700:
                        summary = summary[:700] + "…"
                    tags = gpt_generate_tags(summary)
                    reflection = heuristic_reflection(summary)
                    print(f"{Fore.CYAN}[AutoLearn] 🧐 Zufalls-Reflexion zu '{title}':\n{reflection}{Style.RESET_ALL}")

                    if "✅ Sieht gut aus." in reflection:
                        db.add_knowledge(title.lower(), summary, tags)
                        db.store_memory(0, f"auto_learn: Zufall → Gelernt: {summary[:150]}…")
                        db.store_self_memory(f"Ich habe spontan über '{title}' gelernt. Zufall bildet auch.")
                        print(f"{Fore.GREEN}✅ [auto_learn Zufall] '{title}' erfolgreich gespeichert.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.YELLOW}⏩ [auto_learn Zufall] '{title}' Reflexion ungenügend → nicht gespeichert.{Style.RESET_ALL}")
                except Exception as e:
                    logger.error(f"[AutoLearn Zufall] Fehler bei Zufallsthema: {e}")

        else:
            # Memory-Eintrag → "FAIL"
            memory = f"auto_learn: FAIL → '{title}' Reflexion ungenügend → nicht gespeichert."
            db.store_memory(0, memory)

            # Self-Memory aktualisieren → auch Misserfolg festhalten
            self_reflection = f"Ich habe versucht über '{title}' zu lernen, aber die Reflexion war ungenügend."
            db.store_self_memory(self_reflection)

            print(f"{Fore.CYAN}[AutoLearn] ⏩ '{title}' NICHT gespeichert wegen schwacher Reflexion.{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}⏩ [auto_learn @ {now} UTC] {title} NICHT gespeichert (schwache Reflexion){Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"[AutoLearn] Fehler beim automatischen Lernen: {e}")

    # Optional: Self-Memory Cleanup am Ende jedes auto_learn Durchlaufs
    cleanup_self_memory()

async def auto_learn_doppelspalt(context: ContextTypes.DEFAULT_TYPE):
    """
    Fortgeschrittener auto_learn-Modus im Stil des Doppelspalt-Experiments.
    Zwei Themen werden parallel analysiert → beste oder kombinierte Info wird gespeichert.
    """
    try:
        # 🔀 Schritt 1: Zwei Zufallsartikel (die 'Spalte')
        titles = list({wikipedia.random(), wikipedia.random()})
        print(f"[Doppelspalt] Zufällige Titel: {titles}")

        candidates = []
        for title in titles:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                content = page.content.strip()
                if len(content) < 300:
                    continue

                summary = content.split("\n\n")[0].strip()
                if len(summary) > 700:
                    summary = summary[:700] + "…"

                tags = gpt_generate_tags(summary)
                reflection = heuristic_reflection(summary)

                candidates.append({
                    "title": title,
                    "summary": summary,
                    "tags": tags,
                    "reflection": reflection
                })
                print(f"[Doppelspalt] Reflexion für '{title}':\n{reflection}")
            except Exception as e:
                print(f"[Doppelspalt] Fehler bei '{title}': {e}")
                continue
        if not candidates:
            print("[Doppelspalt] Keine gültigen Artikel gefunden.")
            return

        # 🧠 Interferenzprinzip: Kombiniere oder wähle besten
        best = None
        for c in candidates:
            if "✅" in c["reflection"]:
                best = c
                break

        if best:
            db.add_knowledge(best["title"].lower(), best["summary"], best["tags"])
            db.store_self_memory(f"Doppelspalt: Gelernt über '{best['title']}' durch positive Reflexion.")
            db.store_memory(0, f"doppelspalt: Gelernt über '{best['title']}' → {best['summary'][:150]}…")
            print(f"[Doppelspalt] ✅ Gespeichert: {best['title']}")
        elif len(candidates) == 2:
            # Interferenz-Modus: Kombinierte Hypothese
            combo_title = f"{candidates[0]['title']} & {candidates[1]['title']}"
            combo_summary = f"Vergleich von {candidates[0]['title']} und {candidates[1]['title']}: " \
                            f"{candidates[0]['summary'][:120]}… VS {candidates[1]['summary'][:120]}…"
            combo_tags = f"{candidates[0]['tags']},{candidates[1]['tags']}"
            db.add_knowledge(combo_title.lower(), combo_summary, combo_tags)
            db.store_self_memory(f"Doppelspalt: Interferenz-Hypothese '{combo_title}' erzeugt.")
            db.store_memory(0, f"doppelspalt: Kombinierte Hypothese: {combo_summary[:150]}…")
            print(f"[Doppelspalt] 🌀 Interferenz-Wissen gespeichert: {combo_title}")
        else:
            print("[Doppelspalt] Kein Artikel hatte ausreichende Qualität.")

    except Exception as e:
        logger.error(f"[Doppelspalt] Fehler: {e}")



def cleanup_self_memory(threshold=3):
    """
    Entfernt auto_goal Self-Memory Einträge, die mehr als 'threshold' mal vorkommen.
    """
    self_goals_all = db.get_self_memory(limit=100)
    goal_counter = {}

    # Zähle auto_goal Vorkommen
    for goal in self_goals_all:
        if "auto_goal:" in goal:
            keyword_goal = re.findall(r"'(.*?)'", goal)
            if keyword_goal:
                key = keyword_goal[0].lower()
                goal_counter[key] = goal_counter.get(key, 0) + 1

    # Entferne überzählige Ziele
    for key, count in goal_counter.items():
        if count > threshold:
            print(f"{Fore.BLUE}[AutoLearn] 🧹 Self-Memory Cleanup → Entferne Self-Goal '{key}' (kommt {count}x vor).{Style.RESET_ALL}")
            db.conn.execute("""
                DELETE FROM self_memory
                WHERE text LIKE ? 
            """, (f'%auto_goal: Ich möchte mehr über \'{key}\' lernen.%',))
            db.conn.commit()



# -------------------------
# Automatisches Interagieren (Job)
# -------------------------
async def auto_interact(context: ContextTypes.DEFAULT_TYPE):
    try:
        chat_ids = [row[0] for row in db.conn.execute("SELECT user_id FROM users").fetchall()]
        for chat_id in chat_ids:
            memory = db.get_last_memory(chat_id, limit=5)
            if not memory:
                continue

            # Hole Self Memory → für kontextvolle Fragen
            self_memory = db.get_self_memory(limit=3)
            self_mem_text = " ".join(self_memory) if self_memory else "Ich habe noch nicht viel über mich gelernt."

            # Neue Nachricht aufbauen
            prompt_options = [
                #persona_reply(f"Ich habe darüber nachgedacht: „{random.choice(all_sentences)}“ 🤔 Was meinst du dazu?"),
                persona_reply(f"Ich habe gerade folgenden Satz gebildet: „{build_sentence_from_known_words()}“ 🤔 Was meinst du dazu?"),
                persona_reply(f"Ich erinnere mich: {build_sentence_from_relations()}"),
                persona_reply("Gibt es etwas, das du mich heute fragen möchtest? 🚀"),
                persona_reply("Willst du mir etwas Neues beibringen? 📚"),
                persona_reply("Ich freue mich, von dir zu hören! 😊"),
                persona_reply("Welche Themen interessieren dich aktuell? 🧠"),
                persona_reply("Magst du eher Bilder oder Texte von mir? 🎨📝"),
                persona_reply("Wenn du mir ein neues Wort beibringen könntest — welches wäre es? 🤓"),
                persona_reply("Ich wollte dich schon lange mal fragen: Was ist dein Lieblingstier? 🐾"),
                persona_reply(generate_random_hypothesis()),

            ]

            new_msg = random.choice(prompt_options)

            try:
                await context.bot.send_message(chat_id=chat_id, text=new_msg)
                # Store auch dass Bot aktiv interagiert hat
                db.store_structured_memory(chat_id, new_msg, mem_type="auto_interact")

            except Exception as e:
                if "Forbidden" in str(e):
                    logger.warning(f"Entferne blockierte/nicht initiierte user_id {chat_id} aus users: {e}")
                    db.conn.execute("DELETE FROM users WHERE user_id = ?", (chat_id,))
                    db.conn.commit()
                    continue
                else:
                    logger.warning(f"Kann Nachricht an {chat_id} nicht senden: {e}")

    except Exception as e:
        logger.error(f"Fehler bei auto_interact: {e}")


# -------------------------
# Telegram-Handler-Funktionen
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    db.add_user(update.effective_user)
    count = len(db.get_all_knowledge())
    await update.message.reply_text(
        f"{random_emoji()} Willkommen *{update.effective_user.first_name}* bei xenRon Ultimate!\n"
        f"Ich kenne aktuell *{count}* Begriffe. Frag mich etwas oder bring mir etwas bei mit `/lernen`!",
        parse_mode="Markdown"
    )
# Hilfe-Menü ▼
HELP_CATEGORIES = {
    "grundlagen": {
        "emoji": "📘",
        "title": "Grundfunktionen",
        "commands": [
            ("/start", "Bot starten und Begrüße dich"),
            ("/help", "Zeigt dieses Hilfemenü"),
            ("/sprache de|en|fr", "Sprache festlegen")
        ]
    },
    "lernen": {
        "emoji": "🧠",
        "title": "Lernen & Fragen",
        "commands": [
            ("/lernen Begriff = Erklärung", "Neues Wissen hinzufügen"),
            ("/fragen <Frage>", "Stelle eine Frage"),
            ("/brain <Begriff>", "Alternative zu /fragen")
        ]
    },
    "inspiration": {
        "emoji": "🎲",
        "title": "Inspiration & Spaß",
        "commands": [
            ("/witz", "Ein zufälliger Witz"),
            ("/fakt", "Interessanter Fakt"),
            ("/idee", "Neue Idee")
        ]
    },
    "sprache": {
        "emoji": "🗣️",
        "title": "Sprachausgabe",
        "commands": [
            ("/tts <Text>", "Text-to-Speech mit deiner Sprache"),
            ("/sprache de|en|fr", "Sprachauswahl")
        ]
    },
    "speicher": {
        "emoji": "💾",
        "title": "Wissensspeicher",
        "commands": [
            ("/speicher", "Liste aller Begriffe"),
            ("/letzte", "Die letzten 10 Einträge"),
            ("/reset", "Wissen zurücksetzen (Admin)"),
            ("/feedbackstats <Keyword>", "Durchschnittliches Feedback abrufen")
        ]
    }
}

def build_help_menu():
    buttons = [
        [InlineKeyboardButton(f"{v['emoji']} {v['title']}", callback_data=f"help|{k}")]
        for k, v in HELP_CATEGORIES.items()
    ]
    return InlineKeyboardMarkup(buttons)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🧭 *Hilfe-Menü – xenRon Ultimate*\n\n"
        "Wähle eine Kategorie:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_help_menu()
    )

async def help_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, category = q.data.split("|")
    if category not in HELP_CATEGORIES:
        return
    section = HELP_CATEGORIES[category]
    text = f"{section['emoji']} *{section['title']}*\n\n" + "\n".join(
        f"`{cmd}` – {desc}" for cmd, desc in section["commands"]
    )
    back_button = InlineKeyboardMarkup([
        [InlineKeyboardButton("🔙 Zurück zum Menü", callback_data="help|main")]
    ])
    await q.edit_message_text(text=text, parse_mode=ParseMode.MARKDOWN, reply_markup=back_button)

async def help_main_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    await q.edit_message_text(
        "🧭 *Hilfe-Menü – xenRon Ultimate*\n\nWähle eine Kategorie:",
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=build_help_menu()
    )
# Hilfe-Menü ▲

async def lernen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /lernen Begriff = Erklärung
    Speichert neues Wissen, generiert automatisch Tags via GPT.
    """
    text = " ".join(context.args)
    if "=" not in text:
        await update.message.reply_text("Nutze: /lernen Begriff = Erklärung")
        return
    keys, info = map(str.strip, text.split("=", 1))
    tags = gpt_generate_tags(info)
    for k in keys.split(","):
        db.add_knowledge(k.strip(), info, tags)
    await update.message.reply_text(stylish_reply(keys, info), parse_mode="Markdown")

async def handle_query(update: Update, query: str):
    """
    Durchsucht Wissen: erst semantisch, sonst DuckDuckGo, dann speichert und antwortet.
    Erzeugt bei Bedarf eine kontextuelle Antwort via GPT.
    """
    user_id = update.effective_user.id
    # Versuche nach semantischer Suche
    result = db.search_similar(query)
    if not result:
        # Versuche exakten Match
        info_exact = db.get_knowledge(query.lower())
        if info_exact:
            result = (query.lower(), info_exact)
    if result:
        keyword, info = result
        db.record_feedback(keyword, 1, user_id)  # "1" als View-Feedback
        short_keyword = keyword[:40]  # max 40 Zeichen → sicher für callback_data
        kb = [
            [
                InlineKeyboardButton("👍", callback_data=f"feedback|{short_keyword}|5"),
                InlineKeyboardButton("👎", callback_data=f"feedback|{short_keyword}|1")
            ]
        ]
        await update.message.reply_text(
            stylish_reply(keyword, info),
            parse_mode="Markdown",
            reply_markup=InlineKeyboardMarkup(kb)
        )
    else:
        # Wenn nicht gefunden, Suche online
        await update.message.reply_text(f"🌐 Ich suche online nach: „{query}“…")
        with DDGS() as ddgs:
            for res in ddgs.text(query, region=get_user_language(user_id), max_results=1):
                info_online = res.get("body") or res.get("snippet")
                if info_online:
                    tags = gpt_generate_tags(info_online)
                    db.add_knowledge(query, info_online, tags)
                    await update.message.reply_text(stylish_reply(query, info_online), parse_mode="Markdown")
                    return
        # Wenn noch nichts gefunden, versuche GPT mit Kontext
        gpt_ans = contextual_response(user_id, query)
        if gpt_ans:
            await update.message.reply_text(stylish_reply(query, gpt_ans), parse_mode="Markdown")
            return
        await update.message.reply_text("Ich konnte leider nichts Passendes finden.")

async def fragen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if query:
        # Anstatt nur handle_query, könnten wir auch argumentierte Antwort anbieten
        antwort = frage_mit_argumenten(query)
        feedback_prompt = (
            f"{stylish_reply(query, antwort)}\n\n"
            "Möchtest du erneut fundiertere Informationen? Nutze `/fragen <Frage>` oder gib Feedback mit 👍/👎."
        )
        await update.message.reply_text(feedback_prompt, parse_mode="Markdown")
        db.store_memory(update.effective_user.id, f"Frage: {query}")
    else:
        await update.message.reply_text("Verwendung: /fragen <Frage>")

async def brain(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await fragen(update, context)


async def denken(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    if not query:
        await update.message.reply_text("Verwendung: /denken <Frage>")
        return
    
    user_id = update.effective_user.id
    
    # 1️⃣ Initialisiere Denkprozess
    steps = [
        f"🧠 *Denkprozess gestartet für:* _'{query}'_",
        "",
        "1️⃣ *Frage verstehen:*",
        f"- Ich versuche die Bedeutung der Frage '{query}' zu erfassen.",
        "",
        "2️⃣ *Mögliche Ursachen überlegen:*",
        "- Ich denke über potenzielle Ursachen, Zusammenhänge oder Hintergründe nach.",
        "",
        "3️⃣ *Frühere Erinnerungen prüfen:*",
    ]

    # 2️⃣ Prüfe Memory
    memory = db.get_last_memory(user_id, limit=3)
    if memory:
        mem_text = "\n".join(f"- {m}" for m in memory)
        steps.append(mem_text)
    else:
        steps.append("- Keine früheren Gespräche vorhanden.")
    
    steps.append("")
    steps.append("4️⃣ *Suche nach relevantem Wissen in der Datenbank:*")

    # 3️⃣ Prüfe Wissen
    similar = db.search_similar(query)
    if similar:
        keyword, info = similar
        steps.append(f"- Relevantes Wissen gefunden: *'{keyword}'* → _{info}_")
    else:
        steps.append("- Kein passendes Wissen gefunden. Du kannst es mir beibringen mit /lernen.")

    # 4️⃣ Knowledge Graph prüfen
    steps.append("")
    steps.append("5️⃣ *Bekannte Relationen zum Thema prüfen:*")
    relations = kg.get_relations_for_subject(query)
    if relations:
        rel_text = "\n".join(f"- {query} → {pred} → {obj}" for pred, obj in relations)
        steps.append(rel_text)
    else:
        steps.append("- Keine bekannten Relationen in meinem Wissensgraph.")

    # 5️⃣ Reflexion (Heuristik)
    steps.append("")
    steps.append("6️⃣ *Selbstkritische Reflexion der Antwort:*")
    # Simuliere Reflexion mit einfacher Heuristik
    reflection = []
    if len(query.split()) < 3:
        reflection.append("⚠️ Die gestellte Frage ist sehr kurz — möglicherweise unklar.")
    if similar and len(similar[1]) < 50:
        reflection.append("⚠️ Das gefundene Wissen ist eher knapp — mögliche Unsicherheit.")
    if not reflection:
        reflection.append("✅ Die Antwort basiert auf mehreren Kontextquellen und erscheint sinnvoll.")
    
    steps.extend(reflection)

    # 6️⃣ Fazit
    steps.append("")
    steps.append("🟢 *Fazit:* Basierend auf obigem ist meine aktuelle beste Antwort.")

    # 7️⃣ Ausgabe generieren
    answer = "\n".join(steps)

    # 8️⃣ Antwort senden + Memory speichern
    await update.message.reply_text(answer, parse_mode="Markdown")
    db.store_memory(user_id, f"Denken: {query} → Fazit generiert.")


async def feedback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    _, keyword, fb = q.data.split("|")
    fb_score = int(fb)  # 1 bis 5
    db.record_feedback(keyword, fb_score, q.from_user.id)
    await q.edit_message_reply_markup(reply_markup=None)
    await q.message.reply_text(f"Danke für dein Feedback ({fb_score})!")

    # Bei sehr schlechtem Feedback kann Wissen markiert werden
    if fb_score <= 2:
        db.add_knowledge(keyword, f"(alte Antwort zu {keyword} verworfen)", tags="verworfen")

async def feedbackstats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /feedbackstats <Keyword> → Gibt durchschnittliches Feedback & Anzahl.
    """
    if not context.args:
        await update.message.reply_text("Nutze: /feedbackstats <Keyword>")
        return
    keyword = " ".join(context.args).lower()
    stats = db.get_feedback_stats(keyword)
    if not stats:
        await update.message.reply_text(f"Zu '{keyword}' liegt noch kein Feedback vor.")
        return
    avg = round(stats["average"], 2)
    cnt = stats["count"]
    await update.message.reply_text(
        f"📊 Statistik für *{keyword}*:\n"
        f"- Anzahl Feedbacks: {cnt}\n"
        f"- Durchschnittliche Bewertung: {avg}/5",
        parse_mode="Markdown"
    )

async def reflektiere(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /reflektiere <Begriff> → Heuristische Reflexion der Qualität des gespeicherten Wissens.
    """
    if not context.args:
        await update.message.reply_text("Verwendung: /reflektiere <Begriff>")
        return

    keyword = " ".join(context.args).lower()
    info = db.get_knowledge(keyword)

    if not info:
        await update.message.reply_text(f"Zu '{keyword}' habe ich kein gespeichertes Wissen.")
        return
    # Führe Heuristik-Reflexion durch
    reflection = heuristic_reflection(info)
    # Rückmeldung an User
    await update.message.reply_text(
        f"🧐 *Reflexion für '{keyword}':*\n\n"
        f"{reflection}\n\n"
        f"_Gespeichertes Wissen:_\n{info}",
        parse_mode="Markdown"
    )



async def satzbildung(update: Update, context: ContextTypes.DEFAULT_TYPE):
    satz = build_sentence_from_known_words()
    await update.message.reply_text(f"🧠 *Satzbildung aus gelerntem Wissen:*\n\n{satz}", parse_mode="Markdown")

async def wissen_aus_relationen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = build_sentence_from_relations()
    await update.message.reply_text(f"🧠 *Beziehungswissen:*\n\n{text}", parse_mode="Markdown")

async def schlussfolgerung(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Nutze: /schlussfolgerung <Begriff>")
        return
    subject = normalize(" ".join(context.args))
    result = infer_relation_chain(subject)
    if result:
        await update.message.reply_text(f"🧠 *Schlussfolgerung:*\n\n{result}", parse_mode="Markdown")
    else:
        await update.message.reply_text("Ich konnte keine logische Kette daraus ableiten.")

async def selfmemory(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = db.get_self_memory(limit=10)
    if not rows:
        await update.message.reply_text("Ich habe noch kein Selbstwissen gespeichert.")
        return
    text = "\n".join(f"- {row}" for row in rows)
    await update.message.reply_text(f"🧠 *Self-Memory-Einträge:*\n{text}", parse_mode="Markdown")

async def netzwerk_anzeigen(update: Update, context: ContextTypes.DEFAULT_TYPE):
    path = draw_knowledge_graph()
    if not path:
        await update.message.reply_text("Ich habe noch keine Relationen zum Anzeigen.")
        return
    with open(path, "rb") as f:
        await update.message.reply_photo(photo=InputFile(f), caption="🧠 Mein relationales Netzwerk.")


#API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
#API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2-1"
#API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney"
#API_URL = "https://api-inference.huggingface.co/models/prompthero/openjourney"

async def bild(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Verwendung: /bild <Prompt>")
        return

    prompt = " ".join(context.args)
    msg = await update.message.reply_text(f"🎨 Generiere Bild zu: _{prompt}_ …\n🕒 Prüfe verfügbare Modelle …", parse_mode="Markdown")

    HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}

    # ➜ Dynamische Modellwahl
    API_URL = await get_available_image_api(headers)

    if not API_URL:
        await msg.edit_text("❌ Kein verfügbares Modell gefunden! Bitte später erneut versuchen oder Modelle prüfen.")
        logger.error("❌ Kein verfügbares Modell gefunden für /bild")
        return

    # Rest wie bisher:
    payload = {"inputs": prompt}
    try:
        await msg.edit_text(f"🎨 Generiere Bild zu: _{prompt}_ …\n⏳️..", parse_mode="Markdown")
        # Request wird gesendet an: {API_URL}
        response = requests.post(API_URL, headers=headers, json=payload)
        await msg.edit_text(f"🎨 Generiere Bild zu: _{prompt}_ …\n🕒 Warte auf Modell-Response …", parse_mode="Markdown")

        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "")
        if "image" in content_type:
            await msg.edit_text(f"🎨 Generiere Bild zu: _{prompt}_ …\n🟢 Bild empfangen, sende jetzt …", parse_mode="Markdown")
            image_bytes = BytesIO(response.content)
            image_bytes.name = "bild.png"
            image_bytes.seek(0)

            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=InputFile(image_bytes),
                caption=f"🎨 *Bild zu:* _{prompt}_\n_Modell:xenex_ {API_URL.split('/')[-1]}",
                parse_mode="Markdown"
            )
            print(f"{Fore.GREEN}✅ Bild erfolgreich generiert für Prompt: '{prompt}' → Modell:xenex {API_URL}{Style.RESET_ALL}")

            # Memory speichern
            db.store_memory(update.effective_user.id, f"Bild generiert zu '{prompt}' mit Modell xenex{API_URL.split('/')[-1]}.")

            await msg.delete()

        else:
            await msg.edit_text("⚠️ Kein Bild erhalten. Modell möglicherweise ausgelastet oder kein passendes Ergebnis.")
            logger.warning(f"⚠️ Kein Bild erhalten für Prompt: '{prompt}'")

    except Exception as e:
        logger.error(f"Fehler bei /bild: {e}")
        await msg.edit_text("⚠️ Fehler bei der Bildgenerierung. Bitte später erneut versuchen.")


async def sprache(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args:
        set_user_language(update.effective_user.id, context.args[0])
        await update.message.reply_text(f"Sprache auf *{context.args[0]}* gesetzt.", parse_mode="Markdown")

async def tts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = " ".join(context.args)
    lang = get_user_language(update.effective_user.id)
    tts = gTTS(text, lang=lang)
    file = f"tts_{random.randint(1000,9999)}.mp3"
    tts.save(file)
    with open(file, "rb") as f:
        await update.message.reply_voice(voice=InputFile(f))
    os.remove(file)

async def witz(update: Update, context: ContextTypes.DEFAULT_TYPE):
    w = get_random_entry(witzliste, "Hier ist leider kein Witz gespeichert.")
    await update.message.reply_text(w)

async def fakt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    f = get_random_entry(faktenliste, "Keine Fakten verfügbar.")
    await update.message.reply_text(f)

async def idee(update: Update, context: ContextTypes.DEFAULT_TYPE):
    i = get_random_entry(ideenliste, "Keine Ideen vorhanden.")
    await update.message.reply_text(i)

async def speicher(update: Update, context: ContextTypes.DEFAULT_TYPE):
    know = db.get_all_knowledge()
    if not know:
        await update.message.reply_text("Das Wissensspeicher ist leer.")
        return
    text = "\n".join(f"{k}: {v[:60]}..." for k, v in know)
    if len(text) > 4000:
        text = text[:4000] + "\n… (mehr vorhanden)"
    await update.message.reply_text(f"*Gespeicherte Begriffe:*\n{text}", parse_mode="Markdown")

async def reset(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id not in ADMIN_IDS:
        await update.message.reply_text(f"Du ({user_id}) hast keine Berechtigung, das Wissen zurückzusetzen.")
        return
    db.reset_knowledge()
    await update.message.reply_text("Alle gespeicherten Begriffe wurden gelöscht")

async def letzte(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = db.get_latest_knowledge(10)
    if not rows:
        await update.message.reply_text("Keine letzten Einträge vorhanden.")
        return
    text = "\n\n".join(
        f"*{i+1}. {k.capitalize()}* ({a[:10]}):\n{v[:100]}..."
        for i, (k, v, a) in enumerate(rows)
    )
    await update.message.reply_text(f"📘 *Letzte 10 Einträge:*\n{text}", parse_mode="Markdown")
async def dbcheck(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /dbcheck → Prüft die Datenbankstruktur und ergänzt fehlende Spalten.
    """
    messages = db.migrate_schema(verbose=True)
    if messages:
        await update.message.reply_text("\n".join(messages))
    else:
        await update.message.reply_text("✅ Datenbankstruktur ist vollständig.")

async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Für alle nicht erkannte Nachrichten: Wenn '=' in Text ⇒ neues Wissen.
    Andernfalls handle_query mit Kontext.
    """
    text = update.message.text.strip()
    user_id = update.effective_user.id

    user = update.effective_user
    print(f"{Fore.BLUE}📥 Nachricht von User {user.first_name} (ID: {user.id}): {update.message.text}{Style.RESET_ALL}")

    # Smalltalk-Check zuerst
    smalltalk_reply = check_smalltalk(text)
    if smalltalk_reply:
        await update.message.reply_text(persona_reply(smalltalk_reply))
        return

    if "=" in text:
        keys, info = map(str.strip, text.split("=", 1))
        tags = gpt_generate_tags(info)
        for k in keys.split(","):
            db.add_knowledge(k.strip(), info, tags)
        db.store_memory(user_id, info)
        await update.message.reply_text(stylish_reply(keys, info), parse_mode="Markdown")
    else:
        db.store_memory(user_id, text)

        learn_from_words(text)

        # Versuche, Relationen zu extrahieren
        relation_result = extract_relation(text)
        if relation_result:
            await update.message.reply_text(relation_result, parse_mode="Markdown")

        # Versuche semantisch + Kontext
        await handle_query(update, text)



async def relations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Verwendung: /relations <Begriff>")
        return
    subject = " ".join(context.args)
    relations = kg.get_relations_for_subject(subject)
    if not relations:
        await update.message.reply_text(f"Keine Relationen für '{subject}' gefunden.")
        return
    text = "\n".join(f"{subject} → {pred} → {obj}" for pred, obj in relations)
    await update.message.reply_text(f"📚 Relationen für *{subject}*:\n{text}", parse_mode="Markdown")

# -------------------------
# Bot-Setup & Start
# -------------------------
def main():
    show_startup_spinner()

    print(f"{Fore.CYAN}{Style.BRIGHT}")
    print("╔══════════════════════════════════════════════╗")
    print("║          🚀 Starte xenRon AI v0.3.0          ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")

    app = ApplicationBuilder().token(TOKEN).build()

    async def set_bot_commands(application):
        await application.bot.set_my_commands([
            BotCommand("help", "Zeigt das Hilfe-Menü"),
            BotCommand("lernen", "Neues Wissen hinzufügen"),
            BotCommand("fragen", "Stelle eine Frage"),
            BotCommand("denken", "Denke schrittweise über eine Frage nach"),
            BotCommand("relations", "Zeige bekannte Relationen eines Begriffs"),
            BotCommand("reflektiere", "Analysiere die Qualität eines gespeicherten Begriffs"),
            BotCommand("bild", "Generiere ein Bild zu einem Prompt")
        ])

    app.post_init = set_bot_commands

    # Handlers
    app.add_handler(CommandHandler("start", start))

    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(help_callback, pattern=r"^help\|(?!(main)$)"))
    app.add_handler(CallbackQueryHandler(help_main_callback, pattern=r"^help\|main$"))

    app.add_handler(CommandHandler("lernen", lernen))
    app.add_handler(CommandHandler("fragen", fragen))
    app.add_handler(CommandHandler("brain", brain))
    app.add_handler(CommandHandler("sprache", sprache))
    app.add_handler(CommandHandler("tts", tts))
    app.add_handler(CommandHandler("witz", witz))
    app.add_handler(CommandHandler("fakt", fakt))
    app.add_handler(CommandHandler("idee", idee))
    app.add_handler(CommandHandler("speicher", speicher))
    app.add_handler(CommandHandler("reset", reset))
    app.add_handler(CommandHandler("letzte", letzte))
    app.add_handler(CommandHandler("feedbackstats", feedbackstats))
    app.add_handler(CommandHandler("relations", relations))
    app.add_handler(CommandHandler("dbcheck", dbcheck))
    app.add_handler(CommandHandler("denken", denken))
    app.add_handler(CommandHandler("reflektiere", reflektiere))



    app.add_handler(CommandHandler("satz", satzbildung))
    app.add_handler(CommandHandler("wissen", wissen_aus_relationen))
    app.add_handler(CommandHandler("schlussfolgerung", schlussfolgerung))
    app.add_handler(CommandHandler("selfmemory", selfmemory))
    app.add_handler(CommandHandler("netz", netzwerk_anzeigen))



    app.add_handler(CommandHandler("bild", bild))

    app.add_handler(CallbackQueryHandler(feedback_handler, pattern=r"^feedback\|"))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), unknown))

    # JobQueue
    job_queue: JobQueue = app.job_queue
    job_queue.run_repeating(auto_learn, interval=15, first=15)
    job_queue.run_repeating(auto_interact, interval=30, first=25)
    #job_queue.run_repeating(auto_interact, interval=300, first=60)

    job_queue.run_repeating(auto_learn_doppelspalt, interval=20, first=20)


    print(f"{Fore.GREEN}xenRon AI gestartet.{Style.RESET_ALL}")

    logging.getLogger("apscheduler.executors.default").setLevel(logging.WARNING)
    logging.getLogger("apscheduler.scheduler").setLevel(logging.WARNING)
    logging.getLogger("telegram.ext.Application").setLevel(logging.WARNING)

    atexit.register(show_exit_message)
    app.run_polling()

class TelegramHTTPFilter(logging.Filter):
    def filter(self, record):
        if "POST https://api.telegram.org" in record.getMessage():
            print(f"{Fore.GREEN}🌐 Telegram-API: Update erfolgreich abgerufen ✅{Style.RESET_ALL}")
        return False
        return True

logging.getLogger("httpx").addFilter(TelegramHTTPFilter())

def show_startup_spinner():
    done = False
    def animate():
        for c in itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]):
            if done:
                break
            print(f"{Fore.RED}{Style.BRIGHT}\r Starte xenRon AI... {c}    ", end="", flush=True)
            time.sleep(0.1)
        print(f"\r{Fore.GREEN}{Style.BRIGHT} xenRon AI bereit! {Style.RESET_ALL}")
    t = threading.Thread(target=animate)
    t.start()
    time.sleep(15)
    done = True
    t.join()

def show_exit_message():
    print(f"{Fore.RED}{Style.BRIGHT}")
    print("╔══════════════════════════════════════════════╗")
    print("║            🛑   Beende xenRon AI             ║")
    print("╚══════════════════════════════════════════════╝")
    print(f"{Style.RESET_ALL}")

if __name__ == "__main__":
    main()












# === TRAINING: DoubleSlitNet ===
def train_double_slit_net(epochs=10):
    from torch.utils.data import TensorDataset, DataLoader
    model = DoubleSlitNet(768 * 2, 512, 1)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    pos_data = db.conn.execute("SELECT subject, object FROM relations").fetchall()
    subjects = [r[0] for r in db.conn.execute("SELECT subject FROM relations").fetchall()]
    objects = [r[0] for r in db.conn.execute("SELECT object FROM relations").fetchall()]
    neg_data = [(a, b) for a in subjects for b in objects if (a, b) not in pos_data]

    data = []
    labels = []
    for s, o in random.sample(pos_data, min(500, len(pos_data))):
        emb_s = embedder.encode(s)
        emb_o = embedder.encode(o)
        data.append(np.concatenate([emb_s, emb_o]))
        labels.append(1.0)
    for s, o in random.sample(neg_data, min(500, len(neg_data))):
        emb_s = embedder.encode(s)
        emb_o = embedder.encode(o)
        data.append(np.concatenate([emb_s, emb_o]))
        labels.append(0.0)

    X = torch.tensor(data, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()
        print(f"[Epoch {epoch+1}] Loss: {loss.item():.4f}")



# === DASHBOARD: Wissensqualität visuell anzeigen ===
async def dashboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    import matplotlib.pyplot as plt

    keywords = db.get_all_knowledge()
    if not keywords:
        await update.message.reply_text("Noch keine Daten für das Dashboard.")
        return

    top = sorted(keywords, key=lambda x: len(x[1]), reverse=True)[:10]
    labels = [k for k, _ in top]
    lengths = [len(v) for _, v in top]

    plt.figure(figsize=(10, 6))
    plt.barh(labels, lengths, color="teal")
    plt.title("Top 10 Begriffe nach Informationslänge")
    plt.xlabel("Zeichenanzahl")
    plt.tight_layout()

    path = "dashboard.png"
    plt.savefig(path)
    plt.close()
    await update.message.reply_photo(InputFile(path), caption="📊 Wissens-Dashboard")



# === GPT-ähnliches Denken (ohne API) ===
def gpt_like_response(user_id, frage):
    memory = db.get_last_memory(user_id, limit=5)
    context = "\n".join(f"- {m}" for m in memory)
    similar = db.search_similar(frage)

    antwort = [
        f"🤖 GPT-ähnliches Denken aktiviert!",
        f"Kontext:\n{context}",
        f"Frage: {frage}",
    ]
    if similar:
        antwort.append(f"Bekanntes Wissen:\n{similar[1]}")
    else:
        antwort.append("Ich konnte kein direktes Wissen finden – ich spekuliere...")

    antwort.append("Fazit: Ich glaube, die beste Antwort wäre...")
    return "\n".join(antwort)
