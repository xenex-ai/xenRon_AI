#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xenron_v3
JSON-basiertes
Neuron-/Synapsen-Brain (brain/neurons, brain/synapses, brain.json), Assoziationen
werden bevorzugt über das SynapseNetwork verwaltet.
"""

import json
import time
import requests
import random
import sqlite3
import re
import os
import string
import wikipedia
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
import difflib
from sentence_transformers import SentenceTransformer, util
import threading
from functools import lru_cache
import signal
import sys
import praw
import subprocess
from termcolor import colored
from deep_translator import GoogleTranslator
from collections import Counter
import shutil

import math
import logging
import math


# ---------------------------
# Konfiguration / Konstanten
# ---------------------------
FETCH_URL   = ""
UPLOAD_URL  = ""
INTERVAL    = 5
DB_FILENAME = "brain.db"
VERSION = "v.3.1.2.0 xenron"

API_USAGE_URL = ""
CHAT_PATH     = ""

# Brain JSON-Pfade (neu)
ROOT = os.path.abspath(os.path.dirname(__file__))
BRAIN_DIR = os.path.join(ROOT, "brain")
NEURON_DIR = os.path.join(BRAIN_DIR, "neurons")
SYNAPSE_DIR = os.path.join(BRAIN_DIR, "synapses")
BRAIN_FILE = os.path.join(BRAIN_DIR, "brain.json")
SMALLTALK_FILE = os.path.join(ROOT, "smalltalk.json")
MEMORY_FILE = os.path.join(BRAIN_DIR, "memory.json")

# Synapse-Parameter (können nach Bedarf angepasst werden)
MIN_SYNAPSE_WEIGHT = 0.01
MAX_SYNAPSE_WEIGHT = 10.0
INITIAL_SYNAPSE_WEIGHT = 0.6
SYNAPSE_STRENGTHEN_STEP = 0.35
TRAVERSAL_BREADTH = 8
TRAVERSAL_DEPTH = 3

# Theme Emojis
THEME_EMOJIS = ["🤖", "🧠", "✨", "💡", "🌻", "🐝", "🪶", "🌙", "🪴"]

# Ensure brain dirs exist
def ensure_dirs():
    os.makedirs(BRAIN_DIR, exist_ok=True)
    os.makedirs(NEURON_DIR, exist_ok=True)
    os.makedirs(SYNAPSE_DIR, exist_ok=True)
ensure_dirs()

# ---------------------------
# Hilfsfunktionen für Brain JSON
# ---------------------------
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_load_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def safe_save_json(path: str, data) -> bool:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[safe_save_json] Fehler beim Speichern {path}: {e}")
        return False

def slugify(text: str) -> str:
    if not text:
        return ""
    txt = str(text).strip().lower()
    txt = "".join(ch for ch in txt if ord(ch) >= 32)
    txt = re.sub(r"[^\w\s-]", "", txt, flags=re.UNICODE)
    txt = re.sub(r"\s+", "_", txt)
    return txt[:200]

def normalize_concept(word: str) -> str:
    if not word:
        return ""
    w = word.strip()
    w = re.sub(r"[^\wÄÖÜäöüß\- ]", "", w)
    w = w.strip()
    # einfache heuristiken
    wl = w.lower()
    if wl.endswith("s") and len(w) > 3:
        return w[:-1].title()
    return w.title()

# ---------------------------
# FileManager (Brain JSON, Neurons, Synapses)
# ---------------------------
class FileManager:
    def __init__(self):
        self.brain_file = BRAIN_FILE
        self.neuron_dir = NEURON_DIR
        self.synapse_dir = SYNAPSE_DIR
        self.smalltalk_file = SMALLTALK_FILE
        self.memory_file = MEMORY_FILE

        self._pending_delete = None

        if not os.path.exists(self.brain_file):
            self._init_brain()
        if not os.path.exists(self.smalltalk_file):
            safe_save_json(self.smalltalk_file, {})
        if not os.path.exists(self.memory_file):
            safe_save_json(self.memory_file, {"episodes": []})
        self._load()

    def _init_brain(self):
        base = {
            "meta": {"version": VERSION, "created": now_iso(), "modified": now_iso()},
            "neurons": {},      # slug -> metadata
            "synapses": {},     # from_slug -> {to_slug: weight}
            "relation_index": {}
        }
        safe_save_json(self.brain_file, base)

    def _load(self):
        self._data = safe_load_json(self.brain_file, {"meta": {}, "neurons": {}, "synapses": {}, "relation_index": {}})
        if "neurons" not in self._data:
            self._data["neurons"] = {}
        if "synapses" not in self._data:
            self._data["synapses"] = {}
        if "relation_index" not in self._data:
            self._data["relation_index"] = {}

    def persist(self):
        self._data["meta"]["modified"] = now_iso()
        return safe_save_json(self.brain_file, self._data)

    # Neuron-API
    def has_neuron(self, word: str) -> bool:
        slug = slugify(word)
        return bool(slug and slug in self._data["neurons"])

    def get_neuron(self, word_or_slug: str):
        slug = slugify(word_or_slug)
        return self._data["neurons"].get(slug)

    def create_neuron(self, word: str, types: list = None, sentence: str = "", source: str = "user"):
        types = types or []
        slug = slugify(word)
        if not slug:
            raise ValueError("Empty word for neuron")
        if slug in self._data["neurons"]:
            node = self._data["neurons"][slug]
            changed = False
            if sentence and not node.get("sentence"):
                node["sentence"] = sentence
                changed = True
            for t in types:
                if t not in node.get("types", []):
                    node.setdefault("types", []).append(t)
                    changed = True
            if changed:
                node["modified"] = now_iso()
                self.persist()
            return node
        neuron_file = os.path.join(self.neuron_dir, f"{slug}.json")
        node = {
            "word": normalize_concept(word),
            "slug": slug,
            "types": types,
            "sentence_file": neuron_file,
            "sentence": sentence or "",
            "sources": [source],
            "confidence": 0.5,
            "created": now_iso(),
            "modified": now_iso()
        }
        self._data["neurons"][slug] = node
        safe_save_json(neuron_file, {"word": node["word"], "slug": slug, "types": types, "sentence": sentence or "", "created": now_iso(), "source": source})
        self.persist()
        return node

    def update_neuron_sentence(self, slug: str, sentence: str, source: str = "user"):
        if slug not in self._data["neurons"]:
            raise KeyError("Neuron not found: %s" % slug)
        node = self._data["neurons"][slug]
        node["sentence"] = sentence
        node.setdefault("sources", []).append(source)
        node["modified"] = now_iso()
        node["confidence"] = min(1.0, node.get("confidence", 0.5) + 0.05)
        safe_save_json(node["sentence_file"], {"word": node["word"], "slug": slug, "sentence": sentence, "saved": now_iso(), "source": source})
        self.persist()

    # smalltalk
    def load_smalltalk(self):
        return safe_load_json(self.smalltalk_file, {})

    def save_smalltalk(self, data: dict):
        return safe_save_json(self.smalltalk_file, data)

    # memory
    def record_episode(self, ep: dict):
        mem = safe_load_json(self.memory_file, {"episodes": []})
        episodes = mem.get("episodes", [])
        episodes.append(ep)
        # simple cap
        if len(episodes) > 2000:
            episodes = episodes[-2000:]
        mem["episodes"] = episodes
        safe_save_json(self.memory_file, mem)
        print(f"[MEMORY] Episode gespeichert: {ep.get('topic') or ep.get('content')}")









    def last_topic(self) -> str:
        """Gibt das letzte gespeicherte Lern-Thema zurück."""
        try:
            mem = safe_load_json(self.memory_file, {"episodes": []})
            episodes = mem.get("episodes", [])
            if not episodes:
                return None
            for eintrag in reversed(episodes):
                if "topic" in eintrag and eintrag["topic"]:
                    return eintrag["topic"]
            return None
        except Exception:
            return None




    # synapse file helper
    def synapse_filename(self, a_slug: str, b_slug: str) -> str:
        return os.path.join(self.synapse_dir, f"{a_slug}__{b_slug}.json")

    def save_synapse_file(self, a_slug: str, b_slug: str, weight: float):
        fname = self.synapse_filename(a_slug, b_slug)
        safe_save_json(fname, {"from": a_slug, "to": b_slug, "weight": weight, "updated": now_iso()})

# ---------------------------
# SynapseNetwork
# ---------------------------
class SynapseNetwork:
    def __init__(self, fm: FileManager):
        self.fm = fm
        # in-memory edges: from_slug -> {to_slug: weight}
        self._edges = self.fm._data.get("synapses", {})

    def persist(self):
        self.fm._data["synapses"] = self._edges
        self.fm.persist()

    def _ensure_from(self, a_slug: str):
        if a_slug not in self._edges:
            self._edges[a_slug] = {}

    def get_weight(self, a: str, b: str) -> float:
        a_slug = slugify(a)
        b_slug = slugify(b)
        return float(self._edges.get(a_slug, {}).get(b_slug, 0.0))

    def set_weight(self, a: str, b: str, weight: float):
        a_slug = slugify(a)
        b_slug = slugify(b)
        if not a_slug or not b_slug or a_slug == b_slug:
            return
        self._ensure_from(a_slug)
        w = max(MIN_SYNAPSE_WEIGHT, min(MAX_SYNAPSE_WEIGHT, float(weight)))
        self._edges[a_slug][b_slug] = w
        self.fm.save_synapse_file(a_slug, b_slug, w)
        self.persist()

    def add_synapse(self, a: str, b: str, weight: float = None, bidirectional: bool = True):
        if weight is None:
            weight = INITIAL_SYNAPSE_WEIGHT
        a_slug = slugify(a)
        b_slug = slugify(b)
        if not a_slug or not b_slug or a_slug == b_slug:
            return
        current = float(self._edges.get(a_slug, {}).get(b_slug, 0.0))
        neww = (current + weight) if current > 0 else weight
        neww = max(MIN_SYNAPSE_WEIGHT, min(MAX_SYNAPSE_WEIGHT, neww))
        self.set_weight(a_slug, b_slug, neww)
        if bidirectional:
            cur2 = float(self._edges.get(b_slug, {}).get(a_slug, 0.0))
            mirror = (cur2 + weight * 0.7) if cur2 > 0 else weight * 0.7
            mirror = max(MIN_SYNAPSE_WEIGHT, min(MAX_SYNAPSE_WEIGHT, mirror))
            self.set_weight(b_slug, a_slug, mirror)

    def strengthen(self, a: str, b: str, step: float = SYNAPSE_STRENGTHEN_STEP):
        self.add_synapse(a, b, weight=step, bidirectional=True)

    def neighbors(self, slug: str, limit: int = TRAVERSAL_BREADTH):
        slug = slugify(slug)
        if slug not in self._edges:
            return []
        items = sorted(self._edges[slug].items(), key=lambda kv: kv[1], reverse=True)
        return items[:limit]

    def top_associations(self, slug: str, limit: int = 10):
        results = []
        for s, w in self.neighbors(slug, limit):
            node = self.fm._data["neurons"].get(s)
            if not node:
                nf = os.path.join(self.fm.neuron_dir, f"{s}.json")
                if os.path.exists(nf):
                    node = safe_load_json(nf, {})
                    if node:
                        self.fm._data["neurons"][s] = node
            if node and node.get("word"):
                results.append((node["word"], float(w), s))
            else:
                continue
        return results

    def traverse(self, start: str, depth: int = TRAVERSAL_DEPTH, breadth: int = TRAVERSAL_BREADTH):
        start_slug = slugify(start)
        if not start_slug or start_slug not in self.fm._data["neurons"]:
            return []
        from collections import deque
        visited = {}
        q = deque()
        q.append((start_slug, 1.0, 0))
        results = []
        while q:
            s, score, d = q.popleft()
            if s in visited and visited[s] >= score:
                continue
            visited[s] = score
            node = self.fm._data["neurons"].get(s)
            if node:
                results.append((s, node["word"], score, d))
            if d >= depth:
                continue
            for nb_slug, w in self.neighbors(s, limit=breadth):
                new_score = score * (1.0 + float(w))
                q.append((nb_slug, new_score, d + 1))
        return sorted(results, key=lambda x: (-x[2], x[3]))

    def orphan_cleanup(self):
        changed = False
        for a in list(self._edges.keys()):
            for b in list(self._edges[a].keys()):
                if b not in self.fm._data["neurons"]:
                    nf = os.path.join(self.fm.neuron_dir, f"{b}.json")
                    if not os.path.exists(nf):
                        del self._edges[a][b]
                        changed = True
            if not self._edges[a]:
                del self._edges[a]
                changed = True
        if changed:
            self.persist()

# ---------------------------
# vorhandene xenron_v4_5_5 code (angepasst)
# ---------------------------

variantenA = [
    "Hey, darf ich erfahren wie du heißt?",
    "Herzlich willkommen bei xenexAi! Ich bin xenRon, wie heißt du?",
    "Hallo, schön dich zu sehen! Wie darf ich dich nennen?",
    "Willkommen, wie heißt du?",
    "Hi! Ich bin neugierig – wie heißt du?",
    "Schön, dass du da bist! Wie kann ich dich ansprechen?",
    "Wie darf ich dich nennen?",
    "Darf ich deinen Namen wissen?",
    "Hast du mir auch deinen Namen verraten?",
    "Wie soll ich dich nennen, Freund:in?",
    "Ich würde dich gern beim Namen nennen – wie lautet er?",
    "Du hast einen Namen, oder? 😄",
    "Nenn mir deinen Namen und wir legen los!",
    "Sag mir bitte deinen Namen, damit ich dich richtig ansprechen kann.",
    "Wie heißt du denn mit Vornamen?",
    "Ich bin xenRon – und du?",
    "Lass uns nicht lange förmlich bleiben – wie heißt du?",
    "Hey du! Wie darf ich dich nennen?",
    "Bevor wir starten: Wie heißt du?",
    "Magst du mir deinen Namen verraten?"
]













class XenronBrain:


    _model = SentenceTransformer("all-MiniLM-L6-v2")  # EINMAL global laden


    def __init__(self, db_name=DB_FILENAME, api_key=None):
        # SQLite-DB (wie bisher) — check_same_thread False für Threads
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.api_key = api_key
        self.last_topic = None
        self.user_name = None
        self.smalltalk = {}
        self._init_db()
        # sprache
        self.sprache = "en"  # default

        # 🧠 Denkarchitektur
        self.working_memory = WorkingMemory()
        self.thought_loop = ThoughtLoop(self)
        self.critic = Critic()

        # 🧠 Hintergrunddenken starten
        self.background_thinker = BackgroundThinker(self, interval=45)
        self.background_thinker.start()



        # 🧠 Ziel-Neuronen
        for ziel in ["neugier", "widerspruch", "offene_fragen"]:
            self.fm.create_neuron(
                ziel,
                types=["goal"],
                sentence=f"Internes Ziel: {ziel}",
                source="system"
            )



        # SentenceTransformer (wie bisher)
        try:
            # Lokales Model-Pfad (falls vorhanden). Fallback auf name-string möglich.
            self.model = SentenceTransformer('./models/paraphrase-multilingual-MiniLM-L12-v2')
        except Exception:
            try:
                self.model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            except Exception:
                self.model = None

        # Neuer JSON-FileManager + SynapseNetwork (neu)
        self.fm = FileManager()
        self.sn = SynapseNetwork(self.fm)

        # load smalltalk via fm for consistency
        self.smalltalk = self.fm.load_smalltalk()
        # remote usernames (wie vorher)
        self.remote_usernames = self.lade_remote_usernames()

        if api_key:
            self.user_name = self.remote_usernames.get(api_key)
            if not self.user_name:
                self.user_name = self.benutzername_von_api_key(api_key)

    # ---------------------------
    # unveränderte DB-Init / Tabellen (wie vorher)
    # ---------------------------
    def _init_db(self):
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS benutzer (
                api_key TEXT PRIMARY KEY,
                name     TEXT
            )
        """)
        # Sicherstellen, dass die alten Tabellen existieren (wie zuvor)






        c.execute("""
            CREATE TABLE IF NOT EXISTS wissen (
                frage TEXT NOT NULL,
                antwort TEXT,
                timestamp TEXT,
                api_key TEXT NOT NULL
            )
        """)

        # BESSERE Dublettenbereinigung (Window Function = massiv schneller)
        try:
            c.execute("""
                DELETE FROM wissen
                WHERE rowid NOT IN (
                    SELECT rowid FROM (
                        SELECT rowid,
                               ROW_NUMBER() OVER (PARTITION BY frage, api_key ORDER BY rowid) AS rn
                        FROM wissen
                    )
                    WHERE rn = 1
                )
            """)
            self.conn.commit()
            print("[DB-Migration] Dubletten effizient bereinigt (Window Function).")
        except Exception as e:
            print(f"[DB-Migration Warnung] Dublettenbereinigung fehlgeschlagen: {e}")

        # Index
        try:
            c.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_wissen_frage_api
                ON wissen(frage, api_key)
            """)
            print("[DB] UNIQUE-Index vorhanden.")
        except Exception as e:
            print(f"[DB Patch Warnung] UNIQUE-Index konnte nicht angelegt werden: {e}")









        c.execute("""
            CREATE TABLE IF NOT EXISTS lernprotokoll (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thema TEXT,
                quelle TEXT,
                inhalt TEXT,
                timestamp TEXT,
                api_key TEXT DEFAULT ''
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS gelernte_woerter (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wort TEXT,
                bedeutung TEXT,
                thema TEXT,
                quelle TEXT,
                timestamp TEXT,
                api_key TEXT DEFAULT ''
            )
        """)
        # früher: assoziationen Tabelle — belassen für Abwärtskompatibilität, aber wir nutzen jetzt SynapseNetwork
        c.execute("""
            CREATE TABLE IF NOT EXISTS assoziationen (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wort1 TEXT,
                wort2 TEXT,
                gewicht INTEGER DEFAULT 1,
                api_key TEXT DEFAULT ''
            )
        """)





        self.conn.commit()





    # ===========================
    # 🧠 GOAL SELECTION
    # ===========================

    def pick_goal_neuron(self):
        """
        Wählt ein Ziel-Neuron als Denkimpuls.
        """

        goals = ["neugier", "widerspruch", "offene_fragen"]
        weighted = []

        for g in goals:
            slug = slugify(g)
            assoc = self.sn.top_associations(slug, limit=5)
            score = sum(w for _, w, _ in assoc) if assoc else 0.1
            weighted.append((g, score))

        # höchstes Bedürfnis gewinnt
        weighted.sort(key=lambda x: x[1], reverse=True)
        return weighted[0][0]






    # ===========================
    # 🧠 SELBSTÄNDIGES DENKEN
    # ===========================

    def denke(self, start_thema: str = None):
        """
        Führt einen vollständigen autonomen Denkprozess aus.
        """

        # 🎯 Ziel-getriebenes Denken
        if start_thema in ["neugier", "widerspruch", "offene_fragen"]:
            related = self.sn.top_associations(slugify(start_thema), limit=3)
            if related:
                start_thema = related[0][0]


        seed = start_thema or self.last_topic or self.fm.last_topic()
        if not seed:
            return "Ich habe keinen Ausgangspunkt zum Nachdenken."

        # Denkzyklus starten
        thoughts = self.thought_loop.start(seed)

        # Bewertung
        result = self.critic.evaluate(thoughts, self.working_memory)

        # 🔁 Denk-Ergebnis in Wissen überführen
        if result:
            self.speichere_denk_ergebnis(seed, result)


        # Memory loggen
        self.fm.record_episode({
            "type": "thinking",
            "seed": seed,
            "thoughts": thoughts,
            "result": result,
            "timestamp": now_iso()
        })

        if not result:
            return "Ich habe nachgedacht, aber keine klare Richtung gefunden."

        return (
            f"Ich habe darüber nachgedacht.\n\n"
            f"Ausgangspunkt: **{seed}**\n"
            f"Ergebnis meines Denkens: **{result['selected']}** "
            f"(Kohärenz {result['score']})"
        )




    # ---------------------------
    # Remote usernames
    # ---------------------------
    def lade_remote_usernames(self):
        try:
            r = requests.get(f"{CHAT_PATH}/api_users.json", timeout=5)
            r.raise_for_status()
            data = r.json()
            return data
        except Exception as e:
            print(f"[Fehler beim Laden remote usernames]: {e}")
            return {}

    # ---------------------------
    # Smalltalk
    # ---------------------------
    def lade_smalltalk(self):
        try:
            with open("smalltalk.json", "r", encoding="utf-8") as f:
                raw = json.load(f)
            norm = {}
            for k, v in raw.items():
                key = k.lower().translate(str.maketrans('', '', string.punctuation)).strip()
                norm[key] = v
            return norm
        except Exception as e:
            print(f"[Fehler beim Laden von smalltalk.json]: {e}")
            return {}

    def setze_benutzername_db(self, name, api_key):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO benutzer (api_key, name)
            VALUES (?, ?)
            ON CONFLICT(api_key) DO UPDATE SET name = excluded.name
        """, (api_key, name))
        self.conn.commit()
        self.user_name = name

    def benutzername_von_api_key(self, api_key):
        try:
            c = self.conn.cursor()
            c.execute("SELECT name FROM benutzer WHERE api_key = ?", (api_key,))
            row = c.fetchone()
            if row:
                return row[0]
            else:
                return None
        except Exception as e:
            print(f"[DB Fehler – Name]: {e}")
            return None

    def aktualisiere_remote_username(self, neuer_name):
        try:
            url = f""
            params = {"apikey": self.api_key, "name": neuer_name}
            r = requests.post(url, data=params, timeout=5)
            r.raise_for_status()
            self.remote_usernames[self.api_key] = neuer_name
            self.setze_benutzername_db(neuer_name, self.api_key)
            self.user_name = neuer_name
            return True
        except Exception as e:
            print(f"[Fehler beim Aktualisieren remote username]: {e}")
            return False




    # ---------------------------
    # Wissenssuche / Wikipedia / DDG / Reddit / CoinGecko (wie früher)
    # ---------------------------
    @lru_cache(maxsize=512)
    def wikipedia_suche(self, thema: str) -> str:
        thema = thema.strip().title().rstrip("!?.,")
        if not thema:
            return "xKein Thema angegeben."
        url = f"https://de.wikipedia.org/api/rest_v1/page/summary/{thema.replace(' ', '_')}"
        headers = {"User-Agent": "xenRonBot/1.0"}
        extract = ""
        try:
            response = requests.get(url, headers=headers, timeout=5)
            status = response.status_code
            if status == 200:
                data = response.json()
                extract = data.get("extract", "").strip()


                if extract:
#                    self.speichere_wissen(frage=thema, antwort=extract)

                    self.speichere_wissen(frage=f"was ist {thema.lower()}", antwort=extract, überschreiben=True)


                    # 💡 Neue Logik: Frage für spätere Aktualisierung merken
                    self._pending_refresh = {"keyword": thema}
                    return extract + "\n\n✨ Willst du, dass ich dir noch *neuere* Infos dazu suche?"





                if "title" in data:
                    extract = f"{data['title']} ist ein Begriff, zu dem ich keinen ausführlichen Wikipedia-Text finden konnte."
                    return extract
        except requests.exceptions.Timeout:
            print(f"⏱️ Wikipedia TIMEOUT bei Thema «{thema}»")
        except Exception as e:
            print(f"🚫 Wikipedia-Request-Fehler: {e}")

        # Fallbacks (vereinfachte)
        ddg = self.google_scrape(thema)


        if ddg:
            self.speichere_wissen(frage=thema, antwort=ddg)
            self._pending_refresh = {"keyword": thema}
            return ddg + "\n\n✨ Willst du, dass ich dir noch *neuere* Infos dazu suche?"




        return "xIch konnte dazu leider nichts finden."





    def google_scrape(self, query: str) -> str:
        """
        Fallback-Suchfunktion, wenn Wikipedia nichts liefert.
        Nutzt DuckDuckGo-HTML-Suche für kompakte Textresultate.
        """
        try:
            url = f"https://duckduckgo.com/html/?q={requests.utils.quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0 (compatible; XenronBot/1.0)"}
            r = requests.get(url, headers=headers, timeout=6)
            soup = BeautifulSoup(r.text, "html.parser")

            results = []
            for res in soup.select(".result__snippet"):
                text = res.get_text(" ", strip=True)
                if text:
                    results.append(text)

            if not results:
                return ""
            snippet = " ".join(results[:3])
            self.speichere_wissen(frage=query, antwort=snippet)
            return snippet

        except Exception as e:
            print(f"⚠️ google_scrape Fehler: {e}")
            return ""












    @lru_cache(maxsize=256)
    def reddit_suche(self, thema: str) -> str:
        """
        Hochpräzise Reddit-Analyse (Enterprise Edition):
        - Kontextuelle Embeddings
        - Hybrid-Relevanzmatrix (Titel/Body gewichtet)
        - Sentimentklassifikation mit linguistischen Merkmalen
        - Verknüpfung in interne Wissensgraphstruktur
        - Wissenschaftlich formulierte, strukturierte Ausgabe
        """

        try:
            # -------------------------------
            # Modellinitialisierung
            # -------------------------------
            if not hasattr(self, "_reddit_model"):
                try:
                    self._reddit_model = SentenceTransformer("all-MiniLM-L6-v2")
                except Exception:
                    return (
                        "Das semantische Modell konnte nicht geladen werden. "
                        "Die Analyse wurde abgebrochen."
                    )

            model = self._reddit_model

            # -------------------------------
            # Reddit API
            # -------------------------------
            reddit = praw.Reddit(
                client_id="KF6ADEqnztXQsC2b_0Oq9Q",
                client_secret="RJ0Jd81EYsU6SgJWbxEDSGCOjgWJ2A",
                user_agent="xenexAi/1.0-synapse-enterprise"
            )

            thema_clean = thema.strip()
            posts = list(reddit.subreddit("all").search(thema_clean, limit=60))

            if not posts:
                return f"Für den Suchbegriff „{thema_clean}“ wurden auf Reddit keine verwertbaren Inhalte gefunden."

            # -------------------------------
            # Embedding-Relevanz
            # -------------------------------
            q_emb = model.encode(thema_clean, convert_to_tensor=True)
            scored = []

            for p in posts:
                title = (p.title or "").strip()
                body = (p.selftext or "").strip().replace("\n", " ")

                if not title and not body:
                    continue

                t_emb = model.encode(title or "", convert_to_tensor=True)


                b_emb= self.embed(body[:250])


#                b_emb = model.encode(body[:500] or "", convert_to_tensor=True)

                # Hybridrelevanz (wissenschaftlich sinnvoll)
                score = (
                    float(util.pytorch_cos_sim(q_emb, t_emb)) * 0.75 +
                    float(util.pytorch_cos_sim(q_emb, b_emb)) * 0.25
                )

                scored.append((score, p))

            if not scored:
                return f"Reddit lieferte Inhalte, jedoch keine inhaltlich relevanten Treffer zu „{thema_clean}“."

            # Top N extrahieren
            scored = sorted(scored, key=lambda x: x[0], reverse=True)[:10]

            # -------------------------------
            # Linguistische Sentimentanalyse
            # -------------------------------
            positive = {"good", "strong", "reliable", "increase", "growth", "improving"}
            negative = {"bad", "risk", "decline", "warning", "scam", "critical", "failure"}

            stimmungen = []
            beiträge = []

            for score, p in scored:
                title = p.title or ""
                body = p.selftext or ""
                text_combined = (title + " " + body).lower()

                pos_score = sum(1 for w in positive if w in text_combined)
                neg_score = sum(1 for w in negative if w in text_combined)

                if pos_score > neg_score:
                    mood = "positiv"
                elif neg_score > pos_score:
                    mood = "negativ"
                else:
                    mood = "neutral"

                stimmungen.append(mood)

                beiträge.append(
                    f"- Titel: **{title}**\n"
                    f"  Relevanz: {score:.3f}\n"
                    f"  Einschätzung: {mood}\n"
                    f"  Quelle: https://reddit.com{p.permalink}"
                )

                # -------------------------------
                # Wissensgraph-Verknüpfung
                # -------------------------------
                try:
                    tokens = [t for t in title.lower().split() if len(t) >= 4]
                    for tok in tokens[:5]:
                        self.fm.create_neuron(tok, types=["reddit", "concept"], source="reddit")
                        self.sn.add_synapse(thema_clean, tok, weight=0.8)
                except Exception:
                    pass

            # -------------------------------
            # Statistische Gesamtbewertung
            # -------------------------------
            cnt = Counter(stimmungen)
            n = len(stimmungen)

            verteilung = ", ".join(
                f"{k}: {round((v / n) * 100)}%" for k, v in cnt.items()
            )

            dominant = max(cnt.items(), key=lambda x: x[1])[0]

            # -------------------------------
            # Finale wissenschaftliche Ausgabe
            # -------------------------------
            return (
                f"Analyse aktueller Reddit-Diskussionen zu „{thema_clean}“\n\n"
                f"**Gesamtbewertung der Stimmung:** {dominant}\n"
                f"**Verteilung:** {verteilung}\n\n"
#                f"**Inhaltlich relevante Beiträge:**\n"
#                f"{chr(10).join(beiträge)}\n\n"
                f"**Interne Verarbeitung:**\n"
                f"Begriffliche Kernelemente wurden identifiziert, linguistisch normalisiert und "
                f"in den konzeptuellen Wissensgraph eingeordnet, um die themenspezifische "
                f"Antwortqualität dauerhaft zu erhöhen."
            )

        except Exception as e:
            return f"Die Analyse konnte aufgrund eines unerwarteten Fehlers nicht abgeschlossen werden: {str(e)}"













    @lru_cache(maxsize=256)
    def coingecko_suche(self, thema: str) -> str:
        try:
            search_url = f"https://api.coingecko.com/api/v3/search?query={thema}"
            search_res = requests.get(search_url, timeout=5).json()
            coins = search_res.get("coins", [])
            if not coins:
                return ""
            coin_id = coins[0]["id"]
            coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&market_data=true"
            coin_res = requests.get(coin_url, timeout=5).json()
            name = coin_res["name"]
            desc = coin_res["description"]["en"][:300].strip().replace('\n', ' ')
            price = coin_res["market_data"]["current_price"]["eur"]
            rank = coin_res["market_data"]["market_cap_rank"]
            homepage = coin_res["links"]["homepage"][0]
            return f"{name} auf Platz {rank}, Preis ~ {price:.2f} EUR.\n\n{desc}\n🔗 {homepage}"
        except Exception as e:
            print(f"⚠️ CoinGecko Fehler {e}")
            return ""



    def krypto_preis_kompakt(self, keyword: str) -> str:
        try:
            keyword_clean = re.sub(r'[^\w\s]', '', keyword.lower().strip())
            search_url = f"https://api.coingecko.com/api/v3/search?query={keyword_clean}"
            search_res = requests.get(search_url, timeout=5).json()
            coins = search_res.get("coins", [])
            if not coins:
                return f"❌ Coin {keyword_clean} nicht gefunden."
            coin_id = coins[0]["id"]
            coin_name = coins[0]["name"]
            symbol = coins[0]["symbol"].upper()
            coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&market_data=true"
            coin_res = requests.get(coin_url, timeout=5).json()
            price_eur = coin_res["market_data"]["current_price"]["eur"]
            change_24h = coin_res["market_data"]["price_change_percentage_24h"]
            emoji = "📈" if change_24h >= 0 else "📉"
            return f"{emoji} {coin_name} ({symbol}) ~ {price_eur:.2f} € ({change_24h:+.2f}% 24h)"
        except Exception as e:
            print(f"[Preis kompakt Fehler]: {e}")
            return f"❌ Preis von {keyword} konnte nicht abgerufen werden."

    # ---------------------------
    # Wissensspeicherung (DB)
    # ---------------------------
    def finde_wissen(self, frage, fuzzy=True):
        c = self.conn.cursor()
        c.execute("SELECT antwort FROM wissen WHERE frage = ? AND api_key = ?", (frage, self.api_key))
        row = c.fetchone()
        if row:
            return row[0]
        if fuzzy:
            pattern = f"%{'%'.join(frage.lower().split())}%"
            c.execute("SELECT frage, antwort FROM wissen WHERE LOWER(frage) LIKE ? AND api_key = ?", (pattern, self.api_key))
            row = c.fetchone()
            if row:
                return row[1]
        return None









    def speichere_wissen(self, frage, antwort, überschreiben=False):
        """
        PATCH A:
        - Korrekte ON CONFLICT Logik mit (frage, api_key)
        - Kein Verlust bestehender Daten
        - JSON-Neuron + Synapsen-Injektion bleibt vollständig erhalten
        - Fallback-Sicherheit, falls Index fehlt oder DB beschädigt ist
        """

        frage = (frage or "").strip().lower()   # <-- normalize key
        

        timestamp = datetime.now(timezone.utc).isoformat()
        c = self.conn.cursor()

        try:
            if überschreiben:
                # Korrekte konfliktbehandlung
                c.execute("""
                    INSERT INTO wissen (frage, antwort, timestamp, api_key)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(frage, api_key) DO UPDATE SET
                        antwort = excluded.antwort,
                        timestamp = excluded.timestamp
                """, (frage, antwort, timestamp, self.api_key))
            else:
                # erzeugt Eintrag nur wenn er nicht existiert
                c.execute("""
                    INSERT OR IGNORE INTO wissen (frage, antwort, timestamp, api_key)
                    VALUES (?, ?, ?, ?)
                """, (frage, antwort, timestamp, self.api_key))

            self.conn.commit()

        except Exception as e:
            print(f"[speichere_wissen] DB-Fehler: {e}")
            try:
                # letzter Fallback: normaler Insert (kann Doppelungen erzeugen, aber sichert Daten)
                c.execute("INSERT INTO wissen (frage, antwort, timestamp, api_key) VALUES (?, ?, ?, ?)",
                          (frage, antwort, timestamp, self.api_key))
                self.conn.commit()
            except Exception as e2:
                print(f"[speichere_wissen] Fallback-Insert Fehler: {e2}")

        # --- JSON-BRAIN & SYNAPSEN BLEIBEN KOMPLETT ERHALTEN ---
        try:
            # 1. Neuron für Frage/Thema anlegen
            self.fm.create_neuron(
                frage,
                types=["knowledge"],
                sentence=antwort,
                source="wissen"
            )

            # 2. Synapsen zu Wissenskonzepten stärken
            self.verknuepfe_woerter(frage, "wissen", weight=0.8)
            self.verknuepfe_woerter(frage, "lernen", weight=0.6)
            self.verknuepfe_woerter(frage, "thema", weight=0.5)

            # 3. Letztes Thema aktualisieren
            self.last_topic = frage

        except Exception as e:
            print(f"[JSON-Wissensinjektion Fehler]: {e}")













    def logge_lernschritt(self, thema, quelle, inhalt):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO lernprotokoll (thema, quelle, inhalt, timestamp)
            VALUES (?, ?, ?, ?)
        """, (thema, quelle, inhalt, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def lerne_wort(self, wort: str, bedeutung: str, thema: str = "", quelle: str = "Unbekannt"):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO gelernte_woerter (wort, bedeutung, thema, quelle, timestamp, api_key)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            wort.strip().lower(),
            bedeutung.strip(),
            thema.strip(),
            quelle,
            datetime.now(timezone.utc).isoformat(),
            self.api_key
        ))
        self.conn.commit()
        # Automatic: also create neuron + connect to thema (if provided)
        if thema:
            for teil in thema.lower().split():
                self.verknuepfe_woerter(wort, teil)










    # ---------------------------
    # 🧠 Erweiterte Lernfunktionen (Synapsen-integriert)
    # ---------------------------

    def lerne_ausfuehrlich(self, thema: str, inhalt: str, quelle: str = "unbekannt"):
        """
        Lerne ausführlich über ein Thema.
        Erstellt neuronales Konzept + inhaltliche Synapsenverbindungen.
        """
        if not thema or not inhalt:
            return "❌ Ungültige Eingabe – bitte Thema und Inhalt angeben."

        # Hauptneuron für Thema anlegen
        self.fm.create_neuron(thema, types=["knowledge", "learning"], sentence=inhalt, source=quelle)
        self.fm.record_episode({
            "type": "learning",
            "topic": thema,
            "content": inhalt,
            "source": quelle,
            "timestamp": now_iso()
        })

        # Schlüsselbegriffe aus dem Inhalt extrahieren
        woerter = [w for w in re.findall(r"\b\w{4,}\b", inhalt.lower()) if len(w) > 3]
        wichtig = Counter(woerter).most_common(10)
        for wort, _ in wichtig:
            self.fm.create_neuron(wort, types=["concept"], source=thema)
            self.sn.add_synapse(thema, wort, weight=0.6)
            self.sn.strengthen(thema, wort, step=0.2)

        # Meta-Verknüpfung mit globalen Lernkonzepten
        for basis in ["lernen", "wissen", "erkenntnis", "konzept", "verstehen"]:
            self.sn.add_synapse(thema, basis, weight=0.7)

        self.sn.persist()
        return f"📘 Ich habe ausführlich über **{thema}** gelernt und mein Wissen verknüpft."



    def lerne_neue_aussage(self, aussage: str):
        """
        Erkennt eine neue Aussage und speichert sie als neuronales Konzept.
        Beispiel: 'Katzen mögen Milch' → Verknüpfung (katzen ↔ milch).
        """
        if not aussage or len(aussage.split()) < 2:
            return "Bitte gib eine sinnvolle Aussage ein."

        # Neuron für gesamte Aussage anlegen
        self.fm.create_neuron(aussage, types=["statement"], sentence=aussage, source="user")

        # Wörter extrahieren & semantisch verbinden
        tokens = [t for t in re.findall(r"\b\w{3,}\b", aussage.lower()) if len(t) > 2]
        for i, w1 in enumerate(tokens):
            self.fm.create_neuron(w1, types=["concept"], source="statement")
            for w2 in tokens[i + 1:]:
                self.fm.create_neuron(w2, types=["concept"], source="statement")
                self.sn.add_synapse(w1, w2, weight=0.5)
                self.sn.strengthen(w1, w2, step=0.1)

        # Episode im Memory loggen
        self.fm.record_episode({
            "type": "statement",
            "content": aussage,
            "timestamp": now_iso()
        })

        self.sn.persist()
        return f"🧩 Aussage gespeichert und semantisch verknüpft: «{aussage}»"






    # ---------------------------
    # TRUTH ENGINE: Basis-Wissen für Korrekturen
    # ---------------------------

    def get_true_fact(self, subject, verb):
        """Gibt das wahre Wissen zurück, wenn bekannt."""
        subject = subject.lower()
        verb = verb.lower()

        # Basiswissen – kann später erweitert werden
        truths = {
            ("katzen", "trinken"): "Wasser",
            ("hunde", "trinken"): "Wasser",
            ("menschen", "trinken"): "Wasser",
            ("pflanzen", "brauchen"): "Wasser",
        }

        return truths.get((subject, verb), None)


    # ---------------------------
    # AUSSAGE-PARSER
    # ---------------------------

    def parse_statement(self, text):
        """
        Erkennt: SUBJEKT – VERB – OBJEKT
        Beispiel: 'Katzen trinken Milch'
        """
        text = text.strip().lower()

        m = re.match(r"^(\w+)\s+(\w+)\s+(.*)$", text)
        if not m:
            return None, None, None

        subject = m.group(1)
        verb = m.group(2)
        obj = m.group(3)

        return subject, verb, obj


    # ---------------------------
    # LERNEN MIT AUTOMATISCHER WAHRHEITSKORREKTUR
    # ---------------------------

    def lerne_logische_aussage(self, aussage):
        subject, verb, obj = self.parse_statement(aussage)

        if not subject or not verb:
            return "Ich konnte die Aussage nicht logisch analysieren."

        # 1. Wahres Wissen prüfen
        true_obj = self.get_true_fact(subject, verb)

        if true_obj:
            # → Korrektur nötig
            gespeichertes = true_obj
            korrekt = True
        else:
            # → User-Aussage übernehmen
            gespeichertes = obj
            korrekt = False

        # 2. Wissen speichern
        frage = f"{subject} {verb}"
        antwort = gespeichertes

        self.speichere_wissen(frage, antwort, überschreiben=True)

        # 3. Semantische Synapsen anlegen
        self.verknuepfe_woerter(subject, verb, weight=0.8)
        self.verknuepfe_woerter(subject, gespeichertes, weight=0.8)
        self.verknuepfe_woerter(verb, gespeichertes, weight=0.6)

        if korrekt:
            return f"Ich habe die Aussage gespeichert. Korrektur: {subject.capitalize()} {verb} {gespeichertes}."
        else:
            return f"Ich habe gelernt: {subject.capitalize()} {verb} {gespeichertes}."



    # ---------------------------
    # FRAGEN-BEANTWORTER (Was trinken Katzen?)
    # ---------------------------

    def beantworte_logische_frage(self, frage):
        frage = frage.strip().lower()

        m = re.match(r"^was\s+(\w+)\s+(\w+)\??$", frage)
        if not m:
            return None

        verb = m.group(1)
        subject = m.group(2)

        key = f"{subject} {verb}"
        antwort = self.finde_wissen(key)

        if antwort:
            return f"{subject.capitalize()} {verb} {antwort}."
        else:
            return None








    def formuliere_gelernte_erkenntnis(self, thema: str):
        """
        Formuliert, was das System über ein Thema gelernt hat.
        Traversiert Synapsen & erzeugt natürlich klingende Zusammenfassungen.
        """
        if not thema:
            return "Bitte gib ein Thema an."

        slug = slugify(thema)
        if not self.fm.has_neuron(slug):
            return f"Ich habe zu **{thema}** noch nichts gelernt."

        nachbarn = self.sn.top_associations(slug, limit=6)
        if not nachbarn:
            return f"Ich kenne keine direkten Verknüpfungen zu **{thema}**."

        saetze = []
        for wort, gewicht, _ in nachbarn:
            satz = random.choice([
                f"**{thema.capitalize()}** steht in Beziehung zu **{wort}**.",
                f"Ich habe gelernt, dass **{thema}** oft mit **{wort}** verbunden ist.",
                f"Zwischen **{thema}** und **{wort}** besteht eine synaptische Stärke von {gewicht:.2f}.",
                f"**{wort.capitalize()}** ist ein wichtiger Begriff im Kontext von **{thema}**."
            ])
            saetze.append(satz)

        self.fm.record_episode({
            "type": "reflection",
            "topic": thema,
            "summary": saetze,
            "timestamp": now_iso()
        })

        return "🤖 Hier ist meine gelernte Erkenntnis:\n" + "\n".join(saetze)






    def reflektiere_letztes_thema(self) -> str:
        """
        Erkennt und reflektiert das zuletzt gelernte Thema aus Memory.
        Traversiert die Synapsen und formuliert Erkenntnisse auf Basis der neuronalen Struktur.
        """
        try:

            print(f"[DEBUG] Letztes Thema im Memory: {self.fm.last_topic()}")


            last_topic = self.fm.last_topic()
            if not last_topic:
                return "🤖 Ich erinnere mich gerade an nichts Konkretes."

            slug = slugify(last_topic)
            if not self.fm.has_neuron(slug):
                return f"Ich habe noch kein neuronales Konzept zu **{last_topic}**."

            nachbarn = self.sn.top_associations(slug, limit=6)
            if not nachbarn:
                return f"Ich erinnere mich an **{last_topic}**, aber ohne weitere Verknüpfungen."

            sätze = []
            for wort, gewicht, _ in nachbarn:
                sätze.append(random.choice([
                    f"**{last_topic.capitalize()}** steht in enger Verbindung mit **{wort}**.",
                    f"Ich erinnere mich, dass **{last_topic}** oft mit **{wort}** assoziiert wird.",
                    f"Zwischen **{last_topic}** und **{wort}** besteht eine Stärke von {gewicht:.2f}.",
                    f"**{wort.capitalize()}** ist ein wichtiger Begriff im Kontext von **{last_topic}**."
                ]))

            # 🧠 Gedächtniseintrag aktualisieren
            self.fm.record_episode({
                "type": "reflection",
                "topic": last_topic,
                "summary": sätze,
                "timestamp": now_iso()
            })

            ausgabe = "🧠 Ich erinnere mich:\n" + "\n".join(sätze)
            print(f"[REFLEXION] {ausgabe}")
            return ausgabe

        except Exception as e:
            return f"❌ Reflexionsfehler: {e}"










    # ---------------------------
    # NEUE Neurologische API: Synapsen-basiert (ersetzt DB-assoziationen)
    # ---------------------------
    def verknuepfe_woerter(self, wort1: str, wort2: str, weight: float = 1.0):
        """
        Verstärkt oder erstellt Assoziation zwischen zwei Wörtern/Begriffen.
        Kompatibel zur alten DB-Tabelle, aber nutzt primär SynapseNetwork (JSON).
        - legt Neuronen an (FileManager)
        - erhöht Synapsengewicht bei Wiederholung (self.sn.add_synapse)
        - schreibt optional in alte DB-Tabelle zur Abwärtskompatibilität
        """
        if not wort1 or not wort2:
            return
        w1 = str(wort1).strip().lower()
        w2 = str(wort2).strip().lower()
        if w1 == w2:
            return

        try:
            # 1) Erzeuge/aktualisiere Neuronen
            self.fm.create_neuron(w1, types=["concept"])
            self.fm.create_neuron(w2, types=["concept"])

            # 2) Synapse: add_synapse sorgt für +weight wenn bereits vorhanden
            #    Wir nutzen add_synapse (bidirectional) damit Gewicht wächst
            self.sn.add_synapse(w1, w2, weight=weight, bidirectional=True)

           # 3) zusätzlich: kontrollierte Verstärkung (kleiner Schritt)
           #    Das sorgt dafür, dass wiederholte kurze Treffer stärker werden
           #    (z. B. Smalltalk-Treffer)
            self.sn.strengthen(w1, w2, step=min(SYNAPSE_STRENGTHEN_STEP, weight))


# ? self.sn.strengthen(keyword, self.last_topic, step=min(0.2, similarity))


        except Exception as e:
            print(f"[verknuepfe_woerter] Synapse Fehler: {e}")

        # 4) Backward-compatibility: alte 'assoziationen' Tabelle inkrementell pflegen
        try:
            c = self.conn.cursor()
            c.execute("""
                SELECT id, gewicht FROM assoziationen
                WHERE wort1 = ? AND wort2 = ? AND api_key = ?
            """, (w1, w2, self.api_key))
            row = c.fetchone()
            if row:
                c.execute("UPDATE assoziationen SET gewicht = gewicht + ? WHERE id = ?", (int(max(1, weight)), row[0]))
            else:
                c.execute("""
                    INSERT INTO assoziationen (wort1, wort2, gewicht, api_key) VALUES (?, ?, ?, ?)
                """, (w1, w2, int(max(1, weight)), self.api_key))
            self.conn.commit()
        except Exception:
            # DB optional — kein kritischer Fehler
            pass









    def erzeuge_synapsen_html(self):
        """
        Führt eine vollständige Analyse des neuronalen Gehirnmodells durch,
        inklusive Struktur-, Gewichtungs- und Topologie-Daten.
        Sendet die JSON-Analyse an den Server zur Speicherung (Visualisierung).
        """

        brain_file = getattr(self.fm, "brain_file", None)
        if not brain_file or not os.path.exists(brain_file):
            return {"status": "error", "message": "❌ Kein Gehirn gefunden (brain.json fehlt oder Pfad ungültig)."}

        with open(brain_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        neurons = data.get("neurons", {})
        synapses = data.get("synapses", {})

        neuron_count = len(neurons)
        synapse_count = sum(len(v) for v in synapses.values())

        # --- Gewichtsauswertung ---
        all_weights = [w for s in synapses.values() for w in s.values()]
        if not all_weights:
            all_weights = [0]
        avg_weight = sum(all_weights) / len(all_weights)
        variance = sum((w - avg_weight) ** 2 for w in all_weights) / len(all_weights)
        weight_stats = {
            "min": round(min(all_weights), 4),
            "max": round(max(all_weights), 4),
            "avg": round(avg_weight, 4),
            "var": round(variance, 6)
        }

        # --- Verbindungen pro Neuron ---
        conn_per_neuron = {slug: len(synapses.get(slug, {})) for slug in neurons.keys()}
        avg_conn = round(sum(conn_per_neuron.values()) / neuron_count, 2) if neuron_count else 0

        # --- Netzwerk-Topologie ---
        total_possible = neuron_count * (neuron_count - 1)
        density = round(synapse_count / total_possible, 6) if total_possible else 0
        degree_distribution = {
            "min_degree": min(conn_per_neuron.values()) if conn_per_neuron else 0,
            "max_degree": max(conn_per_neuron.values()) if conn_per_neuron else 0,
            "avg_degree": avg_conn
        }
        topology = {
            "dichte": density,
            "grad_verteilung": degree_distribution,
            "gesamt_verbindungen": synapse_count,
            "neuronen": neuron_count
        }

        # --- Clusterbildung ---
        clusters = []
        threshold = avg_weight * 1.2
        visited = set()

        for a, targets in synapses.items():
            if a in visited:
                continue
            cluster = set([a])
            for b, w in targets.items():
                if w >= threshold:
                    cluster.add(b)
                    visited.add(b)
            visited.add(a)
            if len(cluster) > 1:
                clusters.append(list(cluster))

        # --- Synapsenliste ---
        synapse_edges = [
            {"source": a, "target": b, "weight": round(w, 4)}
            for a, links in synapses.items()
            for b, w in links.items()
        ]

        # --- Neuronliste ---
        neuron_nodes = [
            {
                "id": slug,
                "label": n.get("word", slug),
                "types": n.get("types", []),
                "confidence": n.get("confidence", 0.5)
            }
            for slug, n in neurons.items()
        ]

        # --- Zusammenfassung ---
        brain_summary = {
            "neuronen": neuron_count,
            "synapsen": synapse_count,
            "durchschnittsgewicht": weight_stats["avg"],
            "aktivitätsvarianz": weight_stats["var"],
            "verbindungen_pro_neuron": avg_conn,
            "netz_dichte": density,
        }

        result = {
            "status": "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "api_key": self.api_key or "unknown",
            "meta": {
                "version": getattr(self, "VERSION", "unknown"),
                "source_file": brain_file
            },
            "brain_summary": brain_summary,
            "weights": weight_stats,
            "topology": topology,
            "connections_per_neuron": conn_per_neuron,
            "clusters": clusters,
            "neurons": neuron_nodes,
            "synapses": synapse_edges,
            "message": (
                f"🧠 Gehirn-Analyse abgeschlossen:\n"
                f"- Neuronen: {neuron_count}\n"
                f"- Synapsen: {synapse_count}\n"
                f"- Dichte: {density:.6f}\n"
                f"- ⌀ Gewicht: {weight_stats['avg']}\n"
                f"- Varianz: {weight_stats['var']}\n"
                f"- Cluster erkannt: {len(clusters)}"
            )
        }

        # --- Upload an HTTPS-Server ---
        try:
            php_url = ""
            response = requests.post(php_url, json=result, timeout=15)
            if response.status_code == 200:

                # --- Kompaktes JSON (für save_brain_data) ---
                short_result = {
                    "status": "ok",
                    "brain_summary": brain_summary,
                    "message": (
                        f"🧠 Gehirnstatus:\n"
                        f"- Neuronen: {neuron_count}\n"
                        f"- Synapsen: {synapse_count}\n"
                        f"- ⌀ Gewicht: {avg_weight:.4f}\n"
                        f"- Aktivitätsvarianz: {variance:.6f}\n"
                        f"- Synapsen/Neuron: {brain_summary['verbindungen_pro_neuron']}"
                    )
                }





                BRAIN_STATUS = f"""<div class="brain-status"><h2>🧠 Gehirnstatus</h2><ul><li>Neuronen: {neuron_count}</li><li>Synapsen: {synapse_count}</li><li>⌀ Gewicht: {avg_weight:.4f}</li><li>Aktivitätsvarianz: {variance:.6f}</li><li>Synapsen/Neuron: {brain_summary['verbindungen_pro_neuron']}</li></ul></div>"""



                # --- Upload an save_brain_data.php ---
                try:
                    short_url = ""
                    response_short = requests.post(short_url, json=short_result, timeout=10)
                    if response_short.status_code == 200:
                        print("✅ Kurzstatus erfolgreich an save_brain_data.php gesendet.")
                    else:
                        print(f"⚠️ Fehler beim Kurz-Upload ({response_short.status_code}): {response_short.text}")
                except Exception as e:
                    print(f"❌ Fehler beim Senden an save_brain_data.php: {e}")

                return {
                    "status": "ok",
                    "message": f"🧠 Neuronales netzwerk erfolgreich visualisiert und an {php_url} gesendet. \n  {BRAIN_STATUS}",
                    "server_response": response.text
                }
            else:
                return {"status": "error", "message": f"Serverfehler {response.status_code}: {response.text}"}
        except Exception as e:
            return {"status": "error", "message": f"❌ Fehler beim Senden der Analyse: {e}"}







 




    def zeige_assoziationen(self, wort: str, limit: int = 10) -> str:
        """Zeigt Top-assoziationen. Nutzt SynapseNetwork wenn möglich."""
        wort = wort.lower().strip()
        if not wort:
            return "Bitte gib ein Wort an."
        # ensure neuron exists (best effort)
        if not self.fm.has_neuron(wort):
            # try to create a lightweight neuron
            try:
                self.fm.create_neuron(wort, types=["noun"])
            except Exception:
                pass
        slug = slugify(wort)
        results = self.sn.top_associations(slug, limit=limit)
        if not results:
            # fallback: DB-assoziationen
            c = self.conn.cursor()
            c.execute("""
                SELECT wort2, gewicht FROM assoziationen
                WHERE wort1 = ? AND api_key = ?
                ORDER BY gewicht DESC LIMIT ?
            """, (wort, self.api_key, limit))
            rows = c.fetchall()
            if not rows:
                return f"Ich kenne keine Begriffe, die mit {wort} verbunden sind."
            lines = [f"🔗 Begriffe, die mit {wort} assoziiert sind (DB):"]
            for i, (w2, gewicht) in enumerate(rows, 1):
                lines.append(f"{i}. **{w2}** (Stärke: {gewicht})")
            return "\n".join(lines)
        lines = [f"🔗 Begriffe, die mit {wort} assoziiert sind:"]
        for i, (w2, gewicht, slug_id) in enumerate(results, 1):
            lines.append(f"{i}. **{w2}** (Stärke: {gewicht:.2f})")
        return "\n".join(lines)

    def assoziierte_satzbildung(self, zentralbegriff: str) -> str:
        zentralbegriff = zentralbegriff.strip().lower()
        if not zentralbegriff:
            return "Bitte gib einen zentralen Begriff an."
        slug = slugify(zentralbegriff)
        results = self.sn.top_associations(slug, limit=5)
        if not results:
            return f"Ich kenne noch keine Begriffe, die mit **{zentralbegriff}** verknüpft sind."
        saetze = []
        for wort2, gewicht, s in results:
            satz = random.choice([
                f"**{zentralbegriff.capitalize()}** steht in enger Verbindung mit **{wort2}**.",
                f"Wenn ich an **{zentralbegriff}** denke, fällt mir auch **{wort2}** ein.",
                f"**{wort2}** ist mit **{zentralbegriff}** verknüpft – das habe ich gelernt.",
                f"Zwischen **{zentralbegriff}** und **{wort2}** besteht ein Zusammenhang."
            ])
            saetze.append(satz)
        return "🤖 Hier ist, was ich über **" + zentralbegriff + "** denke:\n" + "\n".join(saetze)

    # ---------------------------
    # Restliche Funktionen bleiben wie in der ursprünglichen v4.5.5
    # (Smalltalk, Lern-Funktionen, Searches, Verarbeitung)
    # Ich habe sie hier aus Platzgründen nicht nochmal komplett verändert,
    # aber die wichtigsten Entry-Punkte sind identisch (z.B. verarbeite_eingabe)
    # ---------------------------
#    def _format_gpt_style(self, keyword: str, inhalte: list) -> str:
#        header = f"Ich analysiere jetzt, {keyword} genauer..\n"
#        parts = []
#        for text in inhalte:
#            parts.append(f"{text}\n")
#        footer = ""
#        return header + "\n".join(parts) + footer



    def _format_gpt_style(self, keyword: str, inhalte: list) -> str:
        """
        Professionelle, wissenschaftliche und kontextsensitiv formulierte Ausgabe.
        Keine Emojis, kein Smalltalk, klare Struktur und hoher sprachlicher Standard.
        """

        thema = keyword.lower()

        # Kontextabhängige, neutrale Einleitung
        if any(w in thema for w in ["technik", "algorithmus", "system", "automatisierung"]):
            intro = f"Technische Analyse zu **{keyword}**:"
        elif any(w in thema for w in ["politik", "gesellschaft", "kultur"]):
            intro = f"Einordnung und Analyse zu **{keyword}**:"
        elif any(w in thema for w in ["natur", "biologie", "umwelt"]):
            intro = f"Fachliche Übersicht zu **{keyword}**:"
        else:
            intro = f"Zusammenfassung der verfügbaren Informationen zu **{keyword}**:"

        # Inhalt formatieren
        body = ""
        for t in inhalte:
            body += t.strip() + "\n\n"

        conclusion = (
            "Wenn ausführlichere Details, Quellenangaben oder eine vertiefte Analyse gewünscht sind, "
            "kann ich diese jederzeit ergänzen."
        )

        return intro + "\n\n" + body.strip() + "\n\n" + conclusion



    # extract_keyword, 	beantworte_smalltalk, fuege_smalltalk_antwort_hinzu, etc.
    # (kopiert / erhalten aus der ursprünglichen Datei)
    def extract_keyword(self, frage):
        frage = frage.strip().lower()
        match = re.search(r"(?:wird|kann|könnte|steigt|fällt|erreicht)\s+(\w+)", frage)
        if match:
            return match.group(1).strip().title()
        if " über " in frage:
            return frage.split(" über ")[-1].strip().title()
        match = re.search(r"(?:preis|kurs|kostet|kostenpunkt)\s+(?:von\s+)?(\w+)", frage)
        if match:
            return match.group(1).strip().title()
        stopwords = {
            "was", "ist", "sind", "ein", "eine", "der", "die", "das", "den", "dem", "mir", "über",
            "erzähle", "erzähl", "welcher", "welche", "welches", "wie", "wo", "wann", "warum",
            "wieso", "weshalb", "gib", "geben", "können", "kannst", "bitte", "danke",
            "lerne", "lern", "studiere", "informiere", "thema", "also", "nein", "zu",
            "war", "genau", "genauer", "mehr", "besser"
        }
        frage_clean = frage.translate(str.maketrans('', '', string.punctuation))
        tokens = frage_clean.split()
        keywords = [w for w in tokens if w not in stopwords]
        if not keywords:
            return frage.strip().title()
        return " ".join(keywords).title()





    def beantworte_smalltalk(self, eingabe: str) -> str:
        """
        Smalltalk + synaptisches Verhalten:
        - erkennt Namen
        - benutzt self.smalltalk (fm.load_smalltalk)
        - bei Treffer: erzeugt Neuron für den Key + verstärkt Synapsen zu den Token-Subbegriffen
        - beim Lernen: legt Neuron an, verknüpft es mit recent context und verstärkt Synapsen
        - erweitert: sichtbares Lern-Feedback & Memory-Logging
        """
        if not isinstance(eingabe, str) or not eingabe.strip():
            return ""

        eingabe_raw = eingabe.strip()
        eingabe_lc = eingabe_raw.lower()
        key = eingabe_lc.translate(str.maketrans('', '', string.punctuation)).strip()

        # Schutz: definierende Wissensfragen nicht im smalltalk lernen
        if re.match(r"^(was\s+ist|wer\s+ist|wie\s+funktioniert|was\s+bedeutet|was\s+sind)\b", key):
            return ""

        # Name-Erkennung (wie vorher)
        m = re.search(r"\b(?:ich heiße|mein name ist|ich bin|ich bin der|ich bin die)\s+(.{1,40})\b", eingabe_lc)
        if m:
            name = m.group(1).strip().split()[0].title()
            try:
                if self.api_key:
                    self.aktualisiere_remote_username(name)
                self.setze_benutzername_db(name, self.api_key or "")
            except Exception:
                pass
            self.user_name = name
            return f"Freut mich, dich kennenzulernen, {name}! 😊"

        # ensure smalltalk loaded
        if not isinstance(self.smalltalk, dict) or not self.smalltalk:
            try:
                self.smalltalk = self.fm.load_smalltalk() or {}
            except Exception:
                self.smalltalk = {}

        # normalize smalltalk index
        norm_index = {}
        for k, v in list(self.smalltalk.items()):
            nk = k.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            norm_index[nk] = v

        # Exact match → Antwort + neuron & synapse handling
        if key in norm_index:
            antworten = norm_index[key]
            antwort = random.choice(antworten) if isinstance(antworten, list) else antworten

            try:
                # create neuron for the smalltalk key + connect to tokens/context
                self.fm.create_neuron(key, types=["smalltalk"], sentence=eingabe_raw, source="smalltalk")
                tokens = [t for t in key.split() if len(t) > 2]
                for tok in tokens:
                    self.verknuepfe_woerter(key, tok, weight=0.6)
                if getattr(self, "last_topic", None):
                    self.verknuepfe_woerter(key, self.last_topic, weight=0.8)
            except Exception:
                pass

            try:
                return antwort.format(name=self.user_name) if "{name" in antwort else antwort
            except Exception:
                return antwort

        # Substring match (part of key matches)
        for nk, antworten in norm_index.items():
            if nk and nk in key:
                antwort = random.choice(antworten) if isinstance(antworten, list) else antworten
                try:
                    self.fm.create_neuron(nk, types=["smalltalk"], sentence=eingabe_raw, source="smalltalk")
                    tokens = [t for t in nk.split() if len(t) > 2]
                    for tok in tokens:
                        self.verknuepfe_woerter(nk, tok, weight=0.45)
                    if getattr(self, "last_topic", None):
                        self.verknuepfe_woerter(nk, self.last_topic, weight=0.7)
                except Exception:
                    pass
                try:
                    return antwort.format(name=self.user_name) if "{name" in antwort else antwort
                except Exception:
                    return antwort

        # Wenn nichts gefunden -> nur bei Aussagen (keine Frage) lernen & synaptisch speichern
        token_count = len(key.split())
        if "?" not in eingabe_raw and token_count <= 12:
            synaptische_antworten = [
                "Interessant, das merke ich mir. 🧠",
                "Spannend — ich verknüpfe das im Hirn.",
                "Das speichere ich ab und verknüpfe es im Netz.",
                "Cool, das notiere ich mir."
            ]

            # 🧠 Sichtbares Lern-Feedback + Kontextaktualisierung
            print(f"[LERNDEBUG] Neue Aussage erkannt: {eingabe_raw}")

            # 1️⃣ Thema im Memory speichern
            try:
                self.fm.record_episode({
                    "type": "learning",
                    "topic": key,
                    "content": eingabe_raw,
                    "timestamp": now_iso()
                })
                self.last_topic = key  # Kontext aktualisieren
            except Exception as e:
                print(f"[LERNDEBUG] Memory-Fehler: {e}")

            # 2️⃣ Neue smalltalk entry speichern
            self.smalltalk[key] = synaptische_antworten
            try:
                self.fm.save_smalltalk(self.smalltalk)
            except Exception:
                try:
                    with open("smalltalk.json", "w", encoding="utf-8") as f:
                        json.dump(self.smalltalk, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass

            # 3️⃣ Neuron + Verknüpfung zur aktuellen Konversation (context)
            try:
                self.fm.create_neuron(key, types=["smalltalk"], sentence=eingabe_raw, source="user")
                tokens = [t for t in key.split() if len(t) > 2]
                for tok in tokens:
                    self.verknuepfe_woerter(key, tok, weight=0.5)
                if getattr(self, "last_topic", None):
                    self.verknuepfe_woerter(key, self.last_topic, weight=0.7)
            except Exception as e:
                print(f"[LERNDEBUG] Neuron-Verknüpfung-Fehler: {e}")

            # 4️⃣ Nach Synapsenbildung: Verknüpfungen anzeigen
            try:
                verkn = self.sn.top_associations(slugify(key), limit=4)
                if verkn:
                    print(f"[LERNDEBUG] Synapsen für '{key}':")
                    for wort, gewicht, slug in verkn:
                        print(f"   ↳ {wort} ({gewicht:.2f})")
                else:
                    print(f"[LERNDEBUG] Keine direkten Synapsen zu '{key}' gefunden.")
            except Exception as e:
                print(f"[LERNDEBUG] Synapsenanzeige-Fehler: {e}")

            # Optional: sichtbares Lern-Feedback an den Benutzer
            try:
                if verkn:
                    gelernte = ", ".join([v[0] for v in verkn])
                    return f"🧠 Ich habe gelernt, dass **{key}** mit {gelernte} verbunden ist."
            except Exception:
                pass

            # Fallback: normale Antwort
            return random.choice(synaptische_antworten)

        # sonst nichts
        return ""

















    def fuege_smalltalk_antwort_hinzu(self, frage: str, antwort: str):
        frage_clean = frage.lower().translate(str.maketrans('', '', string.punctuation)).strip()
        arr = self.smalltalk.get(frage_clean, [])
        if not isinstance(arr, list):
            arr = [arr]
        if antwort not in arr:
            arr.append(antwort)
        self.smalltalk[frage_clean] = arr
        try:
            with open("smalltalk.json", "w", encoding="utf-8") as f:
                json.dump(self.smalltalk, f, ensure_ascii=False, indent=2)
            # Update fm smalltalk as well
            self.fm.save_smalltalk(self.smalltalk)
            return f"✅ Smalltalk-Antwort hinzugefügt: {frage} -> {antwort}"
        except Exception as e:
            return f"❌ Fehler beim Speichern smalltalk: {e}"







    def loesche_smalltalk(self, text: str) -> str:
        """
        Löscht Smalltalk-Fragen oder einzelne Antworten mit Bestätigung.
        Beispiel:
          !smalltalk löschen: Hallo
          !smalltalk löschen: Hallo = Hi!
          ja löschen hallo
          nein abbrechen
        """

        # 🟢 Schritt 1: Wenn Benutzer gerade bestätigt
        if getattr(self, "_pending_delete", None):
            if text.lower().startswith("ja löschen"):
                frage = self._pending_delete.get("frage")
                antwort = self._pending_delete.get("antwort")
                self._pending_delete = None  # reset

                if antwort:
                    # Nur diese Antwort löschen
                    eintrag = self.smalltalk.get(frage, [])
                    if isinstance(eintrag, list) and antwort in eintrag:
                        eintrag.remove(antwort)
                        if not eintrag:
                            del self.smalltalk[frage]
                            msg = f"🗑️ Antwort gelöscht, und da keine Antworten mehr übrig waren, wurde «{frage}» entfernt."
                        else:
                            self.smalltalk[frage] = eintrag
                            msg = f"🗑️ Antwort «{antwort}» aus «{frage}» gelöscht."
                    else:
                        return f"❌ Antwort «{antwort}» in «{frage}» nicht gefunden."
                else:
                    # Ganze Frage löschen
                    if frage in self.smalltalk:
                        del self.smalltalk[frage]
                        msg = f"🗑️ Kompletter Smalltalk-Eintrag «{frage}» wurde gelöscht."
                    else:
                        return f"❌ Frage «{frage}» nicht gefunden."

                # Speichern
                try:
                    with open("smalltalk.json", "w", encoding="utf-8") as f:
                        json.dump(self.smalltalk, f, ensure_ascii=False, indent=2)
                    self.fm.save_smalltalk(self.smalltalk)
                    return msg
                except Exception as e:
                    return f"❌ Fehler beim Speichern: {e}"

            elif text.lower().startswith("nein abbrechen"):
                self._pending_delete = None
                return "🚫 Löschvorgang abgebrochen."

        # 🟡 Schritt 2: Normaler Löschbefehl (Vorbereitung)
        if text.lower().startswith("!smalltalk löschen:"):
            match = re.match(r"!smalltalk löschen:\s*(.*?)(?:\s*=\s*(.+))?$", text, re.IGNORECASE)
            if not match:
                return "❌ Format:\n`!smalltalk löschen: Frage` oder `!smalltalk löschen: Frage = Antwort`"

            frage = match.group(1).strip().lower().translate(str.maketrans('', '', string.punctuation)).strip()
            antwort = match.group(2).strip() if match.group(2) else None

            if frage not in self.smalltalk:
                return f"❌ Frage «{frage}» nicht gefunden."

            # Warnung anzeigen + Pending speichern
            self._pending_delete = {"frage": frage, "antwort": antwort}
            if antwort:
                return (f"⚠️ Willst du wirklich die Antwort «{antwort}» aus «{frage}» löschen?\n"
                        f"Schreibe `ja löschen {frage}` oder `nein abbrechen`.")
            else:
                return (f"⚠️ Willst du wirklich den gesamten Smalltalk-Eintrag «{frage}» löschen?\n"
                        f"Schreibe `ja löschen {frage}` oder `nein abbrechen`.")

        # Wenn kein passender Kontext
        return ""









    # ---------------------------
    # Pending-Refresh / Pending-Save Handlers
    # ---------------------------
    def handle_pending_refresh(self, text: str) -> str:
        """
        Erweiterte Bestätigungslogik für 'neuere Infos?' (analog zu !reset brain -> !confirm reset).
        - 'ja' / 'klar' ... -> holt neue Infos (multi_source_suche) und fragt anschließend, ob gespeichert werden soll.
        - '!confirm refresh' -> holt neue Infos UND speichert sofort (überschreiben=True).
        - 'nein' / '!cancel refresh' -> bricht ab.
        Liefert einen leeren String zurück, wenn die Eingabe nicht zu diesem Pending passt.
        """
        if not getattr(self, "_pending_refresh", None):
            return ""

        lower = text.lower().strip()
        keyword = self._pending_refresh.get("keyword")

        # direkte Bestätigung: hol & SPEICHERE sofort
        if lower in ["!confirm refresh", "confirm refresh", "confirm refresh please", "ja bitte speichern sofort"]:
            try:
                result = self.multi_source_suche(keyword)
            except Exception as e:
                self._pending_refresh = None
                return f"❌ Fehler beim Abrufen der Infos: {e}"

            # sofort speichern (überschreiben)
            try:
                self.speichere_wissen(keyword, result, überschreiben=True)
            except Exception as e:
                self._pending_refresh = None
                return f"🔍 Update zu **{keyword}** geladen, aber Fehler beim Speichern: {e}"

            self._pending_refresh = None
            return f"🔍 Aktualisierte Infos zu **{keyword}** geladen und **sofort gespeichert**.\n\n{result}"

        # normale Bestätigung: hol & frage dann, ob speichern
        if lower in ["ja", "ja bitte", "gerne", "klar", "mach das", "unbedingt", "bitte", "natürlich"]:
            try:
                result = self.multi_source_suche(keyword)
            except Exception as e:
                self._pending_refresh = None
                return f"❌ Fehler beim Abrufen der Infos: {e}"

            # Setze Pending-Save (User kann jetzt !confirm save / !cancel save eingeben)
            self._pending_refresh = None
            self._pending_save = {"topic": keyword, "content": result}
            # Loggen in Memory (sichtbar intern)
            try:
                self.fm.record_episode({
                    "type": "refresh",
                    "topic": keyword,
                    "content": result,
                    "timestamp": now_iso()
                })
            except Exception:
                pass

            return (
                f"🔍 Hier sind aktualisierte Infos zu **{keyword}**:\n\n{result}\n\n"
                "💾 Soll ich das ins Gedächtnis speichern?\n"
                "👉 Bestätige mit `!confirm save` oder breche ab mit `!cancel save`.\n"
                "oder: `!confirm refresh` zum direkten Laden+Speichern."
            )

        # Abbrechen
        if lower in ["nein", "nicht nötig", "später", "abbrechen", "!cancel refresh", "cancel refresh"]:
            self._pending_refresh = None
            return "✅ Kein Update durchgeführt — bestehendes Wissen bleibt erhalten."

        return ""

    def handle_pending_save(self, text: str) -> str:
        """
        Verarbeitet !confirm save / !cancel save für Inhalte, die durch handle_pending_refresh
        abgeholt und in self._pending_save abgespeichert wurden.
        """
        if not getattr(self, "_pending_save", None):
            return ""

        lower = text.lower().strip()
        topic = self._pending_save.get("topic")
        content = self._pending_save.get("content")

        if lower in ["!confirm save", "confirm save", "ja speichern", "speichern"]:
            try:
                # speichere und verknüpfe
                self.speichere_wissen(topic, content, überschreiben=True)
                # episodisch loggen
                try:
                    self.fm.record_episode({
                        "type": "save",
                        "topic": topic,
                        "content": content,
                        "timestamp": now_iso()
                    })
                except Exception:
                    pass
                self._pending_save = None
                return f"💾 **{topic}** wurde gespeichert und ins neuronale Gehirn verknüpft."
            except Exception as e:
                self._pending_save = None
                return f"❌ Fehler beim Speichern: {e}"

        if lower in ["!cancel save", "cancel save", "nein speichern", "nicht speichern"]:
            self._pending_save = None
            return "🚫 Speichern abgebrochen. Die aktualisierten Infos wurden nicht gespeichert."

        return ""






    def handle_pending_question(self, text: str) -> str:
        """
        Verarbeitet Follow-Up Fragen wie:
        'Willst du mehr über XYZ wissen?'
        nutzt self._pending_question = {"type": "mehr_info", "topic": XYZ}
        """

        if not getattr(self, "_pending_question", None):
            return ""

        lower = text.lower().strip()
        topic = self._pending_question.get("topic")

        # JA
        if lower in ["ja", "yes", "klar", "gerne", "mach weiter", "okay", "bitte"]:
            self._pending_question = None
            neue_infos = self.multi_source_suche(topic, force_refresh=True)
            return f"🔍 Hier sind mehr Infos über **{topic}**:\n\n{neue_infos}"

        # NEIN
        if lower in ["nein", "nö", "nee", "nicht", "stop", "abbrechen"]:
            self._pending_question = None
            return "Alles klar 😊 Wenn du später mehr Infos willst, sag einfach Bescheid!"

        # Unklar -> Nachfrage
        return "Meintest du *ja* oder *nein*?"



# --- PATCH FOR xenRon ---
# Insert the following into your XenronBrain class

# 1) Add this helper anywhere inside the class XenronBrain (recommended: above verarbeite_eingabe)

    def indent_code_block(self, full_text: str, spaces: int) -> str:
        """
        Intelligente, robuste Code-Indent-Funktion (Variante C+):
        - erkennt automatisch den Code nach der ersten "Befehl"-Zeile
        - normalisiert Tabs → Spaces
        - erhält bestehende Einrückung korrekt
        - überspringt leere oder rein textliche Zeilen
        - funktioniert auch mit gemischten Einrückungen
        - und mit mehrfachen Codeblöcken
        """
        if not isinstance(full_text, str) or not full_text.strip():
            return "❌ Keine gültige Eingabe."

        lines = full_text.split("\n")

        # Wenn nur eine Zeile → kein Code
        if len(lines) <= 1:
            return "❌ Kein Code gefunden."

        # Erste Zeile = Befehl, Rest = Code
        code_lines = lines[1:]

        # Tabs zu Spaces normalisieren
        normalized = [ln.replace("\t", "    ") for ln in code_lines]

        # Entferne leere Anfangszeilen, aber nicht in der Mitte
        while normalized and not normalized[0].strip():
            normalized.pop(0)

        if not normalized:
            return "❌ Kein Code zum Einrücken erkannt."

        # Einrückung erzeugen
        indent = " " * spaces

        # Intelligentes Einrücken (mit Erhalt vorhandener Struktur)
        result = []
        for ln in normalized:
            if not ln.strip():
                result.append(ln)  # Leerzeile bleibt leer
            else:
                result.append(indent + ln)

        return "\n".join(result)








    # ---------------------------
    # Wrapper: Einheitliche Frage-/Aussage-Handling API
    # ---------------------------

    def handle_definition_question(self, text: str):
        """
        Beantwortet intelligente Definitionsfragen:
        - Was ist …?
        - Was sind …?
        - Was bedeutet …?
        - Wer ist …? (für Personen)
        - Funktioniert auch bei Plurals & Komposita
        """
        txt = text.strip().lower()

        # Breitere Matcher
        m = re.match(
            r"^(was|wer)\s+(ist|sind|bedeutet)\s+(der|die|das|ein|eine)?\s*(.+)\??$",
            txt
        )
        if not m:
            return None

        topic = m.group(4).strip()
        if not topic:
            return None

        # Normalize für interne Speicherung
        store_key = topic.lower()

        # Wikipedia-Suche
        wiki_text = self.wikipedia_suche(topic.title())
        if not wiki_text or wiki_text.startswith("xIch konnte"):
            return None

        # Hauptdefinition speichern
        self.speichere_wissen(
            frage=f"definition {store_key}",
            antwort=wiki_text,
            überschreiben=True
        )

        # Strukturierte Fakten aus Wikipedia extrahieren
        try:
            self.extract_and_store_facts(store_key, wiki_text)
        except Exception as e:
            print(f"[extract_and_store_facts] Fehler: {e}")

        return wiki_text












    def handle_question(self, text: str):
        """
        Frage-Dispatcher — Modus 2 (Wikipedia zuerst).
        Verhalten:
          1) Definitions- & Detailfragen → Wikipedia (wenn vorhanden) / strukturierte Extraktion
          2) Logische Fakten-Engine (beantworte_logische_frage)
          3) Lokale Fakten (self.finde_wissen)
          4) Fuzzy-Fallback (self.finde_wissen(..., fuzzy=True))
          5) None -> lässt das System weiter entscheiden (z.B. statement learning NOT triggered here)
        Robust gegen fehlende Regex-Gruppen und Network/Parser-Fehler.
        """
        import logging
        logger = logging.getLogger(__name__)

        txt = (text or "").strip()
        if not txt:
            return None

        txt_lc = txt.lower()

        # -------------------------
        # Helper: Sicheres Wikipedia-Query + Helpers
        # -------------------------
        def wiki_lookup(topic_raw):
            try:
                # title-case is often better for Wikipedia articles ("Mount Everest")
                topic_key = topic_raw.strip()
                if not topic_key:
                    return None
                wiki = None
                try:
                    wiki = self.wikipedia_suche(topic_key.title())
                except Exception as e:
                    # fallback: try raw
                    logger.debug("wikipedia_suche failed for %r: %s", topic_key, e)
                    try:
                        wiki = self.wikipedia_suche(topic_key)
                    except Exception as e2:
                        logger.debug("wikipedia_suche second attempt failed: %s", e2)
                        wiki = None
                if wiki and isinstance(wiki, str) and not wiki.lower().startswith("xich konnte"):
                    return wiki
            except Exception as e:
                logger.exception("wiki_lookup error: %s", e)
            return None

        def first_sentences(text_block, n=2):
            # return first n sentences (very simple splitter)
            if not text_block:
                return None
            import re
            s = re.split(r'(?<=[.!?])\s+', text_block.strip())
            return " ".join(s[:n]).strip()

        def extract_height(text_block):
            # Suche nach Höhenangaben wie "8.848 m", "8848 m", "8,848 m", "8848 m"
            if not text_block:
                return None
            import re
            # allow NBSP etc.
            t = text_block.replace('\u00A0', ' ')
            m = re.search(r'(\d{3,4}(?:[.,]\d+)?\s*(?:m|m\.|meter|metern)\b)', t, flags=re.I)
            if m:
                return m.group(1)
            # alternative: "with a height of 8,848 m" english
            m2 = re.search(r'height.*?(\d{3,4}(?:[.,]\d+)?\s*m)\b', t, flags=re.I)
            if m2:
                return m2.group(1)
            return None

        def extract_location(text_block):
            # very basic heuristic: try to find "in the X" or "im X" or "in X"
            if not text_block:
                return None
            import re
            t = text_block.replace('\n', ' ')
            m = re.search(r'(?:in|im|auf|an|located in|situated in)\s+([A-ZÄÖÜ][\wäöüÄÖÜ\s\-\–\,]+?)(?:[.,;]|\s|$)', t)
            if m:
                return m.group(1).strip()
            # fallback: first sentence
            return first_sentences(text_block, 1)

        # -------------------------
        # 1) Definitions (already robust in your code) — prefer wiki-based definition first
        # -------------------------
        try:
            def_ans = self.handle_definition_question(text)
            if def_ans:
                return def_ans
        except Exception as e:
            logger.exception("handle_definition_question failed: %s", e)

        # -------------------------
        # 2) Detailfragen (height / where / age / when) — Wiki-first
        # -------------------------
        import re

        # --- 2a: Wie hoch ...?
        m = re.match(r"^wie\s+hoch\s+(?:ist|sind)\s+(?:der|die|das)?\s*(.+?)(?:\?)?$", txt_lc, flags=re.I)
        if m:
            topic = m.group(1).strip()
            # wiki lookup first
            wiki_text = wiki_lookup(topic)
            if wiki_text:
                height = extract_height(wiki_text)
                if height:
                    return f"Die Höhe von {topic.title()} beträgt {height}."
                # fallback: return first two sentences as context
                return first_sentences(wiki_text, 2)
            # if no wiki, check local KB
            res = self.finde_wissen(f"höhe {topic}", fuzzy=True)
            if res:
                return res
            return f"Ich habe keine Höhenangabe zu {topic} gespeichert."

        # --- 2b: Wo liegt ...?
        m = re.match(r"^wo\s+liegt\s+(?:der|die|das)?\s*(.+?)(?:\?)?$", txt_lc, flags=re.I)
        if m:
            topic = m.group(1).strip()
            wiki_text = wiki_lookup(topic)
            if wiki_text:
                loc = extract_location(wiki_text)
                if loc:
                    return f"{topic.title()} liegt {loc}."
                return first_sentences(wiki_text, 2)
            res = self.finde_wissen(f"wo liegt {topic}", fuzzy=True)
            if res:
                return res
            return f"Ich habe noch keinen Standort zu {topic} gelernt."

        # --- 2c: Wie alt ...?
        m = re.match(r"^wie\s+alt\s+(?:ist|sind)\s+(?:der|die|das)?\s*(.+?)(?:\?)?$", txt_lc, flags=re.I)
        if m:
            topic = m.group(1).strip()
            wiki_text = wiki_lookup(topic)
            if wiki_text:
                # try to extract year or age string like "since 1856" or "is X years old"
                y = None
                try:
                    y = re.search(r'\b(1[0-9]{3}|20[0-9]{2})\b', wiki_text)
                    if y:
                        return first_sentences(wiki_text, 2) + f"\n(Datum gefunden: {y.group(0)})"
                except Exception:
                    pass
                return first_sentences(wiki_text, 2)
            res = self.finde_wissen(f"alter {topic}", fuzzy=True)
            if res:
                return res
            return None

        # --- 2d: Wann wurde ... / geschah ...?
        m = re.match(r"^wann\s+(?:wurde|geschah|entstand)\s+(?:der|die|das)?\s*(.+?)(?:\?)?$", txt_lc, flags=re.I)
        if m:
            topic = m.group(1).strip()
            wiki_text = wiki_lookup(topic)
            if wiki_text:
                return first_sentences(wiki_text, 2)
            res = self.finde_wissen(f"datum {topic}", fuzzy=True)
            if res:
                return res
            return None

        # -------------------------
        # 3) Simple logical facts (after wiki)
        # -------------------------
        if hasattr(self, "beantworte_logische_frage"):
            try:
                ans2 = self.beantworte_logische_frage(text)
                if ans2:
                    return ans2
            except Exception as e:
                logger.exception("beantworte_logische_frage failed: %s", e)

        # -------------------------
        # 4) Local KB lookup (exact then fuzzy)
        # -------------------------
        try:
            res_local = self.finde_wissen(txt, fuzzy=False)
            if res_local:
                return res_local
            res_fuzzy = self.finde_wissen(txt, fuzzy=True)
            if res_fuzzy:
                return res_fuzzy
        except Exception as e:
            logger.exception("Local knowledge lookup failed: %s", e)

        # -------------------------
        # 5) Nothing found -> None (caller can decide to ask followup / store)
        # -------------------------
        return None












    def handle_statement(self, text: str):
        """
        Intelligente Verarbeitung von Aussagen:
        - erkennt logisch aufgebaute Sätze (Subjekt Verb Objekt)
        - nutzt die Truth-Engine (falls vorhanden)
        - semantisches Lernen als Fallback
        """
        txt = text.strip()
        if not txt or "?" in txt:
            return None

        # Versuch: logische Struktur erkennen
        try:
            subject, verb, obj = self.parse_statement(txt)

            if subject and verb and obj:
                if hasattr(self, "lerne_logische_aussage"):
                    return self.lerne_logische_aussage(txt)
        except Exception:
            pass

        # Fallback: generisches semantisches Lernen
        if hasattr(self, "lerne_neue_aussage"):
            return self.lerne_neue_aussage(txt)

        return None


















    def verarbeite_eingabe(self, text: str) -> str:
        """
        Verarbeite eine Benutzereingabe und liefere eine passende Antwort.
        Verbesserte, modulare Version mit klarer Prioritätslogik und
        kommentierten "Was passiert wenn / Wohin" Hinweisen.

        Priorität (oberste zuerst):
          1. Leere Eingabe -> kurze Aufforderung
          2. Spezialbefehle zur Smalltalk-Verwaltung (!smalltalk hinzufügen / löschen)
          3. Pending-Work (save / refresh / follow-up question)
          4. Systembefehle (Brain-Reset + Confirm/Cancel)
          5. Diverse Spezialbefehle (Indent, Remove chars, Version, Commands, Visuals, Status, Reflexion)
          6. Pending-Question Follow-Up (Ja/Nein-Antwort)
          7. Krypto-Preisabfragen (priorisiert vor Smalltalk)
          8. Smalltalk-Matching
          9. Lernbefehle (lerne:, aussage:, erkenntnis:)
         10. Kontextuelles "Erzähl mir mehr"
         11. Autonomes Lernen (sich merkende Aussagen)
         12. Wissensfragen (multi_source_suche)
         13. Sonstige Schlüsselwörter -> multi_source_suche
         14. Spontane Reflexion (mit niedriger Wahrscheinlichkeit)
         15. Menschlicher Fallback

        "Was passiert wenn / Wohin" Beispiele:
          - Bei "!reset brain": es wird eine pending-Flag-Datei erstellt (-> `pending_reset.flag`),
            das Gehirn wird nach `!confirm reset` gesichert unter `./backup/brain_X` und
            danach gelöscht und neu angelegt (Wohin: BRAIN_DIR, NEURONS_DIR, SYNAPSES_DIR).
          - Bei Krypto-Erkennung: Preis wird über `krypto_preis_kompakt()` geholt und
            das neuronale Netz lernt das Coin-Konzept (Wohin: self.fm.create_neuron()).
        """

        logger = logging.getLogger(__name__)

        # ---- Normalisierung + schnelle Leereingabe-Abfrage ----
        if text is None:
            text = ""
        text = text.strip()
        if not text:
            # Was passiert wenn: Nutzer hat nichts eingegeben -> leichte Aufforderung zurück
            return random.choice([
                "Sag mir kurz was 😊",
                "Ich höre nichts – alles gut bei dir?",
                "Bin bereit, wenn du es bist 😄"
            ])



        # ⬇️⬇️⬇️ HIER HIN EINFÜGEN ⬇️⬇️⬇️
        # -----------------------------------------
        # FRAGE-ENGINE (sehr hohe Priorität!)
        # -----------------------------------------
        try:
            answer = self.handle_question(text)
            if answer is not None:
                return answer
        except Exception as e:
            logger.exception("Fehler in handle_question: %s", e)
        # ⬆️⬆️⬆️ BIS HIER ⬆️⬆️⬆️



        # ----- nützliche lokale Helfer / Normalisierungen -----
        text_lc = text.lower()
        def startswith_any(s, *prefixes):
            return any(s.startswith(p) for p in prefixes)

        # ---- Root / Brain Pfade (immer aktuell berechnen) ----
        ROOT = os.path.dirname(os.path.abspath(__file__))
        BRAIN_DIR = os.path.join(ROOT, "brain")           # Wohin: zentrale Brain-Daten
        NEURONS_DIR = os.path.join(BRAIN_DIR, "neurons")  # Wohin: Neuronen-Storage
        SYNAPSES_DIR = os.path.join(BRAIN_DIR, "synapses")# Wohin: Synapsen-Storage

        # -------------------------
        # 1) Smalltalk verwalten (Add / Delete)
        # -------------------------
        # Format: "!smalltalk hinzufügen: Frage = Antwort"
        if text_lc.startswith("!smalltalk hinzufügen:"):
            match = re.match(r"!smalltalk hinzufügen:\s*(.*?)\s*=\s*(.+)", text, re.IGNORECASE)
            if match:
                frage, antwort = match.group(1).strip(), match.group(2).strip()
                # Was passiert wenn: neuer Smalltalk -> wird in persistentem Speicher gespeichert
                return self.fuege_smalltalk_antwort_hinzu(frage, antwort)
            return "❌ Falsches Format.\nBeispiel:\n`!smalltalk hinzufügen: Frage = Antwort`"

        # Lösch-Varianten: "!smalltalk löschen:" oder kurze Phrasen (kompatibel)
        if text_lc.startswith(("!smalltalk löschen:", "ja löschen", "nein abbrechen")):
            # Was passiert wenn: Entfernen aus Smalltalk-Datenbank (Wohin: persistent store via self.loesche_smalltalk)
            return self.loesche_smalltalk(text)

        # -------------------------
        # 2) Pending-Work Priorität (Save / Refresh / Pending Question)
        # - Wenn ein Pending-Handler aktiv ist, hat er Vorrang.
        # -------------------------
        try:
            if getattr(self, "_pending_save", None):
                handled = self.handle_pending_save(text)
                if handled:
                    # Wohin: handled enthält finale Antwort oder Next-Step
                    return handled
        except Exception as e:
            logger.exception("Fehler in handle_pending_save: %s", e)

        try:
            if getattr(self, "_pending_refresh", None):
                handled = self.handle_pending_refresh(text)
                if handled:
                    return handled
        except Exception as e:
            logger.exception("Fehler in handle_pending_refresh: %s", e)

        try:
            if getattr(self, "_pending_question", None):
                handled = self.handle_pending_question(text)
                if handled:
                    return handled
        except Exception as e:
            logger.exception("Fehler in handle_pending_question: %s", e)

        # -------------------------
        # 3) Systembefehl: Brain Reset (sichere, bestätigungsbasierte Aktion)
        # -------------------------
        PENDING_FILE = os.path.join(ROOT, "pending_reset.flag")

        if text_lc.strip() in ["!reset brain", "!reset gehirn"]:
            # Was passiert wenn: create pending flag -> Benutzer muss mit !confirm reset bestätigen
            try:
                with open(PENDING_FILE, "w", encoding="utf-8") as f:
                    f.write("pending")
                return (
                    "⚠️ Willst du wirklich das gesamte Gehirn zurücksetzen?\n"
                    "Ich lege vorher automatisch ein Backup an unter `./backup/brain_X` (max. 10).\n\n"
                    "👉 Bestätige mit `!confirm reset` oder breche ab mit `!cancel reset`."
                )
            except Exception as e:
                logger.exception("Kann pending flag nicht schreiben: %s", e)
                return f"❌ Fehler beim Initialisieren des Resets: {e}"

        if text_lc.strip() == "!confirm reset" and os.path.exists(PENDING_FILE):
            # Was passiert wenn: Backup wird erzeugt, altes Brain gelöscht, neues Brain initialisiert.
            try:
                backup_root = os.path.join(ROOT, "backup")
                os.makedirs(backup_root, exist_ok=True)

                # Backup limitieren (max. 10) — wohin: backup_root/brain_N
                backups = sorted([d for d in os.listdir(backup_root) if d.startswith("brain_")])
                next_id = len(backups) + 1
                if next_id > 10 and backups:
                    oldest = os.path.join(backup_root, backups[0])
                    shutil.rmtree(oldest, ignore_errors=True)
                    backups = sorted([d for d in os.listdir(backup_root) if d.startswith("brain_")])
                    next_id = len(backups) + 1

                backup_dir = os.path.join(backup_root, f"brain_{next_id}")

                if os.path.exists(BRAIN_DIR):
                    shutil.copytree(BRAIN_DIR, backup_dir)
                    shutil.rmtree(BRAIN_DIR, ignore_errors=True)
                else:
                    os.makedirs(backup_dir, exist_ok=True)

                # Neu anlegen (Wohin: BRAIN_DIR, NEURONS_DIR, SYNAPSES_DIR)
                os.makedirs(BRAIN_DIR, exist_ok=True)
                os.makedirs(NEURONS_DIR, exist_ok=True)
                os.makedirs(SYNAPSES_DIR, exist_ok=True)

                # Integration mit vorhandener API (deine load/init hooks)
                try:
                    self.fm._init_brain()
                    self.fm._load()
                    # SynapseNetwork neu instanziieren (Achte: SynapseNetwork muss importierbar sein)
                    self.sn = SynapseNetwork(self.fm)
                except Exception as inner_e:
                    logger.exception("Fehler beim Reinitialisieren des Brain-Objekts: %s", inner_e)

                # Cleanup pending flag
                os.remove(PENDING_FILE)
                return f"🧠 Gehirn wurde erfolgreich zurückgesetzt!\n➡️ Backup gespeichert unter: `{backup_dir}`"
            except Exception as e:
                logger.exception("Fehler beim Brain-Reset: %s", e)
                # versuche pending flag zu entfernen, wenn existiert
                try:
                    if os.path.exists(PENDING_FILE):
                        os.remove(PENDING_FILE)
                except Exception:
                    pass
                return f"❌ Fehler beim Brain-Reset: {type(e).__name__}: {e}"

        if text_lc.strip() == "!cancel reset" and os.path.exists(PENDING_FILE):
            try:
                os.remove(PENDING_FILE)
                return "🚫 Gehirn-Reset abgebrochen."
            except Exception as e:
                logger.exception("Fehler beim Entfernen des pending flags: %s", e)
                return f"❌ Konnte Reset-Flag nicht entfernen: {e}"

        # -------------------------
        # 4) Spezialbefehle: Code einrücken / Zeichen entfernen
        # -------------------------
        match_indent = re.search(r"rück[e]*.*?(\d+)\s*(?:leerzeichen|spaces?)", text_lc)
        if match_indent:
            spaces = int(match_indent.group(1))
            # Was passiert wenn: identierter Code wird zurückgegeben -> Nutzer kann ihn einfügen
            try:
                return self.indent_code_block(text, spaces)
            except Exception as e:
                logger.exception("Fehler indent_code_block: %s", e)
                return f"❌ Fehler beim Einrücken: {e}"

        match_remove_chars = re.search(
            r"""
            entfern[e]*                      # entferne / entfernen / entfern
            (?:\s+mir)?                      # optional
            (?:\s+(?:folgende[n]?|diese[n]?|die))?   # optional
            \s+zeich[dn]en?                  # zeichen
            (?:\s+aus\s+(?:dem|der|den)\s+text)?     # optional
            \s*[:\-]?\s*                     # optional
            (.+?)                            # die aufzulistenden Zeichen
            (?:$|\n)                         # bis Zeilenende
            """,
            text_lc,
            re.VERBOSE
        )
        if match_remove_chars:
            chars_to_remove = match_remove_chars.group(1).strip()
            try:
                return self.remove_characters_from_text(text, chars_to_remove)
            except Exception as e:
                logger.exception("Fehler remove_characters_from_text: %s", e)
                return f"❌ Fehler beim Entfernen von Zeichen: {e}"

        # -------------------------
        # 5) Version / Commands HTML
        # -------------------------
        if text_lc.strip() in ["zeige version", "aktuelle version"]:
            try:
                return f'🧠  <div> aktuelle version </div> \n {VERSION}'
            except Exception as e:
                logger.exception("Fehler bei zeige version: %s", e)
                return f"⚠️ Fehler beim Befehl zeige version:\n{e}"

        COMMANDS_HTML = """ <div class="cmd-container"><h2 class="cmd-title">🧠 Verfügbare Befehle</h2><ul class="cmd-list"><li><span class="cmd">zeige befehle</span> — Zeigt diese Liste an ✅</li><li><span class="cmd">iframe</span> — Test-Frame anzeigen</li><li><span class="cmd">zeige gehirn</span> — Gehirn-Visualisierung anzeigen</li><li><span class="cmd">gehirn zustand</span> — Status & Synapsen-Infos</li><li><span class="cmd">!smalltalk hinzufügen: Frage = Antwort</span> — neuen Smalltalk speichern</li><li><span class="cmd">!smalltalk löschen: Frage</span> — Smalltalk entfernen (mit Bestätigung)</li><li><span class="cmd">!reset brain</span> — Gehirn vollständig zurücksetzen ⚠️</li><li><span class="cmd">!confirm reset</span> — Reset bestätigen ✅</li><li><span class="cmd">!cancel reset</span> — Reset abbrechen ❌</li><li><span class="cmd">wie ist der preis von btc?</span> — Krypto-Preisabfrage</li><li><span class="cmd">ich heiße Max</span> — Name speichern</li><li><span class="cmd">rücke code 4 leerzeichen nach rechts [leerzeile] code</span> — Belibige anzahl an Leerzeichen nach rechts rücken</li></ul><div class="cmd-footer">💡 Du kannst jederzeit weitere Befehle hinzufügen!</div></div>
        """

        if text_lc.strip() in ["zeige befehle", "alle befehle"]:
            try:
                return COMMANDS_HTML
            except Exception as e:
                logger.exception("Fehler bei zeige befehle: %s", e)
                return f"⚠️ Fehler beim test:\n{e}"

        # -------------------------
        # 6) Gehirnvisualisierung & Zustand
        # -------------------------
        if text_lc.strip() in ["zeige gehirn", "zeige synapsen", "zeige neuronales netz"]:
            try:
                html_path = self.erzeuge_synapsen_html()
                # Was passiert wenn: HTML/Export wird erzeugt und als Link zurückgegeben
                return (
                    "🧠 Ich habe mein neuronales Netzwerk erfolgreich visualisiert!\n\n"
                )
            except Exception as e:
                logger.exception("Fehler erzeuge_synapsen_html: %s", e)
                return f"⚠️ Fehler beim Erzeugen der Gehirnvisualisierung:\n{e}"

        if text_lc.strip() in ["gehirn zustand", "netzwerk status", "brain status"]:
            try:
                result = self.erzeuge_synapsen_html()

                # Wenn result strukturierte Informationen enthält, formatiere sie schön.
                if isinstance(result, dict) and "brain_summary" in result:
                    bs = result["brain_summary"]
                    local_summary = (
                        f"🧠 Gehirnstatus (lokal):\n"
                        f"- Neuronen: {bs.get('neuronen', 'n/a')}\n"
                        f"- Synapsen: {bs.get('synapsen', 'n/a')}\n"
                        f"- ⌀ Gewicht: {bs.get('durchschnittsgewicht', 0):.4f}\n"
                        f"- Varianz: {bs.get('aktivitätsvarianz', 0):.6f}\n"
                        f"- Synapsen/Neuron: {bs.get('verbindungen_pro_neuron', 0):.2f}\n"
                        f"- Dichte: {bs.get('netz_dichte', 0):.6f}"
                    )
                    server_msg = result.get("message", "")
                    return f"{local_summary}\n\n{server_msg}\n\n📊 Neuronale Analyse wurde erfolgreich durchgeführt."
                else:
                    # Fallback: result kann auch nur string/message sein.
                    summary = result.get("message", "Kein Status verfügbar.") if isinstance(result, dict) else str(result)
                    return f"{summary}\n\n📊 Neuronale Analyse wurde erfolgreich durchgeführt."
            except Exception as e:
                logger.exception("Fehler beim Ermitteln des Gehirnzustands: %s", e)
                return f"❌ Fehler beim Ermitteln des Gehirnzustands: {e}"

        # -------------------------
        # 7) Manuelle Reflexion / Erinnerung
        # -------------------------
        if text_lc.strip() in ["reflektiere", "zeige erinnerung", "zeige letzte erkenntnis"]:
            try:
                reflexion = self.reflektiere_letztes_thema()
                if reflexion and isinstance(reflexion, str):
                    return reflexion
                return "🤖 Ich habe gerade keine gespeicherten Erkenntnisse zum Reflektieren."
            except Exception as e:
                logger.exception("Fehler bei manueller Reflexion: %s", e)
                return "❌ Beim Erinnern ist etwas schiefgelaufen."

        # -------------------------
        # 8) Wenn eine Pending-Question offen ist -> Ja/Nein Verarbeitung (Follow-Up)
        # -------------------------
        if getattr(self, "_pending_question", None):
            frage = self._pending_question
            thema = frage.get("topic", "<unbekannt>")
            antwort_lc = text_lc

            positive = any(w in antwort_lc for w in ["ja", "klar", "gerne", "bitte", "unbedingt"])
            negative = any(w in antwort_lc for w in ["nein", "nicht", "passt", "danke", "kein"])

            if positive:
                # Was passiert wenn: Je nach Frage-Typ wird Folgeaktion ausgeführt (reddit / tiefer)
                try:
                    if frage.get("type") == "mehr_info":
                        return self._zeige_reddit(thema)
                    elif frage.get("type") == "tiefer":
                        return self._tiefer_eintauchen(thema)
                    else:
                        # generischer positive-ack
                        return f"Super — ich schaue gleich tiefer in **{thema}** rein."
                except Exception as e:
                    logger.exception("Fehler bei Follow-Up positive Aktion: %s", e)
                    return "❌ Fehler beim Weiterverarbeiten deiner Bestätigung."
            elif negative:
                # Benutzer lehnt ab -> pending clear
                self._pending_question = None
                return random.choice([
                    "Alles klar 😊 Dann sehen wir das als abgeschlossen an.",
                    "Okay 👍 Ich bleib neugierig, falls du später mehr wissen willst!",
                    "Kein Problem 😌 Ich bin bereit, wenn du wieder tiefer eintauchen willst."
                ])
            else:
                return random.choice([
                    "Ich bin mir nicht sicher, ob du *ja* oder *nein* meintest 🤔",
                    f"Meintest du, ich soll tiefer nach **{thema}** suchen?",
                    "Sag einfach *ja bitte* oder *nein danke* 😊"
                ])

        # -------------------------
        # 9) Krypto-Preis-Erkennung (priorisiert vor Smalltalk)
        # -------------------------
        coin_symbol = None
        preis_pattern = re.search(
            r"(?:wie\s+ist\s+(?:der|die|das)?\s*(?:aktuelle[nr]*\s+)?)?(?:preis|kurs|wert|kostet)\s+(?:von\s+)?([a-zA-Z0-9]{2,20})",
            text, re.IGNORECASE
        )
        if preis_pattern:
            coin_symbol = preis_pattern.group(1).strip()
        else:
            coin_match = re.search(
                r"\b(btc|bitcoin|eth|ethereum|dot|polkadot|sol|solana|ada|xrp|doge|avax|atom|link|ltc)\b",
                text_lc
            )
            if coin_match and any(w in text_lc for w in ["preis", "kurs", "wert", "steht", "aktuell", "kostet"]):
                coin_symbol = coin_match.group(1)

        if coin_symbol:
            # freundlicher Prefix je nach Formulierung
            if any(w in text_lc for w in ["wie ist", "was ist", "wie lautet"]):
                prefix = random.choice([
                    "Moment, ich schau mal eben nach 📊 …",
                    "Sekunde, ich prüfe das für dich 💫",
                    "Ich checke kurz den aktuellen Markt 💹",
                    "Lass mich fix nachsehen 📈"
                ])
            elif any(w in text_lc for w in ["aktuell", "momentan", "gerade"]):
                prefix = random.choice([
                    "Gerade liegt der Kurs bei:",
                    "Im Moment sieht es so aus:",
                    "Aktuell notiert der Preis bei:"
                ])
            else:
                prefix = random.choice([
                    "Hier ist der derzeitige Kurs:",
                    "Ich hab den Preis gefunden:",
                    "So steht der Coin aktuell:"
                ])

            try:
                preis_info = self.krypto_preis_kompakt(coin_symbol)
            except Exception as e:
                logger.exception("Fehler beim Abrufen des Krypto-Preises: %s", e)
                preis_info = f"❌ Konnte Preis für {coin_symbol.upper()} nicht abrufen: {e}"

            # Lernmechanismus: Einfache Verknüpfung in neuronales Netz (nicht-blockierend)
            try:
                coin_name = coin_symbol.upper()
                self.fm.create_neuron(coin_name, types=["coin"], sentence=text, source="user")
                for concept in ["preis", "kurs", "markt", "trend", "wert"]:
                    self.fm.create_neuron(concept, types=["concept"], sentence="", source="system")
                    self.verknuepfe_woerter(coin_name, concept, weight=0.8)
                self.last_topic = coin_name
            except Exception as e:
                logger.exception("[Synapse-Krypto-Fehler]: %s", e)

            suffix = random.choice([
                "📊 Faszinierend, wie volatil Kryptos sind!",
                "💡 Märkte ändern sich ständig – bleib wachsam!",
                "🔮 Der Markt bleibt unberechenbar!",
                "💬 Willst du, dass ich dir auch den Trend der letzten Tage zeige?"
            ])
            return f"{prefix}\n{preis_info}\n\n{suffix}"

        # -------------------------
        # 10) Smalltalk (nur, wenn keine Krypto-Frage)
        # -------------------------
        try:
            smalltalk_antwort = self.beantworte_smalltalk(text)
            if smalltalk_antwort:
                return smalltalk_antwort
        except Exception as e:
            logger.exception("Fehler beim Smalltalk-Matcher: %s", e)

        # -------------------------
        # 11) Lern-/Erkenntnis-Befehle
        # -------------------------
        if text_lc.startswith("lerne:"):
            match = re.match(r"lerne:\s*(.*?)\s*=\s*(.+)", text, re.IGNORECASE)
            if match:
                thema, inhalt = match.group(1).strip(), match.group(2).strip()
                return self.lerne_ausfuehrlich(thema, inhalt, "user")
            return "❌ Format: lerne: Thema = Inhalt"

        if text_lc.startswith("aussage:"):
            aussage = text.split(":", 1)[1].strip()
            return self.lerne_neue_aussage(aussage)

        if text_lc.startswith(("erkenntnis:", "was weißt du über")):
            thema = re.sub(r"^(erkenntnis:|was weißt du über)\s*", "", text, flags=re.IGNORECASE).strip()
            return self.formuliere_gelernte_erkenntnis(thema)

        # dedizierter Erkenntnis-Abruf (mehr Varianten)
        if re.match(r"^(erkenntnis:|was weißt du über|was weißt du von|was kennst du über)", text_lc):
            try:
                thema = re.sub(r"^(erkenntnis:|was weißt du über|was weißt du von|was kennst du über)\s*", "", text, flags=re.IGNORECASE).strip()
                if not thema:
                    return "Bitte sag mir, worüber du mehr wissen möchtest."
                return self.formuliere_gelernte_erkenntnis(thema)
            except Exception as e:
                logger.exception("Erkenntnis-Abruf Fehler: %s", e)
                return "Ich konnte dazu gerade keine Erkenntnis abrufen."

        # -------------------------
        # 12) Kontextbezogenes Erinnern ("Erzähl mir mehr")
        # -------------------------
        if re.search(r"\b(erzähl mir mehr|erzähl mehr|mehr darüber|mehr dazu|sag mir mehr)\b", text_lc):
            try:
                letztes_thema = self.fm.last_topic()
                if not letztes_thema:
                    return "Ich erinnere mich gerade an kein Thema. Sag mir, worüber du mehr erfahren willst."
                antwort = self.formuliere_gelernte_erkenntnis(letztes_thema)
                return f"Ich erinnere mich, du hast zuletzt über **{letztes_thema}** gesprochen.\n\n{antwort}"
            except Exception as e:
                logger.exception("Kontextabruf Fehler: %s", e)
                return "Ich konnte mich gerade nicht erinnern 😅"

        # -------------------------
        # 13) Autonomes Lernen aus Aussagen (keine Frage, >3 Wörter)
        # -------------------------
        if "?" not in text and len(text.split()) > 3:
            if not any(text_lc.startswith(x) for x in ["zeige", "iframe", "reset", "!smalltalk", "erkenntnis", "lerne", "aussage", "was ist", "wie"]):
                try:
                    learned = self.lerne_neue_aussage(text)
                    return f"{learned}\n\n💾 Ich habe das in mein neuronales Netzwerk integriert."
                except Exception as e:
                    logger.exception("Autonomes Lernen Fehler: %s", e)
                    # hier kein return: falls Lernen fehlschlägt, weiter zum nächsten Match

        # -------------------------
        # 14) Wissensfragen (starker Trigger)
        # -------------------------
        if re.match(r"^(was\s+ist|wer\s+ist|wer\s+war|was\s+sind|wie\s+funktioniert|was\s+bedeutet)\b", text_lc):
            # Was passiert wenn: delegiert an multi_source_suche() (z. B. Wikipedia, Web)
            return self.multi_source_suche(text, stilisiert=True)

        # zusätzliche Keywords, die eine externe Suche sinnvoll machen
        if any(w in text_lc for w in ["erzähl", "erkläre", "info", "sag mir", "was weißt du"]):
            return self.multi_source_suche(text, stilisiert=True)

        # -------------------------
        # 15) Spontane Reflexion (kleine Chance)
        # -------------------------
        try:
            if random.random() < 0.45:  # 45 % Chance auf spontane Erinnerung
                reflexion = self.reflektiere_letztes_thema()
                if reflexion and isinstance(reflexion, str):
                    return reflexion
        except Exception as e:
            logger.exception("Fehler bei automatischer Reflexion: %s", e)

        # -------------------------
        # 16) Menschlicher Fallback (freundlich & offen)
        # -------------------------
        fallback_reaktionen = [
            "Interessant 😄 – erzähl mir mehr!",
            "Das klingt spannend! Magst du mir genauer sagen, was du meinst?",
            "Hm, das versteh ich noch nicht ganz 🤔",
            "Klingt nach einer Geschichte! Was steckt dahinter?",
            "Oh! Darüber kann man bestimmt reden 😊"
        ]
        return random.choice(fallback_reaktionen)





















    def multi_source_suche(self, thema: str, stilisiert: bool = False) -> str:
        """
        Professionelle, mehrstufige Multi-Source-Suche.
        Nutzt:
            - interne Wissensdatenbank (Memory)
            - Wikipedia (REST-API)
            - DuckDuckGo/Google-Scrape
            - Reddit-Auswertung
            - Wikidata (optional)
        Führt Filterung, Deduplizierung, Ranking via Embeddings,
        Faktenlernen, Neuronen-/Synapsenbildung,
        Memory-Logging und pending-states durch.
        """

        # ----------------------------------------------------
        # 1️⃣ Keyword aus Anfrage extrahieren
        # ----------------------------------------------------
        keyword = self.extract_keyword(thema)

        # Schutz
        if not keyword:
            return "❌ Ich konnte kein Thema erkennen."

        keyword_clean = keyword.strip()

        # ----------------------------------------------------
        # 2️⃣ Prüfen, ob Wissen bereits existiert
        # ----------------------------------------------------
        vorwissen = self.finde_wissen(keyword_clean)
        if vorwissen:
            # Verstärke neuronale Struktur, da Wissen erneut abgerufen wird
            try:
                self.fm.create_neuron(keyword_clean, types=["topic"], sentence=vorwissen, source="memory")
                self.sn.strengthen(keyword_clean, "wissen", step=0.25)
            except Exception:
                pass

            return (
                f"💡 Ich erinnere mich! Über **{keyword_clean}** weiß ich Folgendes:\n\n"
                f"{vorwissen}\n\n"
                f"✨ Soll ich neuere Informationen dazu abrufen?"
            )

        # ----------------------------------------------------
        # 3️⃣ Multi-Source: Daten sammeln
        # ----------------------------------------------------
        quellen = []

        # 🟦 A) Wikipedia (Hauptquelle)
        wiki = self.wikipedia_suche(keyword_clean)
        if wiki and not wiki.startswith("xIch konnte dazu"):
            quellen.append(("wikipedia", wiki, 0.9))

        # 🟪 B) DuckDuckGo/Google-Scrape
        try:
            ddg = self.google_scrape(keyword_clean)
            if ddg:
                quellen.append(("duckduckgo", ddg, 0.6))
        except:
            pass

        # 🟥 C) Reddit-Sentiments
        try:
            red = self.reddit_suche(keyword_clean)
            if red and "Fehler" not in red:
                quellen.append(("reddit", red, 0.4))
        except:
            pass

        # 🟨 D) Wikidata
        try:
            if hasattr(self, "suche_wikidata"):
                wd = self.suche_wikidata(keyword_clean)
                if wd:
                    quellen.append(("wikidata", wd, 0.7))
        except:
            pass

        if not quellen:
            return f"❌ Ich konnte zu **{keyword_clean}** leider nichts finden."

        # ----------------------------------------------------
        # 4️⃣ NLP-Filterung, Deduplizierung & Scoring
        # ----------------------------------------------------
        scored = []

        if self.model:
            query_emb = self.model.encode(keyword_clean, convert_to_tensor=True)

            for quelle, text, trust in quellen:
                clean = text.replace("\n", " ").strip()
                clean = " ".join(clean.split())
                if not clean:
                    continue

                emb = self.model.encode(clean[:300], convert_to_tensor=True)
                sim = float(util.pytorch_cos_sim(query_emb, emb))
                score = sim * trust

                scored.append((score, quelle, clean))

            # Sortieren nach Score
            scored.sort(reverse=True, key=lambda x: x[0])
        else:
            scored = [(1, q, t) for q, t, _ in quellen]

        # Beste 2–3 Quellen auswählen
        beste_quellen = [t for _, _, t in scored[:3]]
        bester_text = "\n\n".join(beste_quellen)

        # ----------------------------------------------------
        # 5️⃣ Automatisches NLP-Faktenlernen aus Wikipedia
        # ----------------------------------------------------
        if wiki:
            try:
                lines = wiki.split(".")
                for line in lines:
                    l = line.strip()
                    if not l:
                        continue

                    # Wörter extrahieren
                    begriffe = [w for w in l.split() if len(w) > 3]
                    for wort in begriffe:
                        wort_clean = wort.strip("()[]/,.").title()
                        if wort_clean.lower() == keyword_clean.lower():
                            continue

                        self.fm.create_neuron(wort_clean, types=["fact"], sentence=l, source="wiki")
                        self.verknuepfe_woerter(keyword_clean, wort_clean, weight=0.7)

                    # Jahreszahlen → Zeitbezug
                    if any(char.isdigit() for char in l):
                        self.fm.create_neuron("Zeit", types=["concept"], source="wiki")
                        self.verknuepfe_woerter(keyword_clean, "Zeit", weight=0.95)

            except Exception as e:
                print(f"[Faktenlernen] Fehler: {e}")

        # ----------------------------------------------------
        # 6️⃣ Neuronen/Synapsen – semantische Integration
        # ----------------------------------------------------
        try:
            self.fm.create_neuron(keyword_clean, types=["topic"], sentence=bester_text, source="multi-source")

            # Subbegriffe aus Keyword
            for tok in keyword_clean.split():
                if len(tok) > 3:
                    self.fm.create_neuron(tok, types=["subconcept"], source="topic")
                    self.verknuepfe_woerter(keyword_clean, tok, weight=0.85)

            # semantische Verbindung zum letzten Thema
            if getattr(self, "last_topic", None) and self.model:
                try:
                    emb1 = self.model.encode(keyword_clean, convert_to_tensor=True)
                    emb2 = self.model.encode(self.last_topic, convert_to_tensor=True)
                    sim = float(util.pytorch_cos_sim(emb1, emb2))
                    if sim > 0.55:
                        self.verknuepfe_woerter(keyword_clean, self.last_topic, weight=sim)
                except Exception:
                    pass

            self.last_topic = keyword_clean

        except Exception as e:
            print(f"[Synapsenbildung Fehler]: {e}")

        # ----------------------------------------------------
        # 7️⃣ Knowledge speichern + Pending-State setzen
        # ----------------------------------------------------
        self.speichere_wissen(keyword_clean, bester_text)
        self.logge_lernschritt(keyword_clean, "multi-source", bester_text)
        self._pending_refresh = {"keyword": keyword_clean}

        # ----------------------------------------------------
        # 8️⃣ Ausgabe formatieren
        # ----------------------------------------------------
        intro_varianten = [
            f"🧠 Das Thema **{keyword_clean}** ist wirklich spannend:",
            f"🔍 Ich habe ausführlich zu **{keyword_clean}** recherchiert:",
            f"📘 Hier ist eine kompakte Übersicht zu **{keyword_clean}**:",
            f"✨ Analyse zu **{keyword_clean}**:",
        ]
        intro = random.choice(intro_varianten)

        outro = (
            f"✨ Soll ich noch *aktuellere* Informationen zu **{keyword_clean}** abrufen?"
        )

        antwort = f"{intro}\n\n{bester_text}\n\n{outro}"

        if stilisiert:
            antwort += "\n\n" + random.choice([
                "Ich liebe solche Wissensreisen! 💫",
                "Das war ein spannender Deep Dive 😄",
                "Wissen macht Spaß! 🍯"
            ])

        return antwort














    def _zeige_reddit(self, thema: str) -> str:
        """Phase 2: Reddit-Suche nach Bestätigung."""
        reddit_text = self.reddit_suche(thema)
        if not reddit_text or reddit_text.startswith("❌"):
            self._pending_question = None
            return f"👽 Ich habe auf Reddit leider nichts zu **{thema}** gefunden."

        self._pending_question = {"type": "tiefer", "topic": thema}
        return (
            f"👽 Ich habe Reddit nach Stimmen zu **{thema}** durchsucht …\n"
            f"{reddit_text}\n\n"
            f"💬 Willst du, dass ich noch tiefer auf Reddit oder Wikipedia nachsehe?"
        )

    def _tiefer_eintauchen(self, thema: str) -> str:
        """Phase 3: Tieferes Nachschlagen (Wikipedia-Details + DuckDuckGo)."""
        wiki_details = self.wikipedia_suche(f"{thema} Details")
        google_fallback = self.google_scrape(thema)
        texte = []
        if wiki_details:
            texte.append(f"📚 {wiki_details}")
        if google_fallback:
            texte.append(f"🌐 {google_fallback}")
        if not texte:
            return f"Ich konnte keine weiteren Informationen zu **{thema}** finden."
        self._pending_question = None
        return (
            f"✨ Ich tauche tiefer in **{thema}** ein …\n\n"
            + "\n\n".join(texte)
            + "\n\nSo was begeistert mich echt! 🤖"
        )













    def extract_and_store_facts(self, topic, text):
        t = topic.lower()

        # Höhe erkennen
        h = re.search(r"(\d{1,2}\.?\d{0,3})\s*m", text.lower())
        if h:
            self.speichere_wissen(f"höhe {t}", h.group(1) + " Meter", überschreiben=True)

        # Lage erkennen
        if "nepal" in text.lower():
            self.speichere_wissen(f"wo liegt {t}", "Nepal", überschreiben=True)
        if "china" in text.lower():
            self.speichere_wissen(f"wo liegt {t}", "China", überschreiben=True)




    # ===========================
    # 🧠 DENKEN → WISSEN
    # ===========================

    def speichere_denk_ergebnis(self, seed, result):
        """
        Wandelt Denkresultate automatisch in Wissen um.
        """

        if not result or "selected" not in result:
            return

        thema = seed
        erkenntnis = result["selected"]

        # neuronale Speicherung
        self.fm.create_neuron(
            erkenntnis,
            types=["thought", "insight"],
            sentence=f"Ergebnis eines autonomen Denkprozesses zu {thema}",
            source="thinking"
        )

        # Synaptische Verknüpfung
        self.sn.add_synapse(thema, erkenntnis, weight=1.2)

        # Wissens-DB (sanft)
        try:
            self.speichere_wissen(
                frage=f"gedanke zu {thema}",
                antwort=erkenntnis,
                überschreiben=False
            )
        except Exception:
            pass





# xenronbrain class END --------












# ===========================
# 🧠 THOUGHT LOOP EXTENSION
# ===========================

class ThoughtLoop:
    """
    Interner Denkzyklus für Xenron.
    Führt rekursive neuronale Traversierungen durch
    und erzeugt eigenständige Gedankengänge.
    """

    def __init__(self, brain, max_cycles=5):
        self.brain = brain
        self.max_cycles = max_cycles
        self.active = False

    def start(self, seed: str):
        if not seed:
            return []

        self.active = True
        thoughts = []
        current = seed

        for cycle in range(self.max_cycles):
            if not self.active:
                break

            # Synaptische Exploration
            results = self.brain.sn.traverse(
                start=current,
                depth=2,
                breadth=5
            )

            if not results:
                break

            # stärksten Gedanken wählen
            best = results[0]
            slug, word, score, depth = best

            thoughts.append({
                "cycle": cycle,
                "concept": word,
                "score": score
            })

            # Arbeitsgedächtnis aktualisieren
            self.brain.working_memory.store(word, score)

            # Gedanke wird neuer Fokus
            current = word

        self.active = False
        return thoughts






# ===========================
# 🧠 WORKING MEMORY
# ===========================

class WorkingMemory:
    """
    Kurzzeitgedächtnis mit langsamem Zerfall.
    Ermöglicht Gedankenkohärenz über mehrere Zyklen.
    """

    def __init__(self, decay=0.85, limit=12):
        self.decay = decay
        self.limit = limit
        self.memory = {}

    def store(self, concept: str, strength: float):
        self.memory[concept] = max(self.memory.get(concept, 0), strength)
        self._decay()

    def _decay(self):
        for k in list(self.memory.keys()):
            self.memory[k] *= self.decay
            if self.memory[k] < 0.05:
                del self.memory[k]

        # Begrenzen
        if len(self.memory) > self.limit:
            self.memory = dict(
                sorted(self.memory.items(), key=lambda x: x[1], reverse=True)[:self.limit]
            )

    def snapshot(self):
        return dict(self.memory)





# ===========================
# 🧠 CRITIC MODULE
# ===========================

class Critic:
    """
    Bewertet Gedanken nach Kohärenz, Wiederholung und Relevanz.
    """

    def evaluate(self, thoughts, working_memory):
        if not thoughts:
            return None

        scores = {}
        for t in thoughts:
            concept = t["concept"]
            base = t["score"]

            memory_bonus = working_memory.memory.get(concept, 0) * 0.5
            scores[concept] = base + memory_bonus

        # bester Gedanke
        best = max(scores.items(), key=lambda x: x[1])
        return {
            "selected": best[0],
            "score": round(best[1], 3)
        }







# ===========================
# 🧠 BACKGROUND THINKING ENGINE
# ===========================

class BackgroundThinker(threading.Thread):
    """
    Führt periodisch autonome Denkprozesse aus.
    """

    def __init__(self, brain, interval=30):
        super().__init__(daemon=True)
        self.brain = brain
        self.interval = interval
        self.running = True

    def run(self):
        while self.running:
            try:
                # Ausgangspunkt bestimmen
                seed = (
                    self.brain.fm.last_topic()
                    or self.brain.last_topic
                    or self.brain.pick_goal_neuron()
                )

                if seed:
                    result = self.brain.denke(seed)

                    # optional: Debug
                    print(f"[BACKGROUND THINKING] {result}")

            except Exception as e:
                print(f"[BACKGROUND THINKING ERROR] {e}")

            time.sleep(self.interval)

    def stop(self):
        self.running = False










# ==========================================================
# 🚀 NEUER XENRON WAITLIST WORKER (HTTP-basiert)
# ==========================================================

# URLs für Serverkommunikation
WAITLIST_URL = ""
WAITLIST_UPDATE_URL = ""
WRITE_URL = ""

# ----------------------------------------------------------
# Waitlist-Management (HTTP)
# ----------------------------------------------------------
def lade_waitlist():
    """Holt aktuelle xenron_waitlist.json per HTTP."""
    try:
        r = requests.get(WAITLIST_URL, timeout=10)
        if r.status_code == 200:
            return r.json()
        else:
            print(f"[lade_waitlist] HTTP-Status {r.status_code}")
    except Exception as e:
        print(f"[lade_waitlist] Fehler: {e}")
    return []

def finde_offenen_auftrag():
    """Gibt ersten offenen Auftrag aus der Warteliste zurück."""
    wl = lade_waitlist()
    for eintrag in wl:
        if eintrag.get("status") == "offen":
            return eintrag
    return None

def setze_auftrag_geschlossen(frageid):
    """Setzt einen Auftrag in der Waitlist auf 'geschlossen'."""
    wl = lade_waitlist()
    geändert = False
    for e in wl:
        if e.get("frageid") == frageid:
            e["status"] = "geschlossen"
            e["bearbeitet"] = now_iso()
            geändert = True
    if not geändert:
        return
    try:
        r = requests.post(WAITLIST_UPDATE_URL, json=wl, timeout=10)
        if r.status_code != 200:
            print(f"[setze_auftrag_geschlossen] Serverantwort: {r.status_code} {r.text}")
    except Exception as e:
        print(f"[setze_auftrag_geschlossen] Fehler: {e}")

# ----------------------------------------------------------
# Hauptloop des Workers
# ----------------------------------------------------------
def main_loop():
    print("🚀 Xenron Waitlist Worker gestartet (HTTP-Modus)…")
    while True:
        auftrag = finde_offenen_auftrag()
        if not auftrag:
            time.sleep(5)
            continue

        apikey  = auftrag.get("apikey")
        frage   = auftrag.get("frage")
        frageid = auftrag.get("frageid")

        if not (apikey and frage and frageid):
            print("⚠️ Ungültiger Auftrag, überspringe…")
            time.sleep(2)
            continue

        print(f"\n🧠 Bearbeite Auftrag {frageid} ({apikey}): {frage}")

        try:
            brain = XenronBrain(api_key=apikey)
            antwort = brain.verarbeite_eingabe(frage)
        except Exception as e:
            print(f"❌ Fehler bei Verarbeitung: {e}")
            time.sleep(3)
            continue

        # Chatdaten erzeugen
        chatdata = [
            {"role": "user", "content": frage, "id": frageid, "timestamp": now_iso()},
            {"role": "ai", "content": antwort, "id_bezug": frageid, "timestamp": now_iso()}
        ]

        # Ergebnis an den Webserver schreiben
        try:
            res = requests.post(WRITE_URL, params={"apikey": apikey}, json=chatdata, timeout=10)
            if res.status_code == 200:
                print(f"💾 Antwort gespeichert für {apikey}: {antwort}")
            else:
                print(f"⚠️ Upload fehlgeschlagen ({res.status_code})")
        except Exception as e:
            print(f"⚠️ Upload-Fehler: {e}")

        # Auftrag schließen
        setze_auftrag_geschlossen(frageid)
        print(f"✅ Auftrag {frageid} abgeschlossen.\n")
        time.sleep(2)

# ----------------------------------------------------------
# Startpunkt
# ----------------------------------------------------------
if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print("\n🛑 xenRon wurde erfolgreich Manuell beendet.")
