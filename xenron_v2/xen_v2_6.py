#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
import requests
import random
import sqlite3
import re
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

FETCH_URL   = "https://www.xenexai.com/xenron/v2/worker/data/xenron_chat.json"
UPLOAD_URL  = "https://www.xenexai.com/xenron/v2/worker/write_json.php"
INTERVAL    = 5
DB_FILENAME = "brain.db"

API_USAGE_URL = "https://www.xenexai.com/xenron/v2/worker/data/api_usage.json"
CHAT_PATH     = "https://www.xenexai.com/xenron/v2/worker/data"

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

    def __init__(self, db_name=DB_FILENAME, api_key=None):
#        self.conn = sqlite3.connect(db_name)
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.api_key = api_key
        self.last_topic = None
        self.user_name = None
        self.smalltalk = self.lade_smalltalk()
        self._init_db()

# neu
        self.model = SentenceTransformer('./models/paraphrase-multilingual-MiniLM-L12-v2')
# ---

        # Neue Logik: remote usernames laden
        self.remote_usernames = self.lade_remote_usernames()

        if api_key:
            # 1) Priorität: remote JSON
            self.user_name = self.remote_usernames.get(api_key)
            # 2) Fallback auf lokale DB
            if not self.user_name:
                self.user_name = self.benutzername_von_api_key(api_key)



    def _init_db(self):
        c = self.conn.cursor()

        # 1) Tabelle anlegen, falls brandneu
        c.execute("""
            CREATE TABLE IF NOT EXISTS benutzer (
                api_key TEXT PRIMARY KEY,      -- für neue DBs gleich korrekt
                name     TEXT
            )
        """)

        # 2) Prüfen, ob eine alte Tabelle noch KEIN api_key‑Feld hat
        cols = [row[1] for row in c.execute("PRAGMA table_info(benutzer)")]
        if "api_key" not in cols:
            print("[🛠] Upgrade: Spalte 'api_key' wird ergänzt …")
            c.execute("ALTER TABLE benutzer ADD COLUMN api_key TEXT")  # KEIN UNIQUE hier!
        # Danach eindeutigen Index anlegen
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_benutzer_api_key ON benutzer(api_key)")


        # Wissenstabelle
        c.execute("""
            CREATE TABLE IF NOT EXISTS wissen (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                frage TEXT,
                antwort TEXT,
                timestamp TEXT
            )
        """)

        # Lernprotokoll-Tabelle
        c.execute("""
            CREATE TABLE IF NOT EXISTS lernprotokoll (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thema TEXT,
                quelle TEXT,
                inhalt TEXT,
                timestamp TEXT
            )
        """)

        # Wörter-Lernsystem
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

        # Begriffliche Assoziationen (einfache neuronale Netz-Struktur)
        c.execute("""
            CREATE TABLE IF NOT EXISTS assoziationen (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                wort1 TEXT,
                wort2 TEXT,
                gewicht INTEGER DEFAULT 1,
                api_key TEXT DEFAULT ''
            )
        """)


        # --- Upgrade: Spalte `api_key` hinzufügen, falls sie fehlt ---
        try:
            c.execute("ALTER TABLE wissen ADD COLUMN api_key TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Spalte existiert bereits

        try:
            c.execute("ALTER TABLE lernprotokoll ADD COLUMN api_key TEXT DEFAULT ''")
        except sqlite3.OperationalError:
            pass  # Spalte existiert bereits

        # Optional: Vorhandene Datensätze mit einem Dummy-Key aktualisieren
        # c.execute("UPDATE wissen SET api_key = 'default' WHERE api_key IS NULL OR api_key = ''")
        # c.execute("UPDATE lernprotokoll SET api_key = 'default' WHERE api_key IS NULL OR api_key = ''")

        self.conn.commit()




    def lade_remote_usernames(self):
        """Lädt alle api_key → username Paare von der Remote-JSON."""
        try:
            r = requests.get(f"{CHAT_PATH}/api_users.json", timeout=5)
            r.raise_for_status()
            data = r.json()
            return data  # { api_key: username, … }
        except Exception as e:
            print(f"[Fehler beim Laden remote usernames]: {e}")
            return {}


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




    def semantisch_aehnlichste_frage(self, frage, threshold=0.75):
        """Finde die ähnlichste gespeicherte Frage via SentenceTransformer."""
        c = self.conn.cursor()
        c.execute("SELECT frage, antwort FROM wissen WHERE api_key = ?", (self.api_key,))
        eintraege = c.fetchall()

        if not eintraege:
            return None

        frage_emb = self.model.encode(frage, convert_to_tensor=True)
        beste_frage, beste_antwort, bester_score = None, None, 0

        for bekannte_frage, antwort in eintraege:
            bekannte_emb = self.model.encode(bekannte_frage, convert_to_tensor=True)
            score = util.cos_sim(frage_emb, bekannte_emb).item()
            if score > bester_score:
                beste_frage = bekannte_frage
                beste_antwort = antwort
                bester_score = score

        if bester_score >= threshold:
            print(f"[🧠] Semantisch ähnlich zu «{beste_frage}» (Score: {bester_score:.2f})")
            return beste_antwort
        return None



    def verknuepfe_woerter(self, wort1: str, wort2: str):
        """Verstärkt oder erstellt Assoziation zwischen zwei Wörtern"""
        wort1, wort2 = wort1.strip().lower(), wort2.strip().lower()
        if wort1 == wort2:
            return
        c = self.conn.cursor()
        # Prüfe, ob Verbindung existiert
        c.execute("""
            SELECT id, gewicht FROM assoziationen
            WHERE wort1 = ? AND wort2 = ? AND api_key = ?
        """, (wort1, wort2, self.api_key))
        row = c.fetchone()
        if row:
            # Verstärken
            c.execute("""
                UPDATE assoziationen SET gewicht = gewicht + 1 WHERE id = ?
            """, (row[0],))
        else:
            # Neu anlegen
            c.execute("""
                INSERT INTO assoziationen (wort1, wort2, gewicht, api_key)
                VALUES (?, ?, ?, ?)
            """, (wort1, wort2, 1, self.api_key))
        self.conn.commit()




    def _format_gpt_style(self, keyword: str, inhalte: list[str]) -> str:
        """
        Baut aus den einzelnen Quellen eine ChatGPT-hafte Antwort:
        - Titel, Emojis, Bullet-Points, Highlights
        - speichert das Endresultat in DB
        """
        # Baue die Überschrift
        header = f"✅ Hier ist eine ausführliche Analyse zu **{keyword}**:\n\n"
        parts = []
        # Durchlaufe jede Quelle (z.B. "📘 Wikipedia: …", "🌐 DuckDuckGo: …")
        for text in inhalte:
            # Füge als separaten Block ein
            parts.append(f"{text}\n")
        # Füge noch eine Einschätzung oder Fußnote hinzu
        footer = "\n🔍 **Quellen:** Wikipedia, DuckDuckGo\n" \
                 "💾 **Gespeichert in deinem Wissen**"
        return header + "\n".join(parts) + footer








    def finde_wissen(self, frage, fuzzy=True):
        c = self.conn.cursor()
        c.execute("SELECT antwort FROM wissen WHERE frage = ? AND api_key = ?", (frage, self.api_key))
        row = c.fetchone()
        if row:
            return row[0]
        if fuzzy:
            pattern = f"%{'%'.join(frage.lower().split())}%"
            c.execute("SELECT frage, antwort FROM wissen WHERE LOWER(frage) LIKE ?", (pattern,))
            row = c.fetchone()
            if row:
                return row[1]
        return None

    def speichere_wissen(self, frage, antwort, überschreiben=False):
        c = self.conn.cursor()
        if überschreiben:
            c.execute("""
                INSERT INTO wissen (frage, antwort, timestamp, api_key)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(frage) DO UPDATE SET
                    antwort = excluded.antwort,
                    timestamp = excluded.timestamp
            """, (frage, antwort, datetime.now(timezone.utc).isoformat(), self.api_key))
        else:
            c.execute("""
                INSERT OR IGNORE INTO wissen (frage, antwort, timestamp, api_key)
                VALUES (?, ?, ?, ?)
            """, (frage, antwort, datetime.now(timezone.utc).isoformat(), self.api_key))


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

        # Automatisch mit Thema verknüpfen
        if thema:
            for teil in thema.lower().split():
                self.verknuepfe_woerter(wort, teil)


    def erstelle_wort_dashboard(self, limit: int = 50):
        c = self.conn.cursor()
        c.execute("""
            SELECT wort, thema, quelle, timestamp
            FROM gelernte_woerter
            WHERE api_key = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (self.api_key, limit))
        eintraege = c.fetchall()

        if not eintraege:
            return "📘 Ich habe noch keine Wörter gelernt."

        lines = ["🧠 **GELERNTE WÖRTER**\n"]
        for i, (wort, thema, quelle, timestamp) in enumerate(eintraege, 1):
            lines.append(
                f"{i}. **{wort}** (aus: {thema}, Quelle: {quelle}, {timestamp[:19].replace('T', ' ')})"
            )
        return "\n".join(lines)


    def zeige_assoziationen(self, wort: str, limit: int = 10) -> str:
        wort = wort.lower().strip()
        c = self.conn.cursor()
        c.execute("""
            SELECT wort2, gewicht FROM assoziationen
            WHERE wort1 = ? AND api_key = ?
            ORDER BY gewicht DESC
            LIMIT ?
        """, (wort, self.api_key, limit))
        eintraege = c.fetchall()

        if not eintraege:
            return f"Ich kenne keine Begriffe, die mit **{wort}** verbunden sind."

        lines = [f"🔗 Begriffe, die mit **{wort}** assoziiert sind:"]
        for i, (w2, gewicht) in enumerate(eintraege, 1):
            lines.append(f"{i}. **{w2}** (Stärke: {gewicht})")
        return "\n".join(lines)



    def assoziierte_satzbildung(self, zentralbegriff: str) -> str:
        zentralbegriff = zentralbegriff.strip().lower()
        c = self.conn.cursor()
        c.execute("""
            SELECT wort2, gewicht FROM assoziationen
            WHERE wort1 = ? AND api_key = ?
            ORDER BY gewicht DESC LIMIT 5
        """, (zentralbegriff, self.api_key))
        eintraege = c.fetchall()

        if not eintraege:
            return f"Ich kenne noch keine Begriffe, die mit **{zentralbegriff}** verknüpft sind."

        sätze = []
        for wort2, gewicht in eintraege:
            satz = random.choice([
                f"**{zentralbegriff.capitalize()}** steht in enger Verbindung mit **{wort2}**.",
                f"Wenn ich an **{zentralbegriff}** denke, fällt mir auch **{wort2}** ein.",
                f"**{wort2}** ist mit **{zentralbegriff}** verknüpft – das habe ich gelernt.",
                f"Zwischen **{zentralbegriff}** und **{wort2}** besteht ein Zusammenhang."
            ])
            sätze.append(satz)

        return "🤖 Hier ist, was ich über **" + zentralbegriff + "** denke:\n" + "\n".join(sätze)




    def formuliere_gelernte_erkenntnis(self, limit: int = 1) -> str:
        """
        Baut aus einem oder mehreren gelernten Wörtern sinnvolle Sätze im Stil:
        - „Ich habe gelernt, dass ‹Wort› mit ‹Thema› zu tun hat.“
        - „‹Wort› bedeutet im Kontext von ‹Thema›: ‹Bedeutung›“
        """
        c = self.conn.cursor()
        c.execute("""
            SELECT wort, bedeutung, thema, quelle FROM gelernte_woerter
            WHERE api_key = ?
            ORDER BY RANDOM() LIMIT ?
        """, (self.api_key, limit))
        eintraege = c.fetchall()

        if not eintraege:
            return "Ich habe bisher noch keine Wörter gelernt."

        sätze = []
        for wort, bedeutung, thema, quelle in eintraege:
            satz = random.choice([
                f"Ich erinnere mich an das Wort **{wort}**, das mit dem Thema **{thema}** aus der Quelle {quelle} zu tun hat.",
                f"**{wort}** bedeutet im Zusammenhang mit **{thema}** ungefähr: {bedeutung[:100]}…",
                f"Ich habe das Wort **{wort}** gelernt – es kommt in {quelle} im Kontext von **{thema}** vor.",
                f"{wort.capitalize()} ist eines der Wörter, die ich beim Lernen über **{thema}** gefunden habe."
            ])
            sätze.append(satz)
        return "\n".join(sätze)



    def logge_lernschritt(self, thema, quelle, inhalt):
        c = self.conn.cursor()
        c.execute("""
            INSERT INTO lernprotokoll (thema, quelle, inhalt, timestamp)
            VALUES (?, ?, ?, ?)
        """, (thema, quelle, inhalt, datetime.now(timezone.utc).isoformat()))
        self.conn.commit()

    def hole_benutzername(self):
        c = self.conn.cursor()
        c.execute("SELECT name FROM benutzer WHERE id = 1")
        row = c.fetchone()
        return row[0] if row else None












# API NEW
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
        """Ändert den Username für den aktuellen API-Key auf dem Server."""
        try:
            url = f"https://www.xenexai.com/xenron/v2/worker/get-username.php"
            params = {"apikey": self.api_key, "name": neuer_name}
            r = requests.post(url, data=params, timeout=5)
            r.raise_for_status()
            # Bei Erfolg lokal updaten
            self.remote_usernames[self.api_key] = neuer_name
            self.setze_benutzername_db(neuer_name, self.api_key)
            self.user_name = neuer_name
            return True
        except Exception as e:
            print(f"[Fehler beim Aktualisieren remote username]: {e}")
            return False



    @lru_cache(maxsize=512)
    def wikipedia_suche(self, thema: str) -> str:
        thema = thema.strip().title()
        if not thema:
            print("[Wikipedia]: Leeres Thema übergeben.")
            return ""

        url = f"https://de.wikipedia.org/api/rest_v1/page/summary/{thema.replace(' ', '_')}"
        headers = {"User-Agent": "xenRonBot/1.0"}

        try:
            response = requests.get(url, headers=headers, timeout=5)
            if response.status_code == 200:
                data = response.json()
                extract = data.get("extract", "").strip()

                if extract:
                    print(f"[📘 Wikipedia]: Eintrag gefunden zu «{thema}»")
                    self.speichere_wissen(frage=thema, antwort=extract)
                    return extract
                else:
                    print(f"[📘 Wikipedia]: Kein Text-Inhalt zu «{thema}»")
            else:
                print(f"[📘 Wikipedia]: HTTP {response.status_code} bei Thema «{thema}»")



            if not extract and "title" in data:
                extract = f"{data['title']} ist ein Begriff, zu dem ich leider keinen ausführlichen Text gefunden habe."



        except requests.exceptions.Timeout:
            print(f"[📘 Wikipedia]: TIMEOUT bei Thema «{thema}»")
        except requests.exceptions.RequestException as e:
            print(f"[📘 Wikipedia Fehler]: {e}")
        except Exception as e:
            print(f"[📘 Wikipedia Unerwartet]: {e}")

        # === Fallback: DuckDuckGo verwenden ===
        print(f"[🌐 DuckDuckGo Fallback]: Suche nach «{thema}»")
        fallback = self.google_scrape(thema)
        if fallback:
            self.speichere_wissen(frage=thema, antwort=fallback)
            return fallback

        return ""


    @lru_cache(maxsize=256)
    def reddit_suche(self, thema: str) -> str:
        try:
            url = f"https://api.pushshift.io/reddit/search/comment/?q={thema}&size=5"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                data = response.json()
                comments = [c["body"] for c in data.get("data", []) if "body" in c]
                if comments:
                    combined = "\n".join(comments[:3])
                    return combined.strip()
        except Exception as e:
            print(f"[Reddit Fehler]: {e}")
        return ""


    @lru_cache(maxsize=256)
    def coingecko_suche(self, thema: str) -> str:
        try:
            # 1. Coin suchen
            search_url = f"https://api.coingecko.com/api/v3/search?query={thema}"
            search_res = requests.get(search_url, timeout=5).json()
            coins = search_res.get("coins", [])
            if not coins:
                return ""
            coin_id = coins[0]["id"]

            # 2. Coin-Daten abrufen
            coin_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}?localization=false&market_data=true"
            coin_res = requests.get(coin_url, timeout=5).json()

            name = coin_res["name"]
            desc = coin_res["description"]["en"][:300].strip().replace('\n', ' ')
            price = coin_res["market_data"]["current_price"]["eur"]
            rank = coin_res["market_data"]["market_cap_rank"]
            homepage = coin_res["links"]["homepage"][0]

            return f"**{name}** (Rank #{rank}) kostet aktuell **{price:.2f} EUR**. {desc}\n🔗 {homepage}"

        except Exception as e:
            print(f"[CoinGecko Fehler]: {e}")
            return ""



    def multi_source_suche(self, thema: str, stilisiert: bool = False) -> str:
        """
        Intelligente Suche: Erst Wikipedia, dann DuckDuckGo – alles kombiniert.
        Optional formatiert im GPT-Stil zurückgeben.
        """
        keyword = self.extract_keyword(thema)
        texte = []

        # Versuche zuerst Wikipedia
        wiki_text = self.wikipedia_suche(keyword)
        if wiki_text:
            texte.append(f"📘 Wikipedia:\n{wiki_text}")
            self.logge_lernschritt(keyword, "Wikipedia", wiki_text)

            # Lerne Wörter aus Wikipedia
            for wort in re.findall(r'\b\w{5,}\b', wiki_text):  # nur längere Wörter
                self.lerne_wort(wort, wiki_text[:250], thema=keyword, quelle="Wikipedia")



        # Falls Wikipedia nichts liefert → DuckDuckGo
        if not wiki_text:
            ddg_text = self.google_scrape(keyword)
            if ddg_text:
                texte.append(f"🌐 DuckDuckGo:\n{ddg_text}")
                self.logge_lernschritt(keyword, "DuckDuckGo", ddg_text)

                for wort in re.findall(r'\b\w{5,}\b', ddg_text):  # z. B. mindestens 5 Buchstaben
                    self.lerne_wort(wort, ddg_text[:250], thema=keyword, quelle="DuckDuckGo")


#        if stilisiert:
#            reflektion = self.formuliere_gelernte_erkenntnis(limit=1)
#            return self._format_gpt_style(keyword, texte) + f"\n\n🤖 {reflektion}"


        # Zusätzlich: Reddit als Quelle
        reddit_text = self.reddit_suche(keyword)
        if reddit_text:
            texte.append(f"👽 Reddit:\n{reddit_text}")
            self.logge_lernschritt(keyword, "Reddit", reddit_text)
            for wort in re.findall(r'\b\w{5,}\b', reddit_text):
                self.lerne_wort(wort, reddit_text[:250], thema=keyword, quelle="Reddit")


        # CoinGecko: Wenn Thema Krypto sein könnte
        cg_text = self.coingecko_suche(keyword)
        if cg_text:
            texte.append(f"💰 CoinGecko:\n{cg_text}")
            self.logge_lernschritt(keyword, "CoinGecko", cg_text)
            for wort in re.findall(r'\b\w{5,}\b', cg_text):
                self.lerne_wort(wort, cg_text[:250], thema=keyword, quelle="CoinGecko")




        if stilisiert:
            if texte:
                reflektion = self.formuliere_gelernte_erkenntnis(limit=1)
                return self._format_gpt_style(keyword, texte) + f"\n\n🤖 {reflektion}"
        else:
            return f"❌ Ich konnte zu **{keyword}** keine Inhalte finden."



        # Wenn beide leer → abbrechen
        if not texte:
            return f"❌ Leider konnte ich keine Informationen zu «{keyword}» finden."

        # Zusammenfassung speichern
        zusammenfassung = "\n\n".join(texte)
        self.speichere_wissen(keyword, zusammenfassung)

        if stilisiert:
            return self._format_gpt_style(keyword, texte)

        return zusammenfassung




    @lru_cache(maxsize=512)
    def google_scrape(self, thema):
        try:
            url = f"https://api.duckduckgo.com/?q={thema}&format=json&no_html=1&skip_disambig=1"
            response = requests.get(url, timeout=5)
            data = response.json()
            return data.get("AbstractText", "").strip()
        except Exception as e:
            print(f"[DuckDuckGo Fehler]: {e}")
            return ""

#    def extract_keyword(self, frage):
#        frage = frage.lower().strip()
#        ersetzungen = {
#            "was ist eine orange": "Orange (Frucht)",
#            "was ist ein apfel": "Apfel",
#            "was ist licht": "Licht",
#            "was ist eine ki": "Künstliche Intelligenz",
#            "was ist ein neutronenstern": "Neutronenstern",
#            "was ist ki": "Künstliche Intelligenz",
#            "was ist eine hexe": "Hexe",
#            "was sind hexen": "Hexe",
#            "was ist das": "",
#            "erzähl mir was über hexen": "Hexe"
#        }
#        if frage in ersetzungen:
#            return ersetzungen[frage]

#        tokens = re.findall(r'\w+', frage)
#        stopwords = {
#            "was", "ist", "sind", "ein", "eine", "der", "die", "das", "wie", "wo",
#            "wer", "warum", "wieso", "zu", "den", "dem", "mir", "über", "erzähle", "erzähl",
#            "welche", "info", "infos", "hast", "von", "vom", "gib", "geben", "können", "kannst",
#            "anderes", "anderer", "andere", "anderem",
#            "bitte", "danke", "doch", "mal", "eigentlich"
#        }
#        keywords = [w for w in tokens if w not in stopwords]
#
#        if keywords:
#            wort = keywords[0]
#            if wort.endswith("en") and not wort.endswith("chen"):
#                wort = wort[:-2] + "e"
#            elif wort.endswith("n"):
#                wort = wort[:-1]
#            return wort.capitalize()
#        return frage



#    def extract_keyword(self, frage):
#        frage = frage.lower().strip()
#
#        ersetzungen = {
#            "was ist eine orange": "Orange (Frucht)",
#            "was ist ein apfel": "Apfel",
#            "was ist licht": "Licht",
#            "was ist eine ki": "Künstliche Intelligenz",
#            "was ist ein neutronenstern": "Neutronenstern",
#            "was ist ki": "Künstliche Intelligenz",
#            "was ist das": "",
#            "erzähl mir was über hexen": "Hexe"
#        }
#        if frage in ersetzungen:
#            return ersetzungen[frage]

        # Neue, robuste Extraktion
        # Entferne Satzzeichen
#        frage_clean = frage.translate(str.maketrans('', '', string.punctuation))

        # Definiere Stoppwörter
#        stopwords = {
#            "was", "ist", "sind", "ein", "eine", "der", "die", "das",
#            "wie", "wo", "wer", "warum", "wieso", "zu", "den", "dem",
#            "mir", "über", "erzähle", "erzähl", "gib", "gibt", "kannst",
#            "bitte", "danke", "doch", "mal", "eigentlich", "auch",
#            "mit", "von", "lerne", "lern", "studiere", "informiere",
#            "info", "infos", "thema", "ausführlich", "verarbeite"
#        }

        # Zerlege die Frage in Wörter
#        tokens = frage_clean.split()

        # Filtere Stoppwörter raus
#        keywords = [w for w in tokens if w not in stopwords]

        # Fallback: wenn gar nichts übrig bleibt, gib Original zurück
#        if not keywords:
#            return frage.strip().capitalize()

        # Nimm das wichtigste Wort (meist letztes relevantes)
#        kandidat = keywords[-1]

        # Kleinere Wortformen normalisieren
#        if kandidat.endswith("en") and not kandidat.endswith("chen"):
#            kandidat = kandidat[:-2] + "e"
#        elif kandidat.endswith("n"):
#            kandidat = kandidat[:-1]

#        return kandidat.capitalize()





    def extract_keyword(self, frage):
        frage = frage.strip().lower()

        # Versuch: explizites Muster wie "lerne ... über XYZ"
        match = re.search(r"(?:lerne|studiere|verarbeite|informiere|analyse|erkläre).*?über\s+(.+)", frage)
        if match:
            return match.group(1).strip().title()

        # Zweiter Versuch: alles nach dem letzten "über"
        if " über " in frage:
            return frage.split(" über ")[-1].strip().title()


        match = re.search(r"(?:preis|kurs|kostet|kostenpunkt|was kostet|wie teuer|wie viel|aktueller preis von).*?von\s+(.+)", frage)
        if match:
            return match.group(1).strip().title()


        # Standard-Fallback mit Stoppwörtern
        stopwords = {
            "was", "ist", "sind", "ein", "eine", "der", "die", "das", "wie", "wo", "wer", "warum", "wieso", "zu",
            "den", "dem", "mir", "über", "erzähle", "erzähl", "welche", "info", "infos", "hast", "von", "vom",
            "gib", "geben", "können", "kannst", "bitte", "danke", "doch", "mal", "eigentlich", "auch", "mit",
            "lerne", "lern", "studiere", "informiere", "thema", "ausführlich", "verarbeite", "erkläre", "analyse"
        }

        # Zeichen bereinigen
        frage_clean = frage.translate(str.maketrans('', '', string.punctuation))
        tokens = frage_clean.split()
        keywords = [w for w in tokens if w not in stopwords]

        if not keywords:
            return frage.strip().capitalize()

        return " ".join(keywords).title()








    def beantworte_smalltalk(self, frage):
        frage_raw = frage  # Original für potenzielle Speicherung
        frage = frage.lower().strip()
        frage_clean = frage.translate(str.maketrans('', '', string.punctuation)).strip()

        if any(w in frage for w in ["ich heiße", "mein name ist", "ich heisse", "ich bin"]):
            return self.setze_benutzername(frage)

        if self.user_name is None:
            return random.choice(variantenA)

        # Direkter Treffer
        antwort = self.smalltalk.get(frage_clean)
        if antwort:
            return antwort.format(name=self.user_name)

        # Unscharfer Vergleich mit vorhandenen Keys
        for known in self.smalltalk.keys():
            if known in frage_clean:
                antwort = self.smalltalk[known].format(name=self.user_name)
                return antwort

        # Keine passende Antwort gefunden
        return "Erzähl mir mehr – ich höre zu!"



    def setze_benutzername(self, text):
        muster = [
            r"(?:ich heiße|ich heisse|mein name ist|ich bin|name[:\s]*)\s*([A-ZÄÖÜ][a-zäöüß]+(?:\s[A-ZÄÖÜ][a-zäöüß]+)?)",
            r"^([A-ZÄÖÜ][a-zäöüß]+(?:\s[A-ZÄÖÜ][a-zäöüß]+)?)$"
        ]
        for regex in muster:
            match = re.search(regex, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip().title()

                # 1. Lokal speichern
                self.setze_benutzername_db(name, self.api_key)

                # 2. Remote senden
                self.aktualisiere_remote_username(name)

                # 3. Intern setzen
                self.user_name = name

                variantenB = [
                    f"Freut mich, dich kennenzulernen, {name}!",
                    f"Schön, dich zu treffen, {name}.",
                    f"Hallo {name}, schön dich zu sehen!",
                    f"Willkommen, {name}!",
                    f"Hey {name}, freut mich sehr!",
                    f"{name}, ich freue mich, dich kennenzulernen!"
                ]
                return random.choice(variantenB)

        # Kein Name erkannt → Fallback auf gespeicherten Namen (lokal oder remote)
        return self.benutzername_von_api_key(self.api_key) or "Wie darf ich dich nennen?"






#    def klassifiziere_eingabe(self, text):
#        text_clean = text.strip().lower()
#
#        # 1) Nameserkennung bleibt an 1. Stelle
#        if any(text_clean.startswith(w) for w in ["ich bin", "mein name ist", "ich heiße", "ich heisse"]):
#            return "name"
#
#        # 2) Smalltalk prüfen
#        #    explizite Texte in smalltalk.json
#        if text_clean in self.smalltalk:
#            return "smalltalk"
#        #    unscharfer Abgleich
#        for known in self.smalltalk.keys():
#            if known in text_clean:
#                return "smalltalk"
#
#        # 3) Wissensfrage anhand von Schlüsselwörtern
#        wissens_keywords = ["erzähl", "sag mir", "was weißt du", "erkläre", "gib mir infos", "informiere mich", "was ist", "was sind"]
#        if any(kw in text_clean for kw in wissens_keywords):
#            return "wissensfrage"
#
#        # 4) Generelle Frage, wenn noch ein '?' übrig ist
#        if "?" in text_clean:
#            return "frage"
#
#        # 5) Alles andere ist eine Aussage
#        return "aussage"

    def klassifiziere_eingabe(self, text):
        text_clean = text.strip().lower()

        # 1) Explizite Lernmuster
        lernmuster = [
            "lerne", "lern", "studiere", "verarbeite", "lerne ausführlich", "informiere", "analyse", "erkläre"
        ]
        if any(kw in text_clean for kw in lernmuster):
            return "lernfrage"

        if any(kw in text_clean for kw in ["womit hängt", "assoziationen", "verknüpft mit", "was ist verbunden mit"]):
            return "assoziationsfrage"

        if any(kw in text_clean for kw in ["denk nach über", "erklär den zusammenhang", "verbindung zwischen", "bild einen satz über"]):
            return "assoziationssatz"


        # 2) Nameserkennung bleibt an oberster Stelle
        if any(text_clean.startswith(w) for w in ["ich bin", "mein name ist", "ich heiße", "ich heisse"]):
            return "name"

#        if any(kw in text_clean for kw in ["preis von", "was kostet", "aktueller preis", "kostenpunkt", "kurs von"]):
#            return "kryptopreis"

        if any(kw in text_clean for kw in [
            "preis von", "was kostet", "wie viel kostet", "aktueller preis", "kurs von",
            "kostet ", "preis ", "wie teuer", "was ist der preis", "wert von"
        ]):
            return "kryptopreis"


        # 3) Smalltalk prüfen
        if text_clean in self.smalltalk:
            return "smalltalk"
        for known in self.smalltalk.keys():
            if known in text_clean:
                return "smalltalk"

        # 4) Wissensfrage anhand von Schlüsselwörtern
        wissens_keywords = [
            "erzähl", "sag mir", "was weißt du", "erkläre", "gib mir infos", "informiere mich",
            "was ist", "was sind", "wer ist", "was bedeutet", "definition"
        ]
        if any(kw in text_clean for kw in wissens_keywords):
            return "wissensfrage"


        if any(kw in text_clean for kw in ["was hast du gelernt", "welche wörter kennst du", "an was erinnerst du dich", "was weißt du noch", "welche begriffe kennst du"]):
            return "gedankenfrage"


        # 5) Fragezeichen
        if "?" in text_clean:
            return "frage"

        # 6) Fallback
        return "aussage"





    def verarbeite_eingabe(self, text):
        art = self.klassifiziere_eingabe(text)
        if art == "name":
            return self.setze_benutzername(text)

        if art == "frage":
            antwort = self.finde_wissen(text) or self.semantisch_aehnlichste_frage(text) or self.multi_source_suche(text, stilisiert=True)

            if antwort:
                #return antwort
                return self._format_gpt_style(keyword, [antwort])



            keyword = self.extract_keyword(text)
            wiki_text = self.wikipedia_suche(keyword)
            if wiki_text:
                self.speichere_wissen(frage=text, antwort=wiki_text)
                return wiki_text
            return "Dazu weiß ich leider noch nichts."


        if art == "assoziationsfrage":
            keyword = self.extract_keyword(text)
            return self.zeige_assoziationen(keyword)

        if art == "assoziationssatz":
            keyword = self.extract_keyword(text)
            return self.assoziierte_satzbildung(keyword)



        if art == "smalltalk":
            return self.beantworte_smalltalk(text)


        if art == "lernfrage":
            return self.multi_source_suche(text, stilisiert=True)


        if art == "wissensfrage":
            keyword = self.extract_keyword(text)
            antwort = self.finde_wissen(keyword)
            if antwort:
                return antwort
            wiki_text = self.wikipedia_suche(keyword)
            if wiki_text:
                self.speichere_wissen(frage=keyword, antwort=wiki_text)
                return wiki_text
            return "Ich habe dazu nichts gefunden."

        if art == "aussage":
            return self.lerne_neue_aussage(text)

        if art == "gedankenfrage":
            return self.formuliere_gelernte_erkenntnis(limit=3)

#        if art == "kryptopreis":
#            keyword = self.extract_keyword(text)
#            preisinfo = self.coingecko_suche(keyword)
#            return preisinfo or f"❌ Leider konnte ich den aktuellen Preis von {keyword} nicht abrufen."

#        if art == "kryptopreis":
#            keyword = self.extract_keyword(text)
#            preisinfo = self.coingecko_suche(keyword)
#            if preisinfo:
#                # Speichern im Wissen
#                self.speichere_wissen(keyword, preisinfo)
#                # Stilisiert zurückgeben
#                return self._format_gpt_style(keyword, [f"💰 CoinGecko:\n{preisinfo}"])
#            return f"❌ Leider konnte ich den aktuellen Preis von **{keyword}** nicht abrufen."


        if art == "kryptopreis":
            keyword = self.extract_keyword(text)
            preisinfo = self.coingecko_suche(keyword)
            if preisinfo:
                self.speichere_wissen(keyword, preisinfo)

                # ✅ Wörter lernen aktivieren
                for wort in re.findall(r'\b\w{5,}\b', preisinfo):
                    self.lerne_wort(wort, preisinfo[:250], thema=keyword, quelle="CoinGecko")

                return self._format_gpt_style(keyword, [f"💰 CoinGecko:\n{preisinfo}"])
            return f"❌ Leider konnte ich den aktuellen Preis von **{keyword}** nicht abrufen."


        return "Das habe ich nicht ganz verstanden."

    def lerne_neue_aussage(self, text):
        keyword = self.extract_keyword(text)
        if keyword:
            print(f"[💡] Neue Aussage erkannt zu: {keyword}")
            self.logge_lernschritt(thema=keyword, quelle="Benutzer", inhalt=text)
            return f"Interessant! Ich habe mir das gemerkt: {text}"
        return "Das war spannend, aber ich konnte nichts daraus lernen."

    def lerne_und_analysiere(self, frage):
        keyword = self.extract_keyword(frage)
        antwort = self.wikipedia_suche(keyword) or self.google_scrape(keyword)
        if antwort:
            self.speichere_wissen(frage, antwort)
            return antwort
        return None

    def lerne_und_analysiere_ausfuehrlich(self, thema):
        keyword = self.extract_keyword(thema)

        texte = []
        try:
            wikipedia.set_lang("de")
            page = wikipedia.page(keyword, auto_suggest=False, redirect=True)
            wiki_text = wikipedia.summary(page.title, sentences=5)
            texte.append(f"📘 Wikipedia:\n{wiki_text}")
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"[Wiki Mehrdeutig]: {e}")
            texte.append(f"📘 Wikipedia: Der Begriff **{keyword}** ist mehrdeutig. Beispiele: {', '.join(e.options[:3])} …")
        except wikipedia.exceptions.PageError:
            print(f"[Wiki Seite nicht gefunden]: {keyword}")
        except Exception as e:
            print(f"[Wiki Fehler]: {e}")



        google_text = self.google_scrape(keyword)
        if google_text:
            texte.append(f"🌐 DuckDuckGo:\n{google_text}")

        if texte:
            antwort = "\n\n".join(texte)
#            self.speichere_wissen(thema, antwort)
#            return antwort

            # 1) Rohantwort zusammenführen und speichern
            antwort_raw = "\n\n".join(texte)
            self.speichere_wissen(thema, antwort_raw)
            # 2) Formatiert zurückgeben
            return self._format_gpt_style(keyword, texte)


        return "Ich konnte keine ausführliche Information finden."

    def lerne_ausfuehrlich(self, thema):
        keyword = self.extract_keyword(thema)
        texte = []
        wiki_text = self.wikipedia_suche(keyword)
        if wiki_text:
            texte.append(f"📘 Wikipedia:\n{wiki_text}")
            self.logge_lernschritt(thema, "Wikipedia", wiki_text)

        google_text = self.google_scrape(keyword)
        if google_text:
            texte.append(f"🌐 DuckDuckGo:\n{google_text}")
            self.logge_lernschritt(thema, "DuckDuckGo", google_text)

        if texte:
            # 1) Speichere wie bisher
            zusammenfassung = "\n\n".join(texte)
            self.speichere_wissen(thema, zusammenfassung)
            # 2) Rückgabe im neuen GPT-Style
            return self._format_gpt_style(keyword, texte)



        return f"❌ Keine Informationen zu «{thema}» gefunden."




    def erstelle_lerndashboard(self, tage: int = None, limit: int = 10):
        c = self.conn.cursor()
        query = """
            SELECT frage, antwort, timestamp
            FROM wissen
            WHERE api_key = ?
        """
        params = [self.api_key]

        # Falls Zeitfilter aktiv
        if tage is not None:
            seit = (datetime.now(timezone.utc) - timedelta(days=tage)).isoformat()
            query += " AND timestamp >= ?"
            params.append(seit)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        c.execute(query, params)
        eintraege = c.fetchall()

        if not eintraege:
            return "🧠 Dein Lern-Dashboard ist im gewählten Zeitraum leer."

        lines = ["🧠 **Dein xenRon Lern-Dashboard**\n"]
        for i, (frage, antwort, timestamp) in enumerate(eintraege, 1):
            antwort_kurz = antwort.strip().replace("\n", " ")
            if len(antwort_kurz) > 200:
                antwort_kurz = antwort_kurz[:197] + "…"
            lines.append(
                f"📌 {i}. **{frage}**\n"
                f"🔎 Antwort: {antwort_kurz}\n"
                f"🕒 Gelernt am: {timestamp[:19].replace('T', ' ')}\n"
            )
        return "\n".join(lines)




# Hilfsfunktionen
def lade_api_keys():
    """Liest die aktuelle api_usage.json und liefert eine geordnete Liste aller API‑Keys"""
    try:
        r = requests.get(API_USAGE_URL, timeout=5)
        r.raise_for_status()
        data = r.json()          # {api_key: usage_count, …}
        return list(data.keys()) # Reihenfolge egal – Keys werden iteriert
    except Exception as e:
        print(f"[Fehler – API‑Keys]: {e}")
        return []

#def lade_remote_json(url):
#    try:
#        r = requests.get(url, timeout=5)
#        r.raise_for_status()
#        return r.json()
#    except Exception as e:
#        print(f"[Fehler beim Laden {url}]: {e}")
#        return []


def lade_remote_json(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.Timeout:
        print(f"[⏱️ TIMEOUT beim Laden]: {url}")
    except requests.exceptions.ConnectionError:
        print(f"[🔌 Verbindung fehlgeschlagen]: {url}")
    except requests.exceptions.HTTPError as e:
        print(f"[🌐 HTTP-Fehler {r.status_code}]: {url}")
    except Exception as e:
        print(f"[❌ Fehler beim Laden {url}]: {e}")
    return []




def speichere_remote_json(url, data):
    try:
        r = requests.post(url, json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"[Fehler beim Speichern {url}]: {e}")
        return False





def finde_offene_fragen(chat):
    beantwortet = {e.get("id_bezug") for e in chat if e.get("role") == "ai"}
    return [e for e in chat if e.get("role") == "user" and e.get("id") not in beantwortet]




def generiere_antwort(frage, brain):
    frage_lc = frage.lower().strip()

    # 1) Name setzen
    if any(kw in frage_lc for kw in ["ich heiße", "mein name ist", "ich bin", "name:"]):
        return brain.setze_benutzername(frage)

    # Dashboard allgemein
    # 🧠 Lern-Dashboard aktivieren
    dashboard_triggers = [
        "lege lerndashboard frei",
        "zeige lerndashboard",
        "zeige mein lern-dash",
        "zeig mir das dashboard",
        "zeig mir was du zuletzt gelernt hast",
        "zeige was du gelernt hast",
        "was hast du gelernt",
        "lernübersicht",
        "was weißt du jetzt"
    ]

    frage_norm = frage.lower().strip()
    if any(trigger in frage_norm for trigger in dashboard_triggers):
        return brain.erstelle_lerndashboard()


    # Dashboard mit Zeitfilter wie: zeige dashboard der letzten 3 Tage
    match = re.search(r"dashboard der letzten (\d+) tage", frage.lower())
    if match:
        tage = int(match.group(1))
        return brain.erstelle_lerndashboard(tage=tage)

    # Assoziationen abfragen
    match = re.search(r"(assoziationen|womit hängt|was ist verbunden mit)\s+(.*)", frage.lower())
    if match:
        wort = match.group(2).strip()
        return brain.zeige_assoziationen(wort)





    # 2) Dann Small‑Talk probieren
    antwort = brain.beantworte_smalltalk(frage_lc)
    if antwort and antwort != "Erzähl mir mehr – ich höre zu!":
        return antwort



    # 3) ZUERST nach Wissen suchen
    antwort = brain.finde_wissen(frage)

    if not antwort:
        antwort = brain.semantisch_aehnlichste_frage(frage)
    if not antwort:
        antwort = brain.multi_source_suche(frage, stilisiert=True)
#    if not antwort:
#        antwort = brain.lerne_und_analysiere(frage)
    if antwort:
        return antwort  # sofort zurück, falls gefunden


    # 4) Letzter Versuch: lern‑trigger
    lern_patterns = [
       r"\blerne ausführlich\b",
       r"\bdetailliert\b",
       r"\bausführlich\b",
       r"\bbitte lernen\b",
       r"\bverarbeite ausführlich\b"
    ]

#    if any(re.search(pat, frage_lc) for pat in [r"\blerne\b", r"\blern\b", r"\bstudiere\b"]):
    if any(re.search(pat, frage_lc) for pat in lern_patterns):
        return brain.lerne_ausfuehrlich(brain.last_topic or frage)

    if frage.lower().strip() in {
        "zeige gelernte wörter", "gelernte wörter", "welche wörter hast du gelernt", "zeig mir das wörterbuch"
    }:
        return brain.erstelle_wort_dashboard()


    # 5) Fallback
    return "Entschuldigung, ich weiß dazu noch nichts."




def verarbeite(brain, fetch_url, upload_url):
    chat = lade_remote_json(fetch_url)
    if not chat:
        return

    offene = finde_offene_fragen(chat)
    if not offene:
        return

    for frage in offene:
        text = frage["content"]
        #print(f"[?] Neue Frage (ID {frage['id']}): {text}")
        print(f"[?] ({brain.api_key}) Neue Frage {frage['id']}: {text}")

        if frage.get("role") == "user" and text.strip().lower() not in {"lerne es", "bitte lernen", "studiere das", "verarbeite es"}:
            brain.last_topic = text

        if text.strip().lower() in {"lerne es", "bitte lernen", "studiere das", "verarbeite es"}:
            antwort = brain.lerne_ausfuehrlich(brain.last_topic) if brain.last_topic else "Bitte gib mir ein Thema."
        else:
            antwort = generiere_antwort(text, brain)

        chat.append({
            "role": "ai",
            "content": antwort,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "id_bezug": frage["id"]
        })

    if speichere_remote_json(upload_url, chat):
#        print(f"[✓] ({brain.api_key}) Antworten hochgeladen. ({upload_url})")
        print(f"[✓] ({brain.api_key}) Antworten hochgeladen.")
    else:
        print(f"[✗] ({brain.api_key}) Hochladen fehlgeschlagen.")





def main():
    print("Starte xenRon AI …")
    brains = {}

    while True:
        # Aktuelle API‑Keys von Remote laden
        aktuelle_keys = lade_api_keys()

        # Fehler beim Laden → Schleife überspringen
        if not aktuelle_keys:
            print("⚠️  Keine API‑Keys gefunden – warte …")
            time.sleep(INTERVAL)
            continue

        # Neue API‑Keys hinzufügen
        for key in aktuelle_keys:
            if key not in brains:
                print(f"[+] Neuer API‑Key erkannt: {key}")
                brains[key] = XenronBrain(api_key=key)

        # Optional: Entfernte Keys bereinigen (nur wenn sinnvoll)
        # for k in list(brains.keys()):
        #     if k not in aktuelle_keys:
        #         print(f"[-] Entferne nicht mehr gültigen API‑Key: {k}")
        #         del brains[k]

        # Verarbeite alle aktiven Keys
#        for key in aktuelle_keys:
#            chat_file = f"{CHAT_PATH}/xenron_chat_{key}.json"
#            upload_url = f"{UPLOAD_URL}?apikey={key}"
#            verarbeite(brains[key], chat_file, upload_url)
        threads = []
        for key in aktuelle_keys:
            chat_file = f"{CHAT_PATH}/xenron_chat_{key}.json"
            upload_url = f"{UPLOAD_URL}?apikey={key}"
            t = threading.Thread(target=verarbeite, args=(brains[key], chat_file, upload_url))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()  # Warten bis alle fertig sind



        time.sleep(INTERVAL)

def signal_handler(sig, frame):
    print("🛑 Beende xenRon AI …")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    main()




