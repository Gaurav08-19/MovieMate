"""
MovieMate — Gradio Web Interface
Supports real Anthropic API responses (set ANTHROPIC_API_KEY) with a demo fallback.
Run: python demo_app.py
"""

import os, re, sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
import pandas as pd

from src.data_loader import load_movies
from src.preprocessor import preprocess_movies, add_text_representations
from src.embedder import MovieEmbedder
from src.retriever import MovieRetriever

# ── Bootstrap ────────────────────────────────────────────────────────────────
print("=== MovieMate Initializing ===")
df_raw = load_movies()
df = preprocess_movies(df_raw)
df = add_text_representations(df)

embedder = MovieEmbedder()
embeddings = embedder.fit_transform(df["text"].tolist())
retriever = MovieRetriever(df, embedder)
retriever.build_index(embeddings)

TOTAL    = len(df)
AVG_RATE = round(df["rating"].mean(), 1)
YR_MIN   = int(df["year"].min())
YR_MAX   = int(df["year"].max())
all_genres = sorted(set(g.strip() for genres in df["genre"].dropna()
                        for g in str(genres).split("|")))

HAS_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
_chatbot_instance = None


def get_chatbot():
    global _chatbot_instance
    if _chatbot_instance is None:
        from src.chatbot import MovieChatbot
        _chatbot_instance = MovieChatbot(retriever, top_k=5)
    return _chatbot_instance


print(f"=== Ready — {TOTAL} movies | API key: {'✓ Live mode' if HAS_KEY else '✗ Demo mode'} ===\n")

# ── Demo responses ────────────────────────────────────────────────────────────
DEMO_TURNS = [
    (
        "Great picks! Here are top **sci-fi movies after 2010** from the database:\n\n"
        "1. **Interstellar (2014)** ⭐ 8.7 — Nolan's masterpiece on wormhole travel and time dilation. "
        "Matthew McConaughey gives a career-best performance.\n\n"
        "2. **Inception (2010)** ⭐ 8.8 — Mind-bending dream-heist thriller. Leonardo DiCaprio "
        "leads a brilliant ensemble through layers of subconscious.\n\n"
        "3. **Arrival (2016)** ⭐ 7.9 — Denis Villeneuve's meditative first-contact film. "
        "Amy Adams is outstanding as a linguist deciphering alien language.\n\n"
        "4. **Dune (2021)** ⭐ 8.0 — Villeneuve again, adapting Frank Herbert's epic with "
        "breathtaking scale and Timothée Chalamet.\n\n"
        "5. **Blade Runner 2049 (2017)** ⭐ 8.0 — A gorgeous, philosophical sequel with Ryan Gosling "
        "and Harrison Ford. Visually stunning.\n\n"
        "*Want to filter by duration, language, or a specific sub-theme like AI or space exploration?*"
    ),
    (
        "Filtering to **under 2 hours**, here are the best matches:\n\n"
        "1. **Arrival (2016)** ⭐ 7.9 — 116 mins. Perfect runtime for this introspective alien story.\n\n"
        "2. **Edge of Tomorrow (2014)** ⭐ 7.9 — 113 mins. Tom Cruise in a brilliant time-loop action sci-fi.\n\n"
        "3. **Moon (2009)** ⭐ 7.9 — 97 mins. Sam Rockwell alone on the lunar surface. "
        "Quiet, powerful, and deeply unsettling.\n\n"
        "4. **Ex Machina (2015)** ⭐ 7.7 — 108 mins. A tightly crafted AI thriller with Alicia Vikander.\n\n"
        "*Any of these catch your eye? I can tell you more about any of them!*"
    ),
    (
        "Christopher Nolan has a remarkable filmography in our database:\n\n"
        "🎬 **The Dark Knight (2008)** ⭐ 9.0 — His undisputed masterpiece. Heath Ledger's Joker "
        "transformed the superhero genre forever.\n"
        "🎬 **Inception (2010)** ⭐ 8.8 — The dream-heist that redefined the blockbuster.\n"
        "🎬 **Interstellar (2014)** ⭐ 8.7 — Emotionally and scientifically ambitious space opera.\n"
        "🎬 **The Prestige (2006)** ⭐ 8.5 — A duel of magicians with a shocking finale.\n"
        "🎬 **Memento (2000)** ⭐ 8.4 — His most daring structural experiment. Still unmatched.\n"
        "🎬 **The Dark Knight Rises (2012)** ⭐ 8.4 — Epic conclusion to the Batman trilogy.\n"
        "🎬 **Oppenheimer (2023)** ⭐ 8.6 — Cillian Murphy is extraordinary as the father of the bomb.\n\n"
        "**Best overall?** The consensus is *The Dark Knight* — but *Memento* is arguably his most "
        "technically innovative film. Where would you like to start?"
    ),
    (
        "For feel-good, uplifting animated films — these are guaranteed crowd-pleasers:\n\n"
        "1. **Coco (2017)** ⭐ 8.4 — Pixar's love letter to Mexican culture and family bonds. "
        "Gorgeous visuals and an ending that will make you cry.\n\n"
        "2. **Spirited Away (2001)** ⭐ 8.6 — Miyazaki's masterpiece. A young girl's magical "
        "adventure in a spirit world. Suitable for all ages.\n\n"
        "3. **Up (2009)** ⭐ 8.2 — The first 10 minutes will break your heart; the rest will mend it. "
        "An unforgettable adventure.\n\n"
        "4. **Inside Out (2015)** ⭐ 8.1 — Pixar's most emotionally intelligent film. "
        "A brilliant exploration of growing up and mental health.\n\n"
        "5. **My Neighbor Totoro (1988)** ⭐ 8.2 — Gentle, magical, and utterly wholesome Miyazaki. "
        "Perfect for children and adults alike.\n\n"
        "*All rated 8.1+ and guaranteed to leave you smiling! 🎉*"
    ),
]

_demo_turn = {"n": 0}


# ── Helpers ───────────────────────────────────────────────────────────────────
GENRE_COLORS = {
    "Drama":     ("#4f46e5", "#eef2ff"),
    "Action":    ("#dc2626", "#fef2f2"),
    "Comedy":    ("#d97706", "#fffbeb"),
    "Sci-Fi":    ("#0284c7", "#f0f9ff"),
    "Crime":     ("#7c3aed", "#f5f3ff"),
    "Thriller":  ("#be185d", "#fdf2f8"),
    "Animation": ("#059669", "#ecfdf5"),
    "Horror":    ("#92400e", "#fef3c7"),
    "Adventure": ("#0891b2", "#ecfeff"),
    "Romance":   ("#e11d48", "#fff1f2"),
    "Biography": ("#374151", "#f9fafb"),
    "Mystery":   ("#6d28d9", "#faf5ff"),
    "War":       ("#b45309", "#fffbeb"),
    "History":   ("#047857", "#ecfdf5"),
    "Western":   ("#92400e", "#fef3c7"),
    "Fantasy":   ("#7c3aed", "#faf5ff"),
    "Music":     ("#db2777", "#fdf4ff"),
    "Sport":     ("#15803d", "#f0fdf4"),
    "Family":    ("#0369a1", "#f0f9ff"),
}

def genre_badge(genre_str):
    tags = []
    for g in str(genre_str).split("|")[:3]:
        g = g.strip()
        fg, bg = GENRE_COLORS.get(g, ("#374151", "#f3f4f6"))
        tags.append(
            f'<span style="background:{bg};color:{fg};padding:3px 10px;border-radius:20px;'
            f'font-size:11px;font-weight:600;margin:2px 2px 2px 0;display:inline-block;'
            f'border:1px solid {fg}22;">{g}</span>'
        )
    return "".join(tags)


def star_html(r):
    filled = int(r / 2)
    half   = 1 if (r / 2 - filled) >= 0.4 else 0
    empty  = 5 - filled - half
    stars  = '<span style="color:#f59e0b;font-size:13px;">' + "★" * filled
    if half:
        stars += "½"
    stars += '<span style="color:#d1d5db;">' + "☆" * empty + "</span></span>"
    return stars


def movie_card_html(m, rank=None):
    genre_str = str(m.get("genre", ""))
    first_genre = genre_str.split("|")[0].strip()
    accent, _ = GENRE_COLORS.get(first_genre, ("#6366f1", "#f0f9ff"))

    score_bar = ""
    if "similarity_score" in m:
        pct = int(m["similarity_score"] * 100)
        score_bar = f"""
        <div style="margin-top:10px;display:flex;align-items:center;gap:8px;">
          <span style="font-size:11px;color:#9ca3af;white-space:nowrap;">Relevance</span>
          <div style="flex:1;background:#f3f4f6;border-radius:4px;height:5px;">
            <div style="width:{pct}%;height:5px;background:{accent};border-radius:4px;
                        transition:width 0.6s ease;"></div>
          </div>
          <span style="font-size:11px;font-weight:600;color:{accent};">{pct}%</span>
        </div>"""

    rank_badge = ""
    if rank:
        rank_badge = (
            f'<div style="background:{accent};color:white;border-radius:50%;'
            f'width:24px;height:24px;display:inline-flex;align-items:center;'
            f'justify-content:center;font-size:12px;font-weight:700;'
            f'margin-right:8px;flex-shrink:0;">#{rank}</div>'
        )

    plot = str(m.get("plot_summary", ""))
    plot_short = plot[:120] + ("…" if len(plot) > 120 else "")

    cast = str(m.get("cast", "")).split(",")
    cast_short = ", ".join(c.strip() for c in cast[:2]) if cast else ""

    return f"""
<div style="background:white;border:1px solid #f0f0f0;border-radius:14px;padding:14px 16px;
            margin-bottom:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);
            border-left:4px solid {accent};transition:box-shadow 0.2s;">
  <div style="display:flex;align-items:flex-start;gap:4px;">
    {rank_badge}
    <div style="flex:1;min-width:0;">
      <div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;flex-wrap:wrap;">
        <span style="font-size:14px;font-weight:700;color:#111827;">{m['title']}</span>
        <span style="font-size:12px;color:#6b7280;white-space:nowrap;">{int(m['year'])}</span>
      </div>
      <div style="display:flex;align-items:center;gap:10px;margin:5px 0;flex-wrap:wrap;">
        <div style="display:flex;align-items:center;gap:4px;">
          {star_html(m['rating'])}
          <span style="font-size:12px;font-weight:600;color:#374151;">{m['rating']}</span>
        </div>
        <span style="color:#d1d5db;">·</span>
        <span style="font-size:12px;color:#6b7280;">🎬 {m.get('director','')}</span>
        <span style="color:#d1d5db;">·</span>
        <span style="font-size:12px;color:#6b7280;">⏱ {m.get('duration_mins',0)} min</span>
      </div>
      <div style="margin:5px 0;">{genre_badge(genre_str)}</div>
      {f'<div style="font-size:11px;color:#9ca3af;margin:4px 0;">👥 {cast_short}</div>' if cast_short else ''}
      <div style="font-size:12px;color:#4b5563;margin-top:6px;line-height:1.6;">{plot_short}</div>
      {score_bar}
    </div>
  </div>
</div>"""


def retrieved_html(movies):
    if not movies:
        return """
        <div style="text-align:center;padding:32px 16px;color:#9ca3af;">
          <div style="font-size:2em;margin-bottom:8px;">🎞️</div>
          <div style="font-size:13px;">Ask me something to see the movies<br>retrieved by FAISS</div>
        </div>"""
    header = f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;padding:8px 12px;
                background:linear-gradient(135deg,#667eea22,#764ba222);
                border-radius:10px;border:1px solid #e0e7ff;">
      <span style="font-size:16px;">🔍</span>
      <span style="font-size:12px;font-weight:600;color:#4338ca;">
        Top {len(movies)} semantic matches
      </span>
      <span style="margin-left:auto;font-size:11px;color:#818cf8;">FAISS · cosine similarity</span>
    </div>"""
    cards = "".join(movie_card_html(m, i + 1) for i, m in enumerate(movies))
    return header + cards


def stats_html():
    genre_counts = Counter(
        g.strip() for genres in df["genre"].dropna()
        for g in str(genres).split("|")
    )
    top3 = ", ".join(f"{g} ({c})" for g, c in genre_counts.most_common(3))
    if HAS_KEY:
        mode_html = '<span style="display:inline-flex;align-items:center;gap:5px;background:#dcfce7;color:#15803d;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;">🟢 Live — Claude API</span>'
    else:
        mode_html = '<span style="display:inline-flex;align-items:center;gap:5px;background:#fef9c3;color:#854d0e;padding:3px 10px;border-radius:20px;font-size:11px;font-weight:600;">🟡 Demo mode</span>'

    return f"""
<div style="display:flex;gap:10px;flex-wrap:wrap;margin:12px 0;padding:0 4px;">
  <div style="background:linear-gradient(135deg,#ecfdf5,#d1fae5);border-radius:12px;
              padding:10px 18px;text-align:center;min-width:88px;border:1px solid #a7f3d0;">
    <div style="font-size:26px;font-weight:800;color:#065f46;line-height:1;">{TOTAL}</div>
    <div style="font-size:10px;font-weight:600;color:#047857;text-transform:uppercase;
                letter-spacing:0.5px;margin-top:2px;">Movies</div>
  </div>
  <div style="background:linear-gradient(135deg,#eff6ff,#dbeafe);border-radius:12px;
              padding:10px 18px;text-align:center;min-width:88px;border:1px solid #bfdbfe;">
    <div style="font-size:26px;font-weight:800;color:#1e40af;line-height:1;">{AVG_RATE}</div>
    <div style="font-size:10px;font-weight:600;color:#1d4ed8;text-transform:uppercase;
                letter-spacing:0.5px;margin-top:2px;">Avg Rating</div>
  </div>
  <div style="background:linear-gradient(135deg,#fff7ed,#ffedd5);border-radius:12px;
              padding:10px 18px;text-align:center;min-width:88px;border:1px solid #fed7aa;">
    <div style="font-size:26px;font-weight:800;color:#c2410c;line-height:1;">{YR_MIN}–{YR_MAX}</div>
    <div style="font-size:10px;font-weight:600;color:#ea580c;text-transform:uppercase;
                letter-spacing:0.5px;margin-top:2px;">Year Range</div>
  </div>
  <div style="background:linear-gradient(135deg,#faf5ff,#f3e8ff);border-radius:12px;
              padding:10px 18px;text-align:center;min-width:88px;border:1px solid #e9d5ff;">
    <div style="font-size:26px;font-weight:800;color:#6b21a8;line-height:1;">{len(all_genres)}</div>
    <div style="font-size:10px;font-weight:600;color:#7e22ce;text-transform:uppercase;
                letter-spacing:0.5px;margin-top:2px;">Genres</div>
  </div>
  <div style="background:linear-gradient(135deg,#fdf2f8,#fce7f3);border-radius:12px;
              padding:10px 16px;flex:1;min-width:200px;border:1px solid #fbcfe8;">
    <div style="font-size:10px;font-weight:600;color:#9d174d;text-transform:uppercase;
                letter-spacing:0.5px;margin-bottom:4px;">Top Genres</div>
    <div style="font-size:12px;color:#374151;">{top3}</div>
    <div style="margin-top:6px;">{mode_html}</div>
  </div>
</div>"""


# ── Gradio callbacks ──────────────────────────────────────────────────────────
_retrieved_state = []


def respond(user_message: str, history: list):
    global _retrieved_state
    if not user_message.strip():
        return "", history, retrieved_html([])

    if HAS_KEY:
        try:
            bot = get_chatbot()
            response, retrieved = bot.chat(user_message)
            _retrieved_state = retrieved
        except Exception as e:
            response = f"⚠️ API error: {e}"
            retrieved = retriever.search(user_message, k=5)
            _retrieved_state = retrieved
    else:
        idx = _demo_turn["n"] % len(DEMO_TURNS)
        response = DEMO_TURNS[idx]
        _demo_turn["n"] += 1
        _retrieved_state = retriever.search(user_message, k=5)

    history.append((user_message, response))
    return "", history, retrieved_html(_retrieved_state)


def reset_chat():
    global _retrieved_state
    _demo_turn["n"] = 0
    _retrieved_state = []
    if _chatbot_instance is not None:
        _chatbot_instance.reset()
    return [], retrieved_html([])


def _parse_year_intent(query: str):
    """Extract year constraints mentioned in the query text.

    Returns (clean_query, min_year_override, max_year_override) or None overrides.
    Handles patterns like:
      "movies from 2021", "after 2020", "since 2019", "before 2000",
      "in the 90s", "1990s movies", "2020s films"
    """
    q = query
    min_yr = max_yr = None

    # "from / after / since YYYY" → min year
    m = re.search(r'\b(?:from|after|since|post[- ]?)\s*(\d{4})\b', q, re.I)
    if m:
        yr = int(m.group(1))
        if 1900 <= yr <= 2030:
            min_yr = yr
            q = q[:m.start()].strip() + " " + q[m.end():].strip()

    # "before / until / pre- YYYY" → max year
    m2 = re.search(r'\b(?:before|until|pre[- ]?|up to)\s*(\d{4})\b', q, re.I)
    if m2:
        yr = int(m2.group(1))
        if 1900 <= yr <= 2030:
            max_yr = yr
            q = q[:m2.start()].strip() + " " + q[m2.end():].strip()

    # "in YYYY" or "of YYYY" (only if standalone year like "2021 movies")
    if min_yr is None:
        m3 = re.search(r'(?:^|\s)(\d{4})(?:\s|$)', q)
        if m3:
            yr = int(m3.group(1))
            if 1900 <= yr <= 2030:
                min_yr = yr
                max_yr = yr
                q = q[:m3.start()].strip() + " " + q[m3.end():].strip()

    # "90s", "1990s", "2000s" decade patterns
    m4 = re.search(r'\b(?:the\s+)?(\d{2}|\d{4})s\b', q, re.I)
    if m4 and min_yr is None:
        raw = m4.group(1)
        decade = int(raw) * (10 if len(raw) == 3 else 1)
        if len(raw) == 2:  # "90s" → 1990
            decade = 1900 + int(raw) if int(raw) >= 20 else 2000 + int(raw)
        if 1900 <= decade <= 2030:
            min_yr = decade
            max_yr = decade + 9
            q = re.sub(r'\b(?:the\s+)?' + re.escape(raw) + r's\b', '', q, flags=re.I)

    return q.strip(), min_yr, max_yr


def do_search(query: str, genre: str, min_rating: float, min_year: int, max_year: int):
    # Parse year intent from the query text itself
    clean_query, extracted_min, extracted_max = _parse_year_intent(query)

    # Use extracted years only if the slider is still at its default value
    effective_min_year = extracted_min if extracted_min and int(min_year) == YR_MIN else int(min_year)
    effective_max_year = extracted_max if extracted_max and int(max_year) == YR_MAX else int(max_year)

    if not clean_query.strip() and genre == "All" and effective_min_year == YR_MIN and effective_max_year == YR_MAX:
        return """
        <div style="text-align:center;padding:40px 16px;color:#9ca3af;">
          <div style="font-size:2.5em;margin-bottom:10px;">🔍</div>
          <div style="font-size:14px;">Enter a query or pick a genre to search</div>
          <div style="font-size:12px;margin-top:6px;color:#d1d5db;">
            Try: "psychological thriller with an unreliable narrator"
          </div>
        </div>"""
    results = retriever.search_by_filter(
        query=clean_query or query,
        genre=None if genre == "All" else genre,
        min_rating=float(min_rating),
        min_year=effective_min_year,
        max_year=effective_max_year,
        k=12,
    )
    if not results:
        return """
        <div style="text-align:center;padding:32px 16px;color:#ef4444;">
          <div style="font-size:2em;margin-bottom:8px;">😕</div>
          <div style="font-size:14px;font-weight:600;">No movies matched your filters</div>
          <div style="font-size:12px;color:#9ca3af;margin-top:6px;">Try relaxing the rating or year range</div>
        </div>"""

    genre_note = f' in <strong>{genre}</strong>' if genre != "All" else ""
    year_note = f"{effective_min_year}–{effective_max_year}"
    header = f"""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;padding:10px 14px;
                background:linear-gradient(135deg,#f0fdf4,#dcfce7);
                border-radius:10px;border:1px solid #bbf7d0;">
      <span style="font-size:16px;">✅</span>
      <span style="font-size:13px;color:#166534;">
        Found <strong>{len(results)}</strong> movies{genre_note} &nbsp;·&nbsp;
        Rated ≥ {min_rating} &nbsp;·&nbsp; {year_note}
      </span>
    </div>"""
    cards = "".join(movie_card_html(m, i + 1) for i, m in enumerate(results))
    return header + cards


def dataset_table(genre: str, min_rating: float, min_year: int):
    filt = df.copy()
    if genre != "All":
        filt = filt[filt["genre"].str.contains(genre, case=False, na=False)]
    filt = filt[(filt["rating"] >= min_rating) & (filt["year"] >= min_year)]
    filt = filt[["title", "year", "rating", "genre", "director", "duration_mins"]].sort_values(
        "rating", ascending=False
    )
    rows = [
        f"| **{r.title}** | {int(r.year)} | ⭐ {r.rating} | {str(r.genre)[:28]} | {r.director} | {r.duration_mins} min |"
        for r in filt.head(40).itertuples()
    ]
    if not rows:
        return "No movies match the current filters."
    header = f"**{len(filt)} movies** match | showing top 40\n\n"
    table  = "| Title | Year | Rating | Genre | Director | Duration |\n"
    table += "|-------|------|--------|-------|----------|----------|\n"
    return header + table + "\n".join(rows)


# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* Base */
body, .gradio-container {
  font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif !important;
  background: #f8fafc !important;
}
footer { display: none !important; }

/* Tabs */
.tabs > .tab-nav {
  border-bottom: 2px solid #e5e7eb !important;
  background: transparent !important;
  gap: 4px !important;
}
.tabs > .tab-nav > button {
  border-radius: 8px 8px 0 0 !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 10px 18px !important;
  color: #6b7280 !important;
  border: none !important;
  background: transparent !important;
  transition: all 0.2s !important;
}
.tabs > .tab-nav > button:hover {
  color: #4338ca !important;
  background: #eef2ff !important;
}
.tabs > .tab-nav > button.selected {
  color: #4338ca !important;
  background: #eef2ff !important;
  border-bottom: 2px solid #4338ca !important;
}

/* Buttons */
button.primary {
  background: linear-gradient(135deg, #667eea, #764ba2) !important;
  border: none !important;
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 4px 12px rgba(102,126,234,0.35) !important;
  transition: all 0.2s !important;
}
button.primary:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 18px rgba(102,126,234,0.45) !important;
}
button.secondary {
  background: white !important;
  border: 1.5px solid #e5e7eb !important;
  border-radius: 8px !important;
  color: #374151 !important;
  font-size: 12px !important;
  transition: all 0.2s !important;
}
button.secondary:hover {
  border-color: #667eea !important;
  color: #4338ca !important;
  background: #fafafe !important;
}

/* Example pill buttons */
.gr-button-sm {
  border-radius: 20px !important;
  font-size: 12px !important;
  padding: 6px 14px !important;
  background: white !important;
  border: 1.5px solid #e0e7ff !important;
  color: #4338ca !important;
  font-weight: 500 !important;
  transition: all 0.2s !important;
  white-space: nowrap !important;
}
.gr-button-sm:hover {
  background: #eef2ff !important;
  border-color: #818cf8 !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 10px rgba(102,126,234,0.2) !important;
}

/* Chat bubbles */
.chatbot .message.user {
  background: linear-gradient(135deg, #667eea, #764ba2) !important;
  color: white !important;
  border-radius: 18px 18px 4px 18px !important;
  font-size: 14px !important;
}
.chatbot .message.bot {
  background: white !important;
  border: 1px solid #f0f0f0 !important;
  border-radius: 18px 18px 18px 4px !important;
  font-size: 14px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}
.chatbot {
  border-radius: 14px !important;
  border: 1.5px solid #e5e7eb !important;
  background: #f8fafc !important;
}

/* Input */
.gr-text-input, textarea, input[type="text"] {
  border-radius: 10px !important;
  border: 1.5px solid #e5e7eb !important;
  font-size: 14px !important;
  transition: border-color 0.2s !important;
}
.gr-text-input:focus, textarea:focus {
  border-color: #667eea !important;
  box-shadow: 0 0 0 3px rgba(102,126,234,0.12) !important;
}

/* Sliders */
input[type="range"] {
  accent-color: #667eea !important;
}

/* Scrollbars */
#retrieved-panel, #search-results {
  max-height: 640px;
  overflow-y: auto;
  padding-right: 4px;
  scrollbar-width: thin;
  scrollbar-color: #c7d2fe transparent;
}
#retrieved-panel::-webkit-scrollbar,
#search-results::-webkit-scrollbar { width: 5px; }
#retrieved-panel::-webkit-scrollbar-thumb,
#search-results::-webkit-scrollbar-thumb {
  background: #c7d2fe;
  border-radius: 3px;
}

/* Dropdown */
select, .gr-dropdown {
  border-radius: 10px !important;
  border: 1.5px solid #e5e7eb !important;
  font-size: 13px !important;
}

/* Main container padding */
.contain { padding: 0 8px !important; }
"""

EXAMPLES = [
    "Suggest sci-fi movies similar to Interstellar",
    "Movies directed by Christopher Nolan",
    "Feel-good animated movies for the family",
    "Best crime thrillers with unexpected plot twists",
    "Classic westerns with great storytelling",
    "Horror movies that are genuinely scary",
    "Drama movies based on true stories",
    "Movies with the highest IMDb ratings",
]

HOW_IT_WORKS_HTML = """
<div style="margin-top:16px;padding:14px;background:linear-gradient(135deg,#fafafe,#f5f3ff);
            border-radius:12px;border:1px solid #e0e7ff;">
  <div style="font-size:12px;font-weight:700;color:#4338ca;text-transform:uppercase;
              letter-spacing:0.5px;margin-bottom:10px;">How it works</div>
  <div style="display:flex;flex-direction:column;gap:6px;">
    <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#374151;">
      <span style="background:#667eea;color:white;border-radius:50%;width:20px;height:20px;
                   display:inline-flex;align-items:center;justify-content:center;
                   font-size:10px;font-weight:700;flex-shrink:0;">1</span>
      Your query is embedded into a vector
    </div>
    <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#374151;">
      <span style="background:#667eea;color:white;border-radius:50%;width:20px;height:20px;
                   display:inline-flex;align-items:center;justify-content:center;
                   font-size:10px;font-weight:700;flex-shrink:0;">2</span>
      <strong>FAISS</strong> finds the closest movie vectors
    </div>
    <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#374151;">
      <span style="background:#667eea;color:white;border-radius:50%;width:20px;height:20px;
                   display:inline-flex;align-items:center;justify-content:center;
                   font-size:10px;font-weight:700;flex-shrink:0;">3</span>
      Top matches are injected as context
    </div>
    <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#374151;">
      <span style="background:#764ba2;color:white;border-radius:50%;width:20px;height:20px;
                   display:inline-flex;align-items:center;justify-content:center;
                   font-size:10px;font-weight:700;flex-shrink:0;">4</span>
      <strong>Claude</strong> generates the response
    </div>
  </div>
</div>"""

# ── UI Layout ─────────────────────────────────────────────────────────────────
with gr.Blocks(css=CSS, title="🎬 MovieMate") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
                border-radius:16px;padding:28px 24px 20px;margin-bottom:16px;
                box-shadow:0 8px 32px rgba(0,0,0,0.2);">
      <div style="text-align:center;">
        <div style="font-size:3em;line-height:1;margin-bottom:6px;">🎬</div>
        <h1 style="font-size:2.6em;font-weight:900;color:white;margin:0;letter-spacing:-0.5px;
                   background:linear-gradient(135deg,#ffffff,#c7d2fe);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
          MovieMate
        </h1>
        <p style="color:#94a3b8;margin:8px 0 0;font-size:15px;font-weight:400;">
          Conversational AI for Intelligent Movie Search &amp; Recommendations
        </p>
      </div>
    </div>""")

    # ── Stats bar ─────────────────────────────────────────────────────────────
    gr.HTML(stats_html())

    with gr.Tabs():

        # ── Tab 1: Chat ───────────────────────────────────────────────────────
        with gr.TabItem("💬 Chat"):
            with gr.Row(equal_height=False):

                with gr.Column(scale=3):
                    chatbot_ui = gr.Chatbot(
                        label="MovieMate",
                        show_label=False,
                        elem_classes=["chatbot"],
                        height=460,
                        bubble_full_width=False,
                    )
                    with gr.Row():
                        msg_box = gr.Textbox(
                            placeholder='Ask me anything… "What should I watch tonight?"',
                            label="",
                            lines=2,
                            scale=5,
                        )
                        send_btn = gr.Button("Send ➤", variant="primary", scale=1, min_width=90)
                    with gr.Row():
                        reset_btn = gr.Button("🔄 New Conversation", size="sm")
                        if not HAS_KEY:
                            gr.Markdown(
                                "⚠️ *Demo mode — set `ANTHROPIC_API_KEY` for live responses*",
                                elem_id="mode-note",
                            )

                with gr.Column(scale=2):
                    gr.HTML("""
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
                      <span style="font-size:18px;">🗂️</span>
                      <span style="font-size:15px;font-weight:700;color:#111827;">Retrieved Context</span>
                    </div>""")
                    retrieved_panel = gr.HTML(
                        value="""
                        <div style="text-align:center;padding:32px 16px;color:#9ca3af;">
                          <div style="font-size:2em;margin-bottom:8px;">🎞️</div>
                          <div style="font-size:13px;">Ask me something to see the movies<br>retrieved by FAISS</div>
                        </div>""",
                        elem_id="retrieved-panel",
                    )
                    gr.HTML(HOW_IT_WORKS_HTML)

            gr.HTML("""
            <div style="margin:16px 0 8px;display:flex;align-items:center;gap:8px;">
              <span style="font-size:14px;">💡</span>
              <span style="font-size:13px;font-weight:600;color:#374151;">Try one of these</span>
            </div>""")
            with gr.Row():
                for ex in EXAMPLES[:4]:
                    gr.Button(ex, size="sm").click(fn=lambda x=ex: x, outputs=msg_box)
            with gr.Row():
                for ex in EXAMPLES[4:]:
                    gr.Button(ex, size="sm").click(fn=lambda x=ex: x, outputs=msg_box)

        # ── Tab 2: Semantic Search ────────────────────────────────────────────
        with gr.TabItem("🔍 Semantic Search"):
            gr.HTML("""
            <div style="margin:16px 0 20px;">
              <h3 style="font-size:16px;font-weight:700;color:#111827;margin:0 0 4px;">
                Search by natural language + metadata filters
              </h3>
              <p style="font-size:13px;color:#6b7280;margin:0;">
                Describe what you're looking for — the RAG engine finds the best matches semantically.
              </p>
            </div>""")
            with gr.Row():
                search_q = gr.Textbox(
                    placeholder="e.g. 'psychological thriller with an unreliable narrator'",
                    label="What are you looking for?",
                    scale=4,
                )
                search_btn = gr.Button("🔍  Search", variant="primary", scale=1, min_width=110)
            with gr.Row():
                genre_dd = gr.Dropdown(
                    choices=["All"] + all_genres, value="All", label="Genre", scale=1
                )
                min_rat  = gr.Slider(1.0, 9.5, value=7.0, step=0.1, label="Min Rating", scale=2)
                min_yr   = gr.Slider(YR_MIN, YR_MAX, value=1970, step=1, label="From Year", scale=2)
                max_yr   = gr.Slider(YR_MIN, YR_MAX, value=YR_MAX, step=1, label="To Year", scale=2)

            search_results = gr.HTML(
                value="""
                <div style="text-align:center;padding:40px 16px;color:#9ca3af;">
                  <div style="font-size:2.5em;margin-bottom:10px;">🔍</div>
                  <div style="font-size:14px;">Enter a query or pick a genre to search</div>
                  <div style="font-size:12px;margin-top:6px;color:#d1d5db;">
                    Try: "psychological thriller with an unreliable narrator"
                  </div>
                </div>""",
                elem_id="search-results",
            )
            search_btn.click(do_search, [search_q, genre_dd, min_rat, min_yr, max_yr], search_results)
            search_q.submit(do_search, [search_q, genre_dd, min_rat, min_yr, max_yr], search_results)

        # ── Tab 3: Dataset Explorer ───────────────────────────────────────────
        with gr.TabItem("📊 Dataset Explorer"):
            gr.HTML(f"""
            <div style="margin:16px 0 20px;">
              <h3 style="font-size:16px;font-weight:700;color:#111827;margin:0 0 4px;">
                Browse all {TOTAL} movies
              </h3>
              <p style="font-size:13px;color:#6b7280;margin:0;">
                Filter by genre, minimum rating, and release year. Sorted by IMDb rating.
              </p>
            </div>""")
            with gr.Row():
                ds_genre  = gr.Dropdown(["All"] + all_genres, value="All", label="Genre", scale=1)
                ds_rating = gr.Slider(1.0, 9.5, value=7.0, step=0.1, label="Min Rating", scale=2)
                ds_year   = gr.Slider(YR_MIN, YR_MAX, value=YR_MIN, step=1, label="From Year", scale=2)

            ds_out = gr.Markdown(value=dataset_table("All", 7.0, YR_MIN))
            for comp in [ds_genre, ds_rating, ds_year]:
                comp.change(dataset_table, [ds_genre, ds_rating, ds_year], ds_out)

        # ── Tab 4: About ──────────────────────────────────────────────────────
        with gr.TabItem("ℹ️ About"):
            gr.HTML(f"""
            <div style="max-width:720px;margin:20px auto;padding:0 8px;">

              <h2 style="font-size:1.6em;font-weight:800;color:#111827;margin-bottom:6px;">
                About MovieMate
              </h2>
              <p style="color:#6b7280;font-size:14px;line-height:1.7;margin-bottom:24px;">
                MovieMate is a <strong>RAG (Retrieval-Augmented Generation)</strong> conversational
                AI system that combines semantic vector search with a large language model to deliver
                intelligent, context-aware movie recommendations.
              </p>

              <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:14px;
                          padding:20px 24px;margin-bottom:24px;">
                <div style="font-size:12px;font-weight:700;color:#94a3b8;text-transform:uppercase;
                            letter-spacing:1px;margin-bottom:14px;">RAG Pipeline</div>
                <div style="display:flex;flex-direction:column;gap:0;">
                  {''.join([
                    f'<div style="display:flex;align-items:center;gap:12px;padding:8px 0;">'
                    f'<div style="background:{c};border-radius:8px;padding:6px 12px;'
                    f'font-size:12px;font-weight:600;color:white;white-space:nowrap;min-width:52px;text-align:center;">{step}</div>'
                    f'<div style="flex:1;height:1px;background:linear-gradient({c},transparent);"></div>'
                    f'<div style="font-size:13px;color:#e2e8f0;">{label}</div>'
                    f'</div>'
                    for step, c, label in [
                      ("Query","#667eea","User's natural language input"),
                      ("Embed","#7c3aed","sentence-transformers/all-MiniLM-L6-v2 → 384-dim vector"),
                      ("FAISS","#0284c7","IndexFlatIP cosine similarity search"),
                      ("Context","#059669","Top-5 matching movies retrieved"),
                      ("Claude","#764ba2","claude-sonnet-4-6 generates the response"),
                    ]
                  ])}
                </div>
              </div>

              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:24px;">
                <div style="background:white;border:1px solid #f0f0f0;border-radius:12px;
                            padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                  <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                              letter-spacing:0.5px;margin-bottom:8px;">Dataset</div>
                  <div style="font-size:13px;color:#374151;line-height:1.8;">
                    📽️ {TOTAL} curated films<br>
                    ⭐ Avg rating {AVG_RATE}/10<br>
                    📅 {YR_MIN}–{YR_MAX}<br>
                    🎭 {len(all_genres)} genres<br>
                    🌍 International cinema
                  </div>
                </div>
                <div style="background:white;border:1px solid #f0f0f0;border-radius:12px;
                            padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.04);">
                  <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                              letter-spacing:0.5px;margin-bottom:8px;">Tech Stack</div>
                  <div style="font-size:13px;color:#374151;line-height:1.8;">
                    🤗 sentence-transformers<br>
                    ⚡ FAISS vector index<br>
                    🤖 Claude (Anthropic API)<br>
                    🎨 Gradio interface<br>
                    🐼 pandas data pipeline
                  </div>
                </div>
              </div>

              <div style="background:{'#f0fdf4' if HAS_KEY else '#fefce8'};
                          border:1px solid {'#bbf7d0' if HAS_KEY else '#fde68a'};
                          border-radius:12px;padding:14px 18px;margin-bottom:20px;">
                <div style="font-size:13px;color:{'#166534' if HAS_KEY else '#92400e'};">
                  {'✅ <strong>Live mode</strong> — responses generated by Claude API' if HAS_KEY else
                   '⚠️ <strong>Demo mode</strong> — set <code>ANTHROPIC_API_KEY</code> for live Claude responses'}
                </div>
                {'<div style="font-size:12px;color:#6b7280;margin-top:8px;font-family:monospace;background:#f3f4f6;padding:8px 12px;border-radius:8px;">export ANTHROPIC_API_KEY="sk-ant-..."<br>python demo_app.py</div>' if not HAS_KEY else ''}
              </div>

            </div>""")

    # Wire events
    send_btn.click(respond, [msg_box, chatbot_ui], [msg_box, chatbot_ui, retrieved_panel])
    msg_box.submit(respond, [msg_box, chatbot_ui], [msg_box, chatbot_ui, retrieved_panel])
    reset_btn.click(reset_chat, outputs=[chatbot_ui, retrieved_panel])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
