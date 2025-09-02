
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Batch pipeline for AI/NLP analysis of Google Form free-text feedback.

Example:
    python scripts/pipeline.py --csv_path data.csv --out_dir outputs --use_openai
"""
import os, argparse, re, json, textwrap, random
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

# NLP
from pythainlp import word_tokenize
from pythainlp.corpus.common import thai_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import MinMaxScaler

# Topic modeling
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
import hdbscan

# Viz
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Optional LLM
USE_OPENAI = False
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ------------------- Config -------------------
COLS = {
    "timestamp": "timestamp",
    "team": "team",
    "feedback": "feedback",
}
TH_STOPWORDS = set(thai_stopwords())

def clean_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"\s+", " ", s.strip())
    # remove zero-width and control chars
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    return s

def tokenize_th(s: str) -> str:
    # ใช้ newmm เพื่อความแม่นยำภาษาไทย
    toks = word_tokenize(s, engine="newmm")
    toks = [t for t in toks if t.strip() and t not in TH_STOPWORDS]
    return " ".join(toks)

def quick_polarity_heuristic(s: str) -> float:
    """
    คะแนน sentiment แบบคร่าวๆเพื่อความเร็ว:
    +1 บวก, -1 ลบ. ปรับปรุงได้ตาม lexicon ที่องค์กรใช้
    """
    pos_kw = ["ดีมาก","ดี","ชอบ","ชัดเจน","มีประโยชน์","ยอดเยี่ยม","ประทับใจ","สะดวก","เหมาะสม","สนุก"]
    neg_kw = ["ช้า","แย่","งง","ไม่เข้าใจ","ไม่สะดวก","ยาก","น้อยไป","เสียงดัง","เบื่อ","ไม่พอ","นานเกินไป"]
    score = 0
    for w in pos_kw:
        if w in s: score += 1
    for w in neg_kw:
        if w in s: score -= 1
    return float(np.tanh(score / 3.0))  # bound -1..1

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def build_figures_dir(out_dir):
    fig_dir = Path(out_dir)/"figures"
    ensure_dir(fig_dir)
    return fig_dir

def plot_sentiment_bar(df, out_path):
    plt.figure()
    ax = df["sentiment_bucket"].value_counts().reindex(["negative","neutral","positive"]).fillna(0).plot(kind="bar", rot=0)
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_wordcloud(texts, out_path, width=1200, height=800):
    if not texts:
        return
    wc = WordCloud(width=width, height=height, background_color="white").generate(" ".join(texts))
    wc.to_file(out_path)

def representative_examples(df, topic_id, k=5):
    dft = df[df["topic"] == topic_id].sort_values("topic_prob", ascending=False)
    return dft["feedback"].head(k).tolist()

def summarize_with_openai(prompt, model="gpt-4o-mini"):
    client = OpenAI()
    out = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":"You are a helpful analyst. Reply in Thai."},
                  {"role":"user","content": prompt}],
        temperature=0.2,
        max_tokens=600,
    )
    return out.choices[0].message.content

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--min_topic_size", type=int, default=30)
    parser.add_argument("--use_openai", action="store_true")
    args = parser.parse_args()

    global USE_OPENAI
    USE_OPENAI = args.use_openai and ("OPENAI_API_KEY" in os.environ) and (OpenAI is not None)
    if args.use_openai and not USE_OPENAI:
        print("[WARN] --use_openai set but OPENAI_API_KEY or openai client missing. Skipping LLM summaries.")

    ensure_dir(args.out_dir)
    figures = build_figures_dir(args.out_dir)

    df = pd.read_csv(args.csv_path)
    # normalize columns
    df = df.rename(columns={v:k for k,v in COLS.items()})
    for c in ["timestamp","team","feedback"]:
        if c not in df.columns:
            df[c] = np.nan

    # cleaning
    df["feedback"] = df["feedback"].map(clean_text)
    df = df[df["feedback"].str.len() > 0].copy()
    df["team"] = df["team"].fillna("Unspecified")

    # sentiment (fast heuristic; replace with model as needed)
    df["sentiment_score"] = df["feedback"].map(quick_polarity_heuristic)
    df["sentiment_bucket"] = pd.cut(df["sentiment_score"], bins=[-1.01,-0.2,0.2,1.01], labels=["negative","neutral","positive"])

    # tokenization for wordclouds (Thai-aware)
    df["tokens"] = df["feedback"].map(tokenize_th)

    # Embeddings + BERTopic
    print("Loading embedding model:", args.model)
    embedder = SentenceTransformer(args.model)
    embeddings = embedder.encode(df["feedback"].tolist(), show_progress_bar=True)

    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42)
    hdbscan_model = hdbscan.HDBSCAN(min_cluster_size=args.min_topic_size, metric="euclidean", cluster_selection_method="leaf", prediction_data=True)

    topic_model = BERTopic(
        language="multilingual",
        min_topic_size=args.min_topic_size,
        calculate_probabilities=True,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=False,
    )
    topics, probs = topic_model.fit_transform(df["feedback"].tolist(), embeddings=embeddings)
    df["topic"] = topics
    # Extract max prob safely
    def max_prob(p):
        try:
            return float(np.max(p))
        except Exception:
            return float(p) if p is not None else 0.0
    df["topic_prob"] = [max_prob(p) for p in probs]

    # topic descriptions/keywords
    topic_info = topic_model.get_topic_info()
    topic_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df["topic_name"] = df["topic"].map(topic_map).fillna("Outliers")

    # Export enriched
    out_csv = Path(args.out_dir)/"feedback_enriched.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)

    # Figures: overall sentiment bar
    plot_sentiment_bar(df, Path(figures)/"sentiment_overall.png")

    # Wordcloud overall
    plot_wordcloud(df["tokens"].tolist(), Path(figures)/"wordcloud_overall.png")

    # Per-team figures
    for team, dft in df.groupby("team"):
        safe = re.sub(r"[^ก-๙a-zA-Z0-9_-]+","_", str(team))[:40] or "team"
        plot_sentiment_bar(dft, Path(figures)/f"sentiment_{safe}.png")
        plot_wordcloud(dft["tokens"].tolist(), Path(figures)/f"wordcloud_{safe}.png")

    # Build topics summary
    rows = []
    for t in sorted(set(df["topic"])):
        if t == -1:  # outliers
            continue
        name = topic_map.get(t, f"Topic {t}")
        examples = representative_examples(df, t, k=5)
        rows.append({
            "topic_id": t,
            "topic_name": name,
            "count": int((df["topic"]==t).sum()),
            "examples": " | ".join(examples),
        })
    topics_summary = pd.DataFrame(rows).sort_values("count", ascending=False)
    topics_summary.to_csv(Path(args.out_dir)/"topics_summary.csv", index=False, encoding="utf-8-sig")

    # Team findings table
    team_rows = []
    for team, dft in df.groupby("team"):
        cnt = len(dft)
        pos = int((dft["sentiment_bucket"]=="positive").sum())
        neg = int((dft["sentiment_bucket"]=="negative").sum())
        neu = cnt - pos - neg
        top_topics = dft["topic_name"].value_counts().head(5).index.tolist()
        # Top keywords via CountVectorizer on tokens
        vec = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
        try:
            X = vec.fit_transform(dft["tokens"].tolist())
            freqs = np.asarray(X.sum(axis=0)).ravel()
            vocab = np.array(vec.get_feature_names_out())
            top_idx = np.argsort(freqs)[::-1][:10]
            top_kw = vocab[top_idx].tolist()
        except Exception:
            top_kw = []
        team_rows.append({
            "team": team,
            "n_comments": cnt,
            "pct_positive": round(100*pos/max(cnt,1),2),
            "pct_negative": round(100*neg/max(cnt,1),2),
            "pct_neutral": round(100*neu/max(cnt,1),2),
            "top_topics": ", ".join(top_topics),
            "top_keywords": ", ".join(top_kw),
        })
    team_findings = pd.DataFrame(team_rows).sort_values("n_comments", ascending=False)
    team_findings.to_csv(Path(args.out_dir)/"team_findings.csv", index=False, encoding="utf-8-sig")

    # Optional: LLM summaries
    if USE_OPENAI:
        try:
            prompt = textwrap.dedent(f"""
            สรุปเชิงบริหาร (Executive Summary) จาก feedback ผู้เข้าอบรม (ภาษาไทย):
            - จำนวนคอมเมนต์ทั้งหมด: {len(df)}
            - สัดส่วน sentiment โดยคร่าว: positive/neutral/negative
            - หัวข้อหลักที่พบ (topic model): ใช้ชื่อหัวข้อและจำนวน
            - ประเด็นเร่งด่วนที่ควรปรับปรุง (3-5 ข้อ)
            - ข้อเสนอแนะเชิงทีม (team-specific) จากไฟล์ team_findings.csv ที่แนบในบริบทนี้ไม่ได้ แต่โปรดสรุปแนวทางทั่วไปสำหรับทีม วิชาการ/เทคนิค/ประสานงาน
            ให้ตอบย่อหน้าอ่านง่าย bullet สั้น ชัดเจน เป็นภาษาไทย
            """)
            summary_txt = summarize_with_openai(prompt)
            with open(Path(args.out_dir)/"executive_summary_th.txt","w",encoding="utf-8") as f:
                f.write(summary_txt or "")
            print("Saved: executive_summary_th.txt")
        except Exception as e:
            print("[WARN] LLM summary failed:", e)

    print("Done. See outputs/ for CSVs and figures.")

if __name__ == "__main__":
    main()
