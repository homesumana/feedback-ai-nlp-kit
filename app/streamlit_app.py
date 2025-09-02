
# streamlit_app.py
import os, re, textwrap, json
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st

# Optional LLM
try:
    from openai import OpenAI
    HAS_OPENAI = True
except Exception:
    HAS_OPENAI = False

st.set_page_config(page_title="Training Feedback Insights", layout="wide")

st.title("📊 Training Feedback Insights (AI/NLP)")

data_path = st.sidebar.text_input("Path to feedback_enriched.csv", "outputs/feedback_enriched.csv")
if not Path(data_path).exists():
    st.warning("Run the pipeline first to create outputs/feedback_enriched.csv")
    st.stop()

df = pd.read_csv(data_path)
df["team"] = df["team"].fillna("Unspecified")
teams = ["(All)"] + sorted(df["team"].unique().tolist())
sel_team = st.sidebar.selectbox("Team filter", teams, index=0)
if sel_team != "(All)":
    df = df[df["team"] == sel_team]

st.sidebar.markdown("—")
topic_names = ["(All)"] + sorted(df["topic_name"].dropna().unique().tolist())
sel_topic = st.sidebar.selectbox("Topic filter", topic_names, index=0)
if sel_topic != "(All)":
    df = df[df["topic_name"] == sel_topic]

st.sidebar.markdown("—")
kw = st.sidebar.text_input("Keyword contains", "")
if kw.strip():
    df = df[df["feedback"].str.contains(kw.strip(), case=False, na=False)]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Comments", len(df))
with col2:
    pos = int((df["sentiment_bucket"]=="positive").sum())
    neg = int((df["sentiment_bucket"]=="negative").sum())
    st.metric("Positive - Negative", f"{pos} / {neg}")
with col3:
    st.metric("Distinct Topics", df["topic_name"].nunique())

# Sentiment bar
sent_counts = df["sentiment_bucket"].value_counts().rename_axis("sentiment").reset_index(name="count")
fig = px.bar(sent_counts, x="sentiment", y="count", title="Sentiment distribution")
st.plotly_chart(fig, use_container_width=True)

# Top topics
top_topics = df["topic_name"].value_counts().reset_index()
top_topics.columns = ["topic_name","count"]
fig2 = px.bar(top_topics.head(15), x="topic_name", y="count", title="Top Topics (by count)")
st.plotly_chart(fig2, use_container_width=True)

# Table with sample comments
st.subheader("Representative comments")
sample = df.sort_values("topic_prob", ascending=False).head(200)[["team","topic_name","sentiment_bucket","feedback"]]
st.dataframe(sample, use_container_width=True, height=400)

# LLM Recommendations (optional)
st.subheader("Recommendations (LLM-generated, optional)")
if HAS_OPENAI and ("OPENAI_API_KEY" in os.environ):
    client = OpenAI()
    # compact topic and sentiment context
    ctx_topics = df["topic_name"].value_counts().head(10).to_dict()
    ctx_sents = df["sentiment_bucket"].value_counts().to_dict()
    prompt = textwrap.dedent(f"""
    ช่วยสรุป "ข้อเสนอแนะที่ลงมือทำได้ (actionable)" สำหรับทีมงานจัดอบรม (ตอบไทย):
    - สรุปภาพรวมจาก context: sentiment={ctx_sents}, top_topics={ctx_topics}
    - ให้ bullet 5-8 ข้อ จัดลำดับตามผลกระทบก่อนหลัง
    - แยกข้อเสนอแนะตามทีมงาน: วิชาการ / เทคนิค / ประสานงาน (ถ้าไม่แน่ใจให้เสนอเป็นแนวทางทั่วไป)
    - เขียนสั้น กระชับ ชัดเจน
    """)
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"system","content":"You are a helpful analyst. Reply in Thai."},
                      {"role":"user","content": prompt}],
            temperature=0.2,
            max_tokens=600,
        )
        st.markdown(resp.choices[0].message.content)
    except Exception as e:
        st.info(f"LLM unavailable: {e}")
else:
    st.info("ใส่ OPENAI_API_KEY ใน environment เพื่อเปิดการสรุปอัตโนมัติด้วย LLM")
