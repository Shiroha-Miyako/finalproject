import os
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from transformers import pipeline


st.set_page_config(
    page_title="Game Review Intelligence Assistant",
    page_icon="🎮",
    layout="wide",
)

SENTIMENT_MODEL_ID = os.getenv(
    "SENTIMENT_MODEL_ID",
    "ShirohaNaruse/game-review-sentiment-distilbert",
)
ISSUE_MODEL_FILE = Path("final_issue_tfidf_logreg.joblib")
ISSUE_LABEL_MAPPING_FILE = Path("final_issue_label_mapping.csv")

DEFAULT_ID2ISSUE = {
    0: "Bug / Crash",
    1: "Multiplayer / Server",
    2: "Performance",
    3: "Gameplay",
    4: "Content",
    5: "Price / Value",
    6: "Praise / Strength",
}

ISSUE_KEYWORDS = {
    "Bug / Crash": [
        "crash", "crashes", "crashing", "bug", "bugs", "buggy", "broken",
        "won't launch", "won't start", "black screen", "freezes", "freezing",
        "not working", "stuck on the loading screen", "save system", "lose progress",
    ],
    "Performance": [
        "lag", "laggy", "stutter", "stuttering", "low fps", "frame rate",
        "framerate", "optimization", "runs poorly", "sluggish", "unplayable",
    ],
    "Gameplay": [
        "controls", "clunky", "boring", "repetitive", "unbalanced",
        "too hard", "too easy", "difficulty", "hit detection", "combat",
    ],
    "Content": [
        "too short", "short campaign", "lack of content", "not much to do",
        "nothing to do", "story", "ending", "side missions", "dlc", "update", "patch",
    ],
    "Price / Value": [
        "sale", "full price", "worth", "not worth", "waste of money",
        "overpriced", "refund", "money back", "cheap",
    ],
    "Multiplayer / Server": [
        "server", "servers", "multiplayer", "online", "disconnect", "disconnects",
        "can't join", "dead community", "no online players", "player base",
    ],
    "Praise / Strength": [
        "highly recommend", "recommended", "masterpiece", "classic", "gem",
        "great soundtrack", "great music", "compelling story", "addictive", "immersive",
        "charming", "solid game",
    ],
}


def normalize_text(text: str) -> str:
    text = str(text).lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fallback_issue_tag(text: str) -> str:
    normalized = normalize_text(text)
    scores = {}
    for category, keywords in ISSUE_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in normalized)
        if score > 0:
            scores[category] = score
    if not scores:
        return "General"
    return max(scores, key=scores.get)


@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    return pipeline(
        task="text-classification",
        model=SENTIMENT_MODEL_ID,
        tokenizer=SENTIMENT_MODEL_ID,
        truncation=True,
        max_length=256,
    )


@st.cache_resource(show_spinner=False)
def load_issue_model():
    if not ISSUE_MODEL_FILE.exists():
        return None
    return joblib.load(ISSUE_MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_issue_mapping():
    if not ISSUE_LABEL_MAPPING_FILE.exists():
        return DEFAULT_ID2ISSUE
    mapping_df = pd.read_csv(ISSUE_LABEL_MAPPING_FILE)
    if "label" not in mapping_df.columns:
        return DEFAULT_ID2ISSUE
    category_col = "primary_issue_category" if "primary_issue_category" in mapping_df.columns else "category"
    if category_col not in mapping_df.columns:
        return DEFAULT_ID2ISSUE
    return dict(zip(mapping_df["label"].astype(int), mapping_df[category_col].astype(str)))


def sentiment_from_output(output: dict) -> tuple[str, float]:
    label = str(output.get("label", "")).upper()
    score = float(output.get("score", 0.0))
    if label in {"LABEL_1", "POSITIVE", "1"}:
        return "Positive", score
    if label in {"LABEL_0", "NEGATIVE", "0"}:
        return "Negative", score
    return label.title(), score


def predict_issue(text: str, issue_model, id2issue: dict[int, str]) -> str:
    if issue_model is None:
        return fallback_issue_tag(text)
    raw_label = issue_model.predict([text])[0]
    try:
        return id2issue[int(raw_label)]
    except Exception:
        return str(raw_label)


def assign_priority(sentiment: str, issue_category: str) -> str:
    if sentiment == "Negative" and issue_category in {"Bug / Crash", "Performance", "Multiplayer / Server"}:
        return "High"
    if sentiment == "Negative" and issue_category in {"Gameplay", "Content", "Price / Value"}:
        return "Medium"
    if sentiment == "Positive" and issue_category == "Praise / Strength":
        return "Marketing Insight"
    return "Low"


def suggest_action(issue_category: str, sentiment: str, priority: str) -> str:
    if issue_category == "Bug / Crash":
        return "Check crash reports, reproduce the issue, and prioritize a stability patch."
    if issue_category == "Performance":
        return "Review optimization issues such as FPS drops, loading time, and hardware-specific problems."
    if issue_category == "Multiplayer / Server":
        return "Monitor server stability, matchmaking, online connectivity, and player population complaints."
    if issue_category == "Gameplay":
        return "Review complaints about controls, balance, difficulty, and repetitive mechanics."
    if issue_category == "Content":
        return "Check whether players mention lack of content, story weakness, short playtime, or update expectations."
    if issue_category == "Price / Value":
        return "Review pricing, discount strategy, refund complaints, and perceived value for money."
    if issue_category == "Praise / Strength":
        return "Use this as positive feedback for marketing messages, store-page copy, or community posts."
    if sentiment == "Negative":
        return "Review this negative feedback manually and decide whether it points to a product issue."
    return "Use this feedback as general player opinion for monitoring."


def analyze_reviews(texts: list[str], batch_size: int = 16) -> pd.DataFrame:
    clean_texts = [str(text) if pd.notna(text) else "" for text in texts]
    sentiment_pipe = load_sentiment_pipeline()
    issue_model = load_issue_model()
    id2issue = load_issue_mapping()

    sentiment_outputs = sentiment_pipe(clean_texts, batch_size=batch_size)

    rows = []
    for text, output in zip(clean_texts, sentiment_outputs):
        sentiment, confidence = sentiment_from_output(output)
        issue_category = predict_issue(text, issue_model, id2issue)
        priority = assign_priority(sentiment, issue_category)
        rows.append(
            {
                "review_text": text,
                "sentiment": sentiment,
                "sentiment_confidence": round(confidence, 4),
                "issue_category": issue_category,
                "priority": priority,
                "suggested_action": suggest_action(issue_category, sentiment, priority),
            }
        )
    return pd.DataFrame(rows)


def fetch_steam_reviews(app_id: str, num_reviews: int = 200) -> pd.DataFrame:
    url = f"https://store.steampowered.com/appreviews/{app_id}"
    params = {
        "json": 1,
        "filter": "recent",
        "language": "english",
        "review_type": "all",
        "purchase_type": "all",
        "num_per_page": min(num_reviews, 100),
    }

    reviews = []
    cursor = "*"
    while len(reviews) < num_reviews:
        params["cursor"] = cursor
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        batch = data.get("reviews", [])
        if not batch:
            break
        for item in batch:
            reviews.append(item.get("review", ""))
            if len(reviews) >= num_reviews:
                break
        cursor = data.get("cursor")
        if not cursor:
            break
    return pd.DataFrame({"review_text": reviews})


def render_dashboard(result_df: pd.DataFrame):
    st.subheader("Analysis Results")

    metric_cols = st.columns(4)
    metric_cols[0].metric("Reviews", len(result_df))
    metric_cols[1].metric("Positive", int((result_df["sentiment"] == "Positive").sum()))
    metric_cols[2].metric("Negative", int((result_df["sentiment"] == "Negative").sum()))
    metric_cols[3].metric("High Priority", int((result_df["priority"] == "High").sum()))

    chart_cols = st.columns(3)
    with chart_cols[0]:
        sentiment_counts = result_df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        st.plotly_chart(px.bar(sentiment_counts, x="sentiment", y="count", title="Sentiment"), use_container_width=True)
    with chart_cols[1]:
        issue_counts = result_df["issue_category"].value_counts().reset_index()
        issue_counts.columns = ["issue_category", "count"]
        st.plotly_chart(px.bar(issue_counts, x="issue_category", y="count", title="Issue Categories"), use_container_width=True)
    with chart_cols[2]:
        priority_counts = result_df["priority"].value_counts().reset_index()
        priority_counts.columns = ["priority", "count"]
        st.plotly_chart(px.bar(priority_counts, x="priority", y="count", title="Priority"), use_container_width=True)

    st.subheader("Main Takeaways")
    top_issue = result_df["issue_category"].value_counts().idxmax()
    negative_rate = (result_df["sentiment"] == "Negative").mean()
    high_priority_rate = (result_df["priority"] == "High").mean()
    st.write(
        f"The most frequent issue category is **{top_issue}**. "
        f"Negative reviews account for **{negative_rate:.1%}** of the analyzed reviews, "
        f"and **{high_priority_rate:.1%}** are marked as high priority."
    )

    st.dataframe(result_df, use_container_width=True, height=420)
    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Results CSV",
        csv_bytes,
        file_name="review_analysis_results.csv",
        mime="text/csv",
    )


def main():
    st.title("Game Review Intelligence Assistant")
    st.caption("Fine-tuned DistilBERT for sentiment classification + TF-IDF Logistic Regression for issue category classification.")

    with st.sidebar:
        st.header("Model Settings")
        st.write("Sentiment model:")
        st.code(SENTIMENT_MODEL_ID)
        st.write("Issue classifier:")
        if ISSUE_MODEL_FILE.exists():
            st.success("Loaded final_issue_tfidf_logreg.joblib")
        else:
            st.warning("Issue TF-IDF model file is missing. Keyword fallback will be used.")

    tab_single, tab_csv, tab_steam = st.tabs(["Single Review", "CSV Upload", "Steam App ID"])

    with tab_single:
        st.subheader("Analyze One Review")
        review_text = st.text_area("Review text", height=160)
        if st.button("Analyze Review", type="primary"):
            if not review_text.strip():
                st.warning("Please enter a review first.")
            else:
                render_dashboard(analyze_reviews([review_text]))

    with tab_csv:
        st.subheader("Analyze Uploaded CSV")
        uploaded = st.file_uploader("Upload a CSV file with a review_text column", type=["csv"])
        if uploaded is not None:
            input_df = pd.read_csv(uploaded)
            if "review_text" not in input_df.columns:
                string_cols = input_df.select_dtypes(include=["object"]).columns.tolist()
                if not string_cols:
                    st.error("No review_text column or text column found.")
                else:
                    selected_col = st.selectbox("Select text column", string_cols)
                    input_df = input_df.rename(columns={selected_col: "review_text"})
            if "review_text" in input_df.columns:
                max_rows = st.slider("Number of rows to analyze", 10, min(1000, len(input_df)), min(200, len(input_df)))
                if st.button("Analyze CSV", type="primary"):
                    render_dashboard(analyze_reviews(input_df["review_text"].head(max_rows).tolist()))

    with tab_steam:
        st.subheader("Fetch Recent English Steam Reviews")
        st.info("Enter the numeric Steam App ID. For example, in https://store.steampowered.com/app/1086940/..., the App ID is 1086940.")
        app_id = st.text_input("Steam App ID", value="1086940")
        num_reviews = st.slider("Number of recent English reviews to fetch", 20, 500, 100, step=20)
        if st.button("Fetch and Analyze", type="primary"):
            try:
                fetched_df = fetch_steam_reviews(app_id.strip(), num_reviews=num_reviews)
                if fetched_df.empty:
                    st.warning("No reviews were fetched. Please check the App ID or try again later.")
                else:
                    st.write(f"Fetched {len(fetched_df)} reviews.")
                    render_dashboard(analyze_reviews(fetched_df["review_text"].tolist()))
            except Exception as exc:
                st.error("Failed to fetch Steam reviews. Please check the App ID or use CSV upload.")
                st.exception(exc)


if __name__ == "__main__":
    main()
