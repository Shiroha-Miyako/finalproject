import time
from typing import List

import joblib
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


# =========================
# Configuration
# =========================
SENTIMENT_MODEL_ID = "ShirohaNaruse/game-review-sentiment-distilbert"

ISSUE_MODEL_FILE = "final_issue_tfidf_logreg.joblib"
ISSUE_LABEL_MAPPING_FILE = "final_issue_label_mapping.csv"

APP_TITLE = "Steam Review Intelligence Assistant"
FETCH_MAX_REVIEWS = 1000

SENTIMENT_LABEL_MAP = {
    "LABEL_0": "Negative / Not Recommended",
    "LABEL_1": "Positive / Recommended",
    0: "Negative / Not Recommended",
    1: "Positive / Recommended",
    "0": "Negative / Not Recommended",
    "1": "Positive / Recommended",
    "NEGATIVE": "Negative / Not Recommended",
    "POSITIVE": "Positive / Recommended",
}

ACTION_MAP = {
    "Bug / Crash": "Prioritize bug fixing and investigate crash logs or broken gameplay flows.",
    "Performance": "Optimize FPS, loading time, memory usage, and hardware compatibility.",
    "Gameplay": "Review core mechanics, controls, difficulty balance, and progression design.",
    "Content": "Consider adding more levels, missions, story content, updates, or replay value.",
    "Price / Value": "Review pricing, discount strategy, refund reasons, and perceived content value.",
    "Multiplayer / Server": "Improve server stability, matchmaking, connection quality, and anti-cheat systems.",
    "Praise / Strength": "Use this feedback to identify marketing messages and core product strengths.",
    "General": "Review manually if the comment receives many votes or appears in negative feedback.",
}

ISSUE_ORDER = [
    "Bug / Crash",
    "Multiplayer / Server",
    "Performance",
    "Gameplay",
    "Content",
    "Price / Value",
    "Praise / Strength",
]

PRIORITY_ORDER = ["High", "Medium", "Marketing Insight", "Low"]


# =========================
# Model loading
# =========================
@st.cache_resource(show_spinner=True)
def load_artifacts():
    sentiment_tokenizer = AutoTokenizer.from_pretrained(SENTIMENT_MODEL_ID)
    sentiment_tokenizer.model_input_names = ["input_ids", "attention_mask"]

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_MODEL_ID
    )

    sentiment_pipe = pipeline(
        "text-classification",
        model=sentiment_model,
        tokenizer=sentiment_tokenizer,
        truncation=True,
        max_length=256,
    )

    issue_model = joblib.load(ISSUE_MODEL_FILE)
    issue_mapping_df = pd.read_csv(ISSUE_LABEL_MAPPING_FILE)

    if "label" not in issue_mapping_df.columns:
        raise ValueError("final_issue_label_mapping.csv must contain a 'label' column.")

    if "primary_issue_category" in issue_mapping_df.columns:
        category_col = "primary_issue_category"
    elif "category" in issue_mapping_df.columns:
        category_col = "category"
    else:
        raise ValueError(
            "final_issue_label_mapping.csv must contain either "
            "'primary_issue_category' or 'category'."
        )

    issue_mapping_df["label"] = issue_mapping_df["label"].astype(int)

    id2issue = dict(
        zip(
            issue_mapping_df["label"],
            issue_mapping_df[category_col],
        )
    )

    return sentiment_pipe, issue_model, id2issue


# =========================
# Helper functions
# =========================
def normalize_sentiment_label(raw_label):
    return SENTIMENT_LABEL_MAP.get(
        raw_label,
        SENTIMENT_LABEL_MAP.get(str(raw_label), str(raw_label)),
    )


def normalize_issue_label(raw_label, id2issue):
    try:
        return id2issue[int(raw_label)]
    except Exception:
        return str(raw_label)


def is_negative(sentiment: str) -> bool:
    return str(sentiment).lower().startswith("negative")


def get_review_status(df: pd.DataFrame) -> str:
    if df.empty:
        return "N/A"

    negative_share = df["sentiment"].apply(is_negative).mean() * 100
    positive_share = 100 - negative_share

    if positive_share >= 80:
        return "Very Positive"
    elif positive_share >= 70:
        return "Mostly Positive"
    elif positive_share >= 40:
        return "Mixed"
    elif positive_share >= 20:
        return "Mostly Negative"
    else:
        return "Very Negative"


def assign_priority(sentiment: str, issue_category: str) -> str:
    if is_negative(sentiment) and issue_category in [
        "Bug / Crash",
        "Performance",
        "Multiplayer / Server",
    ]:
        return "High"

    if is_negative(sentiment) and issue_category in [
        "Gameplay",
        "Content",
        "Price / Value",
    ]:
        return "Medium"

    if (not is_negative(sentiment)) and issue_category == "Praise / Strength":
        return "Marketing Insight"

    return "Low"


def get_suggested_action(issue_category: str) -> str:
    return ACTION_MAP.get(issue_category, ACTION_MAP["General"])


def analyze_texts(
    texts: List[str],
    sentiment_pipe,
    issue_model,
    id2issue,
    batch_size: int = 16,
) -> pd.DataFrame:
    clean_texts = [str(x) if pd.notna(x) else "" for x in texts]
    clean_texts = [x.strip() for x in clean_texts]

    sentiment_outputs = sentiment_pipe(clean_texts, batch_size=batch_size)
    issue_outputs = issue_model.predict(clean_texts)

    issue_scores = None
    if hasattr(issue_model, "predict_proba"):
        try:
            issue_scores = issue_model.predict_proba(clean_texts)
        except Exception:
            issue_scores = None

    rows = []

    for idx, (text, sentiment_out, issue_label) in enumerate(
        zip(clean_texts, sentiment_outputs, issue_outputs)
    ):
        sentiment = normalize_sentiment_label(sentiment_out.get("label"))
        issue_category = normalize_issue_label(issue_label, id2issue)
        priority = assign_priority(sentiment, issue_category)

        if issue_scores is not None:
            issue_confidence = round(float(max(issue_scores[idx])), 4)
        else:
            issue_confidence = None

        rows.append(
            {
                "review_text": text,
                "sentiment": sentiment,
                "sentiment_confidence": round(float(sentiment_out.get("score", 0)), 4),
                "issue_category": issue_category,
                "issue_confidence": issue_confidence,
                "priority": priority,
                "suggested_action": get_suggested_action(issue_category),
            }
        )

    return pd.DataFrame(rows)


def generate_summary(df: pd.DataFrame) -> str:
    if df.empty:
        return "No reviews were analyzed."

    total = len(df)

    negative_count = int(df["sentiment"].apply(is_negative).sum())
    negative_share = negative_count / total * 100
    review_status = get_review_status(df)

    high_priority = int((df["priority"] == "High").sum())

    issue_counts = df["issue_category"].value_counts()
    top_issue = issue_counts.index[0] if not issue_counts.empty else "N/A"
    top_issue_share = issue_counts.iloc[0] / total * 100 if not issue_counts.empty else 0

    high_issue_counts = df[df["priority"] == "High"]["issue_category"].value_counts()

    if not high_issue_counts.empty:
        top_high_issue = high_issue_counts.index[0]
        focus_sentence = (
            f"Among high-priority reviews, the most frequent issue is {top_high_issue}. "
            f"This suggests that developers may need to review this area first."
        )
    else:
        focus_sentence = "No high-priority issue cluster was detected in this batch."

    praise_count = int((df["issue_category"] == "Praise / Strength").sum())

    if praise_count > 0:
        praise_sentence = (
            f"The app also identified {praise_count} praise-related reviews, which may help "
            f"identify product strengths and potential marketing messages."
        )
    else:
        praise_sentence = (
            "Praise-related reviews are limited in this batch, so the current analysis is mainly useful "
            "for identifying product issues."
        )

    return (
        f"Among {total:,} analyzed reviews, the recent review status is **{review_status}**, "
        f"with {negative_share:.1f}% classified as negative. "
        f"The most frequent issue category is {top_issue}, accounting for {top_issue_share:.1f}% of all analyzed reviews. "
        f"The app identified {high_priority:,} high-priority reviews that may require developer attention. "
        f"{focus_sentence} {praise_sentence}"
    )


def render_kpi_cards(df: pd.DataFrame):
    total = len(df)
    negative_share = df["sentiment"].apply(is_negative).mean() * 100 if total else 0
    positive_share = 100 - negative_share if total else 0
    high_priority = int((df["priority"] == "High").sum()) if total else 0
    review_status = get_review_status(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Reviews", f"{total:,}")
    c2.metric("Positive Share", f"{positive_share:.1f}%")
    c3.metric("High Priority", f"{high_priority:,}")
    c4.metric("Recent Review Status", review_status)


def render_interactive_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df

    st.subheader("Interactive Review Explorer")
    st.write(
        "Use the filters below to focus on a specific issue category, sentiment group, or priority level. "
        "The dashboard and detailed results will update automatically without fetching the reviews again."
    )

    col1, col2, col3 = st.columns(3)

    issue_options = ["All"] + [x for x in ISSUE_ORDER if x in set(df["issue_category"])]
    sentiment_options = ["All"] + list(df["sentiment"].dropna().unique())
    priority_options = ["All"] + [x for x in PRIORITY_ORDER if x in set(df["priority"])]

    with col1:
        selected_issue = st.selectbox(
            "Issue Category",
            issue_options,
            key=f"{key_prefix}_issue_filter",
        )

    with col2:
        selected_sentiment = st.selectbox(
            "Sentiment",
            sentiment_options,
            key=f"{key_prefix}_sentiment_filter",
        )

    with col3:
        selected_priority = st.selectbox(
            "Priority",
            priority_options,
            key=f"{key_prefix}_priority_filter",
        )

    filtered_df = df.copy()

    if selected_issue != "All":
        filtered_df = filtered_df[filtered_df["issue_category"] == selected_issue]

    if selected_sentiment != "All":
        filtered_df = filtered_df[filtered_df["sentiment"] == selected_sentiment]

    if selected_priority != "All":
        filtered_df = filtered_df[filtered_df["priority"] == selected_priority]

    st.caption(
        f"Showing {len(filtered_df):,} of {len(df):,} analyzed reviews after filtering."
    )

    return filtered_df


def render_charts(df: pd.DataFrame):
    if df.empty:
        st.info("No data to visualize.")
        return

    col1, col2 = st.columns(2)

    with col1:
        sentiment_counts = df["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]

        fig = px.bar(
            sentiment_counts,
            x="sentiment",
            y="count",
            title="Sentiment Distribution",
            text="count",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Number of Reviews")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        priority_counts = (
            df["priority"]
            .value_counts()
            .reindex(PRIORITY_ORDER)
            .dropna()
            .reset_index()
        )
        priority_counts.columns = ["priority", "count"]

        fig = px.bar(
            priority_counts,
            x="priority",
            y="count",
            title="Priority Distribution",
            text="count",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Number of Reviews")
        st.plotly_chart(fig, use_container_width=True)

    issue_counts = (
        df["issue_category"]
        .value_counts()
        .reindex(ISSUE_ORDER)
        .dropna()
        .reset_index()
    )
    issue_counts.columns = ["issue_category", "count"]

    fig = px.bar(
        issue_counts,
        x="issue_category",
        y="count",
        title="Issue Category Distribution",
        text="count",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Number of Reviews")
    st.plotly_chart(fig, use_container_width=True)

    heatmap_data = (
        df.groupby(["issue_category", "sentiment"])
        .size()
        .reset_index(name="count")
    )

    if not heatmap_data.empty:
        pivot = (
            heatmap_data
            .pivot(index="issue_category", columns="sentiment", values="count")
            .fillna(0)
        )
        pivot = pivot.reindex([x for x in ISSUE_ORDER if x in pivot.index])

        fig = px.imshow(
            pivot,
            text_auto=True,
            aspect="auto",
            title="Issue Category × Sentiment Heatmap",
        )
        st.plotly_chart(fig, use_container_width=True)


def render_analysis_results(df: pd.DataFrame, key_prefix: str):
    st.subheader("Executive Summary")
    render_kpi_cards(df)
    st.write(generate_summary(df))

    filtered_df = render_interactive_filters(df, key_prefix=key_prefix)

    if filtered_df.empty:
        st.warning("No reviews match the selected filters.")
        return

    st.subheader("Filtered Dashboard")
    render_charts(filtered_df)

    st.subheader("Filtered Detailed Results")
    st.dataframe(filtered_df, use_container_width=True, height=420)

    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download filtered results as CSV",
        data=csv_bytes,
        file_name="filtered_analyzed_reviews.csv",
        mime="text/csv",
        key=f"{key_prefix}_download_filtered",
    )

    with st.expander("Show all analyzed results"):
        st.dataframe(df, use_container_width=True, height=360)

        all_csv_bytes = df.to_csv(index=False).encode("utf-8-sig")

        st.download_button(
            label="Download all analyzed results as CSV",
            data=all_csv_bytes,
            file_name="all_analyzed_reviews.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_all",
        )


def fetch_steam_reviews_by_app_id(
    app_id: str,
    max_reviews: int = 100,
    language: str = "english",
) -> pd.DataFrame:
    app_id = str(app_id).strip()

    if not app_id.isdigit():
        raise ValueError("Steam App ID must be a numeric ID, such as 1086940.")

    max_reviews = int(max_reviews)
    max_reviews = max(1, min(max_reviews, FETCH_MAX_REVIEWS))

    base_url = f"https://store.steampowered.com/appreviews/{app_id}"
    reviews = []
    cursor = "*"

    while len(reviews) < max_reviews:
        params = {
            "json": 1,
            "filter": "recent",
            "language": language,
            "num_per_page": min(100, max_reviews - len(reviews)),
            "cursor": cursor,
            "purchase_type": "all",
        }

        response = requests.get(
            base_url,
            params=params,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=20,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Steam request failed with status code {response.status_code}.")

        data = response.json()

        if not data.get("success"):
            raise RuntimeError("Steam review endpoint returned an unsuccessful response.")

        batch = data.get("reviews", [])

        if not batch:
            break

        for item in batch:
            review_text = item.get("review", "")

            if review_text and str(review_text).strip():
                reviews.append(
                    {
                        "review_text": review_text,
                        "voted_up": item.get("voted_up", None),
                        "votes_up": item.get("votes_up", 0),
                        "weighted_vote_score": item.get("weighted_vote_score", 0),
                        "timestamp_created": item.get("timestamp_created", None),
                    }
                )

        new_cursor = data.get("cursor")

        if not new_cursor or new_cursor == cursor:
            break

        cursor = new_cursor
        time.sleep(0.25)

    return pd.DataFrame(reviews).head(max_reviews)


# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Steam Review Intelligence Assistant")
st.caption(
    "A developer-oriented review triage tool combining a fine-tuned Hugging Face sentiment model "
    "with a TF-IDF issue category classifier."
)

try:
    sentiment_pipe, issue_model, id2issue = load_artifacts()
except Exception as exc:
    st.error(
        "Model loading failed. Please check the Hugging Face sentiment model ID, "
        "final_issue_tfidf_logreg.joblib, and final_issue_label_mapping.csv."
    )
    st.exception(exc)
    st.stop()


if "steam_result_df" not in st.session_state:
    st.session_state["steam_result_df"] = None

if "steam_app_id" not in st.session_state:
    st.session_state["steam_app_id"] = None

if "csv_result_df" not in st.session_state:
    st.session_state["csv_result_df"] = None


tab_steam, tab_csv, tab_about = st.tabs(
    [
        "Fetch by Steam App ID",
        "Batch CSV Dashboard",
        "About the Model",
    ]
)


with tab_steam:
    st.header("Fetch Steam Reviews by App ID")

    st.write(
        "Enter a numeric Steam App ID. You can find it in the Steam store URL. "
        "For example, in `https://store.steampowered.com/app/1086940/...`, "
        "the App ID is `1086940`."
    )

    st.info(
        "This feature uses Steam's public review endpoint. For live demos, "
        "fetching 100–200 reviews is usually more stable than very large batches."
    )

    app_id = st.text_input("Steam App ID", value="1086940")

    review_count = st.selectbox(
        "Number of recent English reviews to fetch",
        [20, 50, 100, 200, 500, 1000],
        index=2,
    )

    if review_count >= 500:
        st.warning(
            "Large fetches may take longer on Streamlit Cloud. "
            "For classroom demos, 100–200 reviews is safer."
        )

    fetch_clicked = st.button("Fetch and Analyze Steam Reviews", type="primary")

    if fetch_clicked:
        try:
            with st.spinner("Fetching recent Steam reviews..."):
                fetched_df = fetch_steam_reviews_by_app_id(
                    app_id,
                    max_reviews=review_count,
                )

            if fetched_df.empty:
                st.warning("No reviews found. Please check the App ID or try CSV upload.")
                st.session_state["steam_result_df"] = None
                st.session_state["steam_app_id"] = None
            else:
                st.success(f"Fetched {len(fetched_df):,} reviews from Steam App ID {app_id}.")

                with st.spinner("Analyzing fetched reviews..."):
                    result_df = analyze_texts(
                        fetched_df["review_text"].tolist(),
                        sentiment_pipe,
                        issue_model,
                        id2issue,
                    )

                meta_cols = [c for c in fetched_df.columns if c != "review_text"]

                if meta_cols:
                    result_df = pd.concat(
                        [result_df, fetched_df[meta_cols].reset_index(drop=True)],
                        axis=1,
                    )

                st.session_state["steam_result_df"] = result_df
                st.session_state["steam_app_id"] = app_id

        except Exception as exc:
            st.error(
                "Failed to fetch or analyze Steam reviews. Please check the App ID, "
                "try a smaller number of reviews, or use CSV upload."
            )
            st.exception(exc)

    if st.session_state["steam_result_df"] is not None:
        st.divider()
        st.caption(
            f"Showing stored analysis results for Steam App ID "
            f"{st.session_state['steam_app_id']}. Changing filters below will not fetch reviews again."
        )
        render_analysis_results(
            st.session_state["steam_result_df"],
            key_prefix="steam",
        )


with tab_csv:
    st.header("Batch CSV Dashboard")

    st.write(
        "Upload a CSV file with one required column named `review_text`. "
        "Each row should contain one review."
    )

    st.markdown(
        """
**Required CSV format**

| review_text |
|---|
| The game keeps crashing after the latest update. |
| I love the story and soundtrack. |
| The multiplayer server is unstable and laggy. |

Optional columns such as `app_name`, `review_score`, or `review_votes` can be included, but only `review_text` is required.
        """
    )

    sample_csv = pd.DataFrame(
        {
            "review_text": [
                "The game keeps crashing after the latest update.",
                "I love the story and soundtrack.",
                "The multiplayer server is unstable and laggy.",
            ]
        }
    ).to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download sample CSV template",
        data=sample_csv,
        file_name="sample_review_upload.csv",
        mime="text/csv",
        key="download_sample_csv",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    max_rows = st.slider(
        "Maximum rows to analyze",
        min_value=10,
        max_value=1000,
        value=200,
        step=10,
    )

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)

            if "review_text" not in input_df.columns:
                st.error("The uploaded CSV must contain a column named `review_text`.")
            else:
                input_df = input_df.dropna(subset=["review_text"]).head(max_rows)

                st.write(f"Loaded {len(input_df):,} reviews for analysis.")

                analyze_csv_clicked = st.button("Analyze Uploaded Reviews", type="primary")

                if analyze_csv_clicked:
                    with st.spinner("Running sentiment and issue category models..."):
                        result_df = analyze_texts(
                            input_df["review_text"].tolist(),
                            sentiment_pipe,
                            issue_model,
                            id2issue,
                        )

                    extra_cols = [
                        c for c in input_df.columns
                        if c != "review_text"
                    ]

                    if extra_cols:
                        result_df = pd.concat(
                            [result_df, input_df[extra_cols].reset_index(drop=True)],
                            axis=1,
                        )

                    st.session_state["csv_result_df"] = result_df

        except Exception as exc:
            st.error("Failed to read or analyze the uploaded CSV.")
            st.exception(exc)

    if st.session_state["csv_result_df"] is not None:
        st.divider()
        st.caption(
            "Showing stored CSV analysis results. Changing filters below will not re-run the model."
        )
        render_analysis_results(
            st.session_state["csv_result_df"],
            key_prefix="csv",
        )


with tab_about:
    st.header("About the Model")

    st.markdown(
        """
### Project purpose
This app helps developers analyze player reviews by turning unstructured review text into structured product feedback.

### Final model selection
The final system uses two complementary models:

1. **Sentiment Classification**  
   A fine-tuned DistilBERT model predicts whether a review is positive/recommended or negative/not recommended.

2. **Issue Category Classification**  
   A TF-IDF + Logistic Regression model predicts the developer-oriented issue category:
   Bug / Crash, Multiplayer / Server, Performance, Gameplay, Content, Price / Value, or Praise / Strength.

### Why this combination?
The fine-tuned DistilBERT model performed better for sentiment classification.  
For issue category classification, TF-IDF + Logistic Regression was selected because it outperformed the fine-tuned DistilBERT issue model under the limited training setting. This is also reasonable because the issue labels were generated through keyword-based weak labeling, making sparse keyword features effective for this task.

### Interactive analysis
The dashboard includes filters for issue category, sentiment, and priority level. After selecting a filter, the charts and detailed review table update automatically using the stored analysis results, without fetching reviews again.

### App outputs
The app provides:
- sentiment prediction,
- issue category prediction,
- priority level,
- suggested developer action,
- interactive dashboard visualizations,
- filtered detailed review results,
- downloadable analyzed results.

### Important limitation
The issue category labels were created through weak labeling. Therefore, issue category predictions should be interpreted as developer-oriented triage support rather than perfect human annotations.
        """
    )
