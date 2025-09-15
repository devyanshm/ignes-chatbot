import os, json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from google import genai
from google.genai import types


# ---------- Helpers ----------
def analyze_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        st.error("Only CSV or Excel files are supported.")
        return None


def call_gemini(prompt_text, temperature=0.3):
    api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        api_key = "AIzaSyBvUlevb7sqzlO6iaiEHqnCPA3aoO5IP2M"  # fallback to hardcoded key
        st.warning("Using hardcoded API key. For security, set the GOOGLE_GENAI_API_KEY environment variable.")
    client = genai.Client(api_key=api_key)
    cfg = types.GenerateContentConfig(temperature=temperature)
    contents = [types.Content(role="user",
                              parts=[types.Part.from_text(text=prompt_text)])]
    out = ""
    for chunk in client.models.generate_content_stream(
            model="gemini-2.5-flash", contents=contents, config=cfg):
        if chunk.candidates and chunk.candidates[0].content.parts:
            part = chunk.candidates[0].content.parts[0]
            if getattr(part, "text", None):
                out += part.text
    return out.strip()


def parse_chart_spec(text):
    try:
        start, end = text.find("{"), text.rfind("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def draw_chart(df, spec):
    valid_cols = {c.lower().strip(): c for c in df.columns}
    norm = lambda n: valid_cols.get(n.lower().strip()) if n else None

    chart_type = spec.get("chart_type")
    x, y, hue = norm(spec.get("x_axis")), norm(spec.get("y_axis")), norm(spec.get("hue"))
    if not x or (chart_type != "histogram" and not y and chart_type != "scatter"):
        st.error("Gemini chose invalid columns.")
        return

    fig, ax = plt.subplots()
    if chart_type == "line":
        sns.lineplot(data=df, x=x, y=y, hue=hue, ax=ax)
    elif chart_type == "bar":
        sns.barplot(data=df, x=x, y=y, hue=hue, ax=ax)
    elif chart_type == "scatter":
        sns.scatterplot(data=df, x=x, y=y, hue=hue, ax=ax)
    elif chart_type == "histogram":
        sns.histplot(df[x], kde=True, ax=ax)
        ax.set_ylabel("Frequency")
    else:
        st.error(f"Unsupported chart type: {chart_type}")
        return
    ax.set_title(f"{chart_type.capitalize()} Chart")
    st.pyplot(fig)


# ---------- Streamlit App ----------
st.set_page_config(page_title="Ingnes – AI Data Analyst", layout="wide")
st.title("📊 Ingnes – AI Data Analyst")

# Sidebar controls
st.sidebar.header("Options")
uploaded = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
get_insights = st.sidebar.button("🔍 Overall Insights")
user_q = st.sidebar.text_input("💬 Ask a Question")
chart_request = st.sidebar.text_input("📈 Describe a Chart")

# Main content
if uploaded:
    df = analyze_file(uploaded)
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df, use_container_width=True, height=600)  # full scrollable grid

        st.subheader("Summary Statistics")
        st.dataframe(
            df.describe(include="all").transpose().style.format(precision=3),
            use_container_width=True,
        )

        if get_insights:
            prompt = (
                f"You are Ingnes, a helpful data-analysis assistant.\n"
                f"Dataset columns: {', '.join(df.columns)}.\n"
                "Provide key insights, patterns, and chart suggestions."
            )
            with st.spinner("Thinking..."):
                st.subheader("Ingnes Insights")
                st.write(call_gemini(prompt))

        if user_q:
            q_prompt = (
                f"Columns: {', '.join(df.columns)}.\n"
                f"Question: {user_q}\n"
                "Answer in plain language."
            )
            with st.spinner("Thinking..."):
                st.write(call_gemini(q_prompt))

        if chart_request:
            c_prompt = (
                f"Dataset columns: {', '.join(df.columns)}\n"
                f"User request: \"{chart_request}\"\n"
                'Respond ONLY in JSON with keys: '
                '{"chart_type","x_axis","y_axis","hue"}'
            )
            with st.spinner("Planning chart..."):
                spec = parse_chart_spec(call_gemini(c_prompt))
            if spec:
                st.json(spec)
                draw_chart(df, spec)
            else:
                st.error("Could not interpret chart request.")
