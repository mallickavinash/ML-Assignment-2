import io

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from utils import compute_metrics

st.set_page_config(
    page_title="ModelScope | Classifier Analytics",
    page_icon="MS",
    layout="wide",
    initial_sidebar_state="expanded",
)

sns.set_theme(style="whitegrid")

st.markdown(
    """
    <style>
        [data-testid="stAppViewContainer"] { background: #f5f7fb; }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8edf5; }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #697386; }
        .block-container { padding-top: 1.75rem; padding-bottom: 3rem; max-width: 1280px; }
        h1, h2, h3 { color: #172033 !important; letter-spacing: -0.02em; }
        [data-testid="stMetric"] {
            background: #ffffff; border: 1px solid #e8edf5; border-radius: 14px;
            padding: 18px 20px; box-shadow: 0 4px 16px rgba(31, 45, 61, 0.04);
        }
        [data-testid="stMetricLabel"] { color: #697386 !important; font-size: 0.78rem; }
        [data-testid="stMetricValue"] { color: #172033 !important; font-size: 1.65rem; }
        div[data-testid="stExpander"] {
            border: 1px solid #e8edf5; border-radius: 12px; background: white;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #edf0f5; border-radius: 10px; overflow: hidden;
        }
        .hero {
            background: linear-gradient(120deg, #18233a 0%, #263858 100%);
            border-radius: 18px; padding: 28px 32px; margin-bottom: 24px; color: white;
            box-shadow: 0 12px 30px rgba(24, 35, 58, 0.16);
        }
        .hero h1 { color: white !important; font-size: 2rem; margin-bottom: 6px; }
        .hero p { color: #cbd5e5 !important; margin: 0; }
        .section-label {
            color: #7b8495; font-size: 0.72rem; font-weight: 700;
            letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px;
        }
        .status-pill {
            display: inline-block; background: #ecfdf3; color: #16794a;
            border-radius: 999px; padding: 5px 10px; font-size: 0.78rem; font-weight: 600;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib",
}


@st.cache_resource
def load_model(path):
    return joblib.load(path)


@st.cache_resource
def load_meta():
    return joblib.load("model/meta.joblib")


@st.cache_data(max_entries=4)
def load_uploaded_csv(file_content):
    return pd.read_csv(io.BytesIO(file_content))


@st.cache_data(max_entries=20)
def evaluate_model(file_content, model_path, features, target, classes):
    df = load_uploaded_csv(file_content)
    label_to_index = {label: index for index, label in enumerate(classes)}
    y = df[target].map(label_to_index).to_numpy()
    return compute_metrics(load_model(model_path), df[list(features)], y)


meta = load_meta()
le, features, target = meta["label_encoder"], meta["features"], meta["target"]

with st.sidebar:
    st.markdown("## ModelScope")
    st.caption("Classifier analytics workspace")
    st.markdown("---")
    st.markdown("### Evaluation setup")
    uploaded = st.file_uploader("Upload test CSV", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))
    st.markdown(f"**Required columns:** {len(features)} features + `{target}`")
    with st.expander("View feature schema"):
        st.code("\n".join(features), language="text")
    st.markdown("---")
    st.caption("Models are pre-trained and loaded from the local model registry.")

st.markdown(
    """
    <section class="hero">
        <h1>Classifier performance workspace</h1>
        <p>Validate pre-trained models, investigate classification behavior, and compare performance.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

if uploaded is None:
    st.markdown('<div class="section-label">Evaluation workspace</div>', unsafe_allow_html=True)
    empty, content = st.columns([1, 4])
    with content:
        st.info("Upload `test_data.csv` from the sidebar to begin an evaluation.")
        st.caption("The uploaded file should contain all required feature columns and the target column.")
    st.stop()

file_content = uploaded.getvalue()
df = load_uploaded_csv(file_content)
missing = [c for c in features + [target] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing columns: {missing}")
    st.stop()

unseen = set(df[target].unique()) - set(le.classes_)
if unseen:
    st.error(f"Target contains labels the model has never seen: {unseen}")
    st.stop()

X, y = df[features], le.transform(df[target])
class_labels = tuple(le.classes_.tolist())
feature_names = tuple(features)
metrics, y_pred = evaluate_model(
    file_content, MODEL_FILES[choice], feature_names, target, class_labels
)

report_tab, inspect_tab = st.tabs(["Performance overview", "Dataset & predictions"])

with report_tab:
    st.markdown('<span class="status-pill">Evaluation ready</span>', unsafe_allow_html=True)
    overview = st.columns(4)
    overview[0].metric("Model", choice)
    overview[1].metric("Samples", f"{len(df):,}")
    overview[2].metric("Features", len(features))
    overview[3].metric("Classes", len(le.classes_))

st.markdown('<div class="section-label">Performance summary</div>', unsafe_allow_html=True)
cols = st.columns(len(metrics))
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label.replace("_", " ").title(), f"{value:.4f}")

analysis_left, analysis_right = st.columns([1.05, 0.95], gap="large")
with analysis_left:
    st.markdown("#### Confusion matrix")
    st.caption(f"Actual versus predicted labels for {choice}")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with analysis_right:
    st.markdown("#### Class-level metrics")
    st.caption("Precision, recall, F1-score, and support")
    report = classification_report(
        y,
        y_pred,
        target_names=le.classes_.astype(str),
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.round(3)
    report_df["support"] = report_df["support"].astype(int)
    st.dataframe(report_df, use_container_width=True, height=330)

with st.expander("Model comparison", expanded=False):
    st.caption("Compare all models on the current test data. This may take longer on large files.")
    if st.button("Run model comparison", type="primary", use_container_width=True):
        rows = []
        progress = st.progress(0)
        for index, (name, path) in enumerate(MODEL_FILES.items()):
            model_metrics, _ = evaluate_model(
                file_content, path, feature_names, target, class_labels
            )
            rows.append({"Model": name, **model_metrics})
            progress.progress((index + 1) / len(MODEL_FILES))
        comparison = pd.DataFrame(rows).round(4)
        comparison = comparison.sort_values("F1", ascending=False).reset_index(drop=True)
        st.dataframe(comparison, hide_index=True, use_container_width=True)
        st.download_button(
            "Download comparison CSV",
            comparison.to_csv(index=False).encode("utf-8"),
            "model_comparison.csv",
            "text/csv",
        )

with inspect_tab:
    st.markdown("#### Prediction review")
    predictions = pd.DataFrame(
        {
            "Actual": df[target].astype(str).to_numpy(),
            "Predicted": le.inverse_transform(y_pred).astype(str),
        }
    )
    predictions["Result"] = predictions["Actual"].eq(predictions["Predicted"]).map(
        {True: "Correct", False: "Incorrect"}
    )
    st.dataframe(predictions.head(200), use_container_width=True, height=320)
    st.download_button(
        "Download predictions CSV",
        predictions.to_csv(index=False).encode("utf-8"),
        "predictions.csv",
        "text/csv",
        use_container_width=True,
    )
