import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="World Development Clustering Studio",
    page_icon="🌍",
    layout="wide",
)

SAMPLE_PATH = Path(__file__).parent / "World_development_mesurement.xlsx"


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_excel(SAMPLE_PATH)


def clean_dataset(df: pd.DataFrame, null_threshold: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    cleaned = df.copy()

    # Standardize column names
    cleaned.columns = [str(c).strip() for c in cleaned.columns]

    # Remove common numeric symbols from object columns only
    object_cols = cleaned.select_dtypes(include="object").columns.tolist()
    for col in object_cols:
        cleaned[col] = (
            cleaned[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .replace({"nan": np.nan, "None": np.nan, "": np.nan})
        )

    country_col = "Country" if "Country" in cleaned.columns else None

    # Convert all non-country columns to numeric when possible
    for col in cleaned.columns:
        if col == country_col:
            continue
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Drop high-null columns
    null_ratio = cleaned.isna().mean()
    dropped_cols = null_ratio[null_ratio > null_threshold].index.tolist()
    if country_col in dropped_cols:
        dropped_cols.remove(country_col)
    cleaned = cleaned.drop(columns=dropped_cols, errors="coerce")

    # Remove duplicates
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Numeric preparation
    numeric_cols = cleaned.select_dtypes(include=np.number).columns.tolist()
    cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median(numeric_only=True))

    # Keep a useful text id column if present
    id_cols = [country_col] if country_col else []

    return cleaned, numeric_cols, id_cols


def choose_pca_components(scaled_data: np.ndarray, variance_target: float) -> tuple[np.ndarray, PCA, np.ndarray, int]:
    pca_full = PCA()
    pca_full.fit(scaled_data)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.argmax(cumulative_variance >= variance_target) + 1)

    pca = PCA(n_components=n_components, random_state=42)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data, pca, cumulative_variance, n_components


def score_clustering(X: np.ndarray, labels: np.ndarray) -> dict:
    unique_labels = np.unique(labels)
    valid_cluster_labels = [x for x in unique_labels if x != -1]
    n_clusters = len(valid_cluster_labels)

    if n_clusters < 2:
        return {
            "silhouette": np.nan,
            "db_index": np.nan,
            "ch_score": np.nan,
            "clusters": n_clusters,
            "noise_points": int((labels == -1).sum()) if -1 in unique_labels else 0,
        }

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "db_index": float(davies_bouldin_score(X, labels)),
        "ch_score": float(calinski_harabasz_score(X, labels)),
        "clusters": n_clusters,
        "noise_points": int((labels == -1).sum()) if -1 in unique_labels else 0,
    }


def plot_variance_curve(cumulative_variance: np.ndarray, chosen_n: int, target: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(1, len(cumulative_variance) + 1)
    ax.plot(xs, cumulative_variance * 100, marker="o")
    ax.axhline(target * 100, linestyle="--", label=f"Target = {target:.0%}")
    ax.axvline(chosen_n, linestyle="--", label=f"Chosen PCs = {chosen_n}")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance (%)")
    ax.set_title("PCA explained variance")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray, title: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if label == -1:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=65,
                marker="x",
                label="Noise",
            )
        else:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                s=60,
                edgecolors="black",
                label=f"Cluster {label}",
            )
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)


def plot_dendrogram(X: np.ndarray):
    linked = linkage(X, method="ward")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(
        linked,
        truncate_mode="lastp",
        p=20,
        leaf_rotation=45,
        leaf_font_size=9,
        show_contracted=True,
        ax=ax,
    )
    ax.set_title("Hierarchical clustering dendrogram")
    ax.set_xlabel("Cluster groups")
    ax.set_ylabel("Distance")
    ax.grid(alpha=0.2)
    st.pyplot(fig)


def cluster_profiles(df_final: pd.DataFrame, label_col: str) -> pd.DataFrame:
    profile = df_final.groupby(label_col).mean(numeric_only=True).round(2)
    profile.insert(0, "Cluster Size", df_final[label_col].value_counts().sort_index())
    return profile


st.title("🌍 World Development Clustering Studio")
st.caption(
    "A polished clustering app for country segmentation using PCA, KMeans, DBSCAN, and Hierarchical clustering."
)

with st.sidebar:
    st.header("Controls")
    source = st.radio("Data source", ["Use included sample file", "Upload my own file"])
    null_threshold = st.slider("Drop columns above this null ratio", 0.10, 0.80, 0.40, 0.05)
    variance_target = st.slider("PCA variance target", 0.70, 0.99, 0.90, 0.01)
    k_clusters = st.slider("KMeans / Hierarchical clusters", 2, 8, 3, 1)
    dbscan_eps = st.slider("DBSCAN eps", 0.10, 3.00, 1.50, 0.05)
    dbscan_min_samples = st.slider("DBSCAN min_samples", 2, 20, 3, 1)

if source == "Use included sample file":
    raw_df = load_sample_data()
else:
    uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file is None:
        st.info("Upload a file to continue.")
        st.stop()
    if uploaded_file.name.lower().endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)

with st.expander("Preview raw data", expanded=False):
    st.dataframe(raw_df.head(20), use_container_width=True)

cleaned_df, numeric_cols, id_cols = clean_dataset(raw_df, null_threshold)

# Aggregate repeated country rows into one row per country
if "Country" in cleaned_df.columns:
    cleaned_df = cleaned_df.groupby("Country").mean(numeric_only=True).reset_index()

# Recompute numeric columns after grouping
numeric_cols = cleaned_df.select_dtypes(include=np.number).columns.tolist()

if not numeric_cols:
    st.error("No numeric columns found after cleaning. Please review the uploaded file.")
    st.stop()

X = cleaned_df[numeric_cols].copy()

scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)

pca_data, pca_model, cumulative_variance, chosen_n = choose_pca_components(scaled_data, variance_target)
pca_plot = PCA(n_components=2, random_state=42).fit_transform(scaled_data)

st.subheader("Data preparation summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{cleaned_df.shape[0]:,}")
c2.metric("Usable numeric features", len(numeric_cols))
c3.metric("Chosen PCA components", chosen_n)
c4.metric("Variance retained", f"{cumulative_variance[chosen_n - 1] * 100:.2f}%")

plot_variance_curve(cumulative_variance, chosen_n, variance_target)

# Models
kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(pca_data)

dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
dbscan_labels = dbscan.fit_predict(pca_data)

linked = linkage(pca_data, method="ward")
hier_labels = fcluster(linked, t=k_clusters, criterion="maxclust") - 1

results = pd.DataFrame(
    [
        {"Model": "KMeans", **score_clustering(pca_data, kmeans_labels)},
        {"Model": "DBSCAN", **score_clustering(pca_data, dbscan_labels)},
        {"Model": "Hierarchical", **score_clustering(pca_data, hier_labels)},
    ]
)

results_display = results.copy()
for col in ["silhouette", "db_index", "ch_score"]:
    results_display[col] = results_display[col].round(4)

st.subheader("Model comparison")
st.dataframe(
    results_display.rename(
        columns={
            "silhouette": "Silhouette",
            "db_index": "Davies-Bouldin",
            "ch_score": "Calinski-Harabasz",
            "clusters": "Clusters",
            "noise_points": "Noise points",
        }
    ),
    use_container_width=True,
)

valid_results = results.dropna(subset=["silhouette"]).sort_values(
    by=["silhouette", "db_index", "ch_score"],
    ascending=[False, True, False],
)
best_model = valid_results.iloc[0]["Model"] if not valid_results.empty else None

if best_model:
    st.success(f"Best model by silhouette score: {best_model}")

model_to_labels = {
    "KMeans": kmeans_labels,
    "DBSCAN": dbscan_labels,
    "Hierarchical": hier_labels,
}

selected_model = st.selectbox(
    "Select model to visualize and profile",
    ["KMeans", "DBSCAN", "Hierarchical"],
    index=["KMeans", "DBSCAN", "Hierarchical"].index(best_model) if best_model else 0,
)

selected_labels = model_to_labels[selected_model]
final_df = cleaned_df.copy()
final_df["Cluster"] = selected_labels

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"{selected_model} cluster map")
    plot_clusters_2d(pca_plot, selected_labels, f"{selected_model} clusters on 2D PCA view")

with col_right:
    st.subheader("Cluster distribution")
    cluster_counts = pd.Series(selected_labels).value_counts().sort_index()
    st.bar_chart(cluster_counts)

st.subheader("Cluster profiles")
profiles = cluster_profiles(final_df, "Cluster")
st.dataframe(profiles, use_container_width=True)

if "Country" in final_df.columns:
    st.subheader("Country lookup")
    country = st.selectbox("Choose a country", sorted(final_df["Country"].dropna().astype(str).unique()))
    st.dataframe(final_df[final_df["Country"].astype(str) == country], use_container_width=True)

st.subheader("Hierarchical dendrogram")
plot_dendrogram(pca_data)

download_df = final_df.copy()
csv_bytes = download_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download clustered data as CSV",
    data=csv_bytes,
    file_name="world_development_clustered.csv",
    mime="text/csv",
)

with st.expander("Why this app looks professional", expanded=False):
    st.markdown(
        """
        - It cleans raw economic data automatically.
        - It aggregates repeated country rows into one country-level record.
        - It chooses PCA components using explained variance.
        - It compares three clustering models on the same PCA feature space.
        - It reports Silhouette, Davies-Bouldin, and Calinski-Harabasz metrics.
        - It includes a dendrogram, cluster profiles, and downloadable clustered output.
        """
    )
