from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="Country Development Clustering App",
    page_icon="🌍",
    layout="wide",
)

SAMPLE_PATH = Path(__file__).parent / "World_development_mesurement.xlsx"


@st.cache_data
def load_sample_data() -> pd.DataFrame:
    return pd.read_excel(SAMPLE_PATH)


def clean_dataset(df: pd.DataFrame, null_threshold: float) -> tuple[pd.DataFrame, list[str], list[str]]:
    cleaned = df.copy()

    cleaned.columns = [str(c).strip() for c in cleaned.columns]

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

    for col in cleaned.columns:
        if col == country_col:
            continue
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    null_ratio = cleaned.isna().mean()
    dropped_cols = null_ratio[null_ratio > null_threshold].index.tolist()
    if country_col in dropped_cols:
        dropped_cols.remove(country_col)

    cleaned = cleaned.drop(columns=dropped_cols, errors="ignore")
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    numeric_cols = cleaned.select_dtypes(include=np.number).columns.tolist()
    cleaned[numeric_cols] = cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
    cleaned[numeric_cols] = cleaned[numeric_cols].fillna(cleaned[numeric_cols].median(numeric_only=True))

    id_cols = [country_col] if country_col else []

    return cleaned, numeric_cols, id_cols


def choose_pca_components(scaled_data: np.ndarray, variance_target: float):
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


def plot_clusters_2d(X_2d: np.ndarray, labels: np.ndarray, title: str, selected_country_point=None):
    fig, ax = plt.subplots(figsize=(9, 5))
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
                s=55,
                edgecolors="black",
                label=f"Cluster {label}",
            )

    if selected_country_point is not None:
        ax.scatter(
            selected_country_point[0],
            selected_country_point[1],
            s=220,
            marker="*",
            edgecolors="black",
            label="Selected Country",
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


def get_matching_columns(columns, keywords):
    matched = []
    for col in columns:
        cl = col.lower()
        if any(k in cl for k in keywords):
            matched.append(col)
    return list(dict.fromkeys(matched))


def create_development_score(df_numeric: pd.DataFrame) -> pd.Series:
    cols = df_numeric.columns.tolist()

    positive_keywords = [
        "gdp", "income", "life", "health", "tourism", "internet",
        "literacy", "education", "urban", "employment", "exports"
    ]
    negative_keywords = [
        "mortality", "death", "fertility", "poverty", "birth", "co2",
        "inflation", "unemployment"
    ]

    positive_cols = get_matching_columns(cols, positive_keywords)
    negative_cols = get_matching_columns(cols, negative_keywords)

    z = (df_numeric - df_numeric.mean()) / df_numeric.std(ddof=0)
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0)

    score = pd.Series(0, index=df_numeric.index, dtype=float)

    if positive_cols:
        score += z[positive_cols].mean(axis=1)
    if negative_cols:
        score -= z[negative_cols].mean(axis=1)

    return score


def label_clusters_by_development(df_final: pd.DataFrame, cluster_col: str = "Cluster"):
    numeric_df = df_final.select_dtypes(include=np.number).drop(columns=[cluster_col], errors="ignore")
    df_temp = df_final.copy()
    df_temp["Development_Score"] = create_development_score(numeric_df)

    cluster_rank = (
        df_temp.groupby(cluster_col)["Development_Score"]
        .mean()
        .sort_values()
        .index
        .tolist()
    )

    label_names = ["Underdeveloped", "Developing", "Developed"]

    if len(cluster_rank) == 2:
        label_names = ["Developing", "Developed"]
    elif len(cluster_rank) > 3:
        label_names = [f"Level {i+1}" for i in range(len(cluster_rank))]

    mapping = {cluster: label_names[i] for i, cluster in enumerate(cluster_rank)}
    df_final["Cluster_Name"] = df_final[cluster_col].map(mapping)

    return df_final, mapping


def get_country_dashboard_columns(df: pd.DataFrame):
    preferred = []
    target_names = [
        "GDP", "Life Expectancy", "Health Exp/Capita", "CO2 Emissions",
        "Income", "Population", "Tourism Inbound", "Tourism Outbound"
    ]

    for key in target_names:
        for col in df.columns:
            if col.lower() == key.lower():
                preferred.append(col)

    if not preferred:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        preferred = numeric_cols[:6]

    return preferred[:6]


def plot_world_map(df: pd.DataFrame):
    if "Country" not in df.columns or "Cluster_Name" not in df.columns:
        st.warning("Country or Cluster_Name column not found for world map.")
        return

    map_df = df.copy()

    map_df["Country"] = map_df["Country"].replace({
        "United States of America": "United States",
        "Virgin Islands (U.S.)": "United States Virgin Islands",
        "Congo, Dem. Rep.": "Democratic Republic of the Congo",
        "Congo, Rep.": "Republic of the Congo",
        "Kyrgyz Republic": "Kyrgyzstan",
        "Slovak Republic": "Slovakia",
        "Russian Federation": "Russia",
        "Egypt, Arab Rep.": "Egypt",
        "Iran, Islamic Rep.": "Iran",
        "Yemen, Rep.": "Yemen",
        "Venezuela, RB": "Venezuela",
        "Gambia, The": "Gambia",
        "Bahamas, The": "Bahamas",
        "Brunei Darussalam": "Brunei",
        "Hong Kong SAR, China": "Hong Kong",
        "Macao SAR, China": "Macao",
        "Korea, Rep.": "South Korea",
        "Korea, Dem. People's Rep.": "North Korea",
        "Lao PDR": "Laos",
    })

    fig = px.choropleth(
        map_df,
        locations="Country",
        locationmode="country names",
        color="Cluster_Name",
        hover_name="Country",
        title="World Development Cluster Map",
        color_discrete_map={
            "Underdeveloped": "red",
            "Developing": "orange",
            "Developed": "green"
        }
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type="equirectangular"),
        height=550
    )

    st.plotly_chart(fig, use_container_width=True)


st.title("🌍 Country Development Clustering App")
st.caption("Interactive clustering and country development dashboard")

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

if "Country" in cleaned_df.columns:
    cleaned_df = cleaned_df.groupby("Country").mean(numeric_only=True).reset_index()

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
final_df, cluster_name_mapping = label_clusters_by_development(final_df, "Cluster")

selected_country = None
selected_country_row = None
selected_country_point = None

if "Country" in final_df.columns:
    st.subheader("Select Country")
    selected_country = st.selectbox("Choose a country", sorted(final_df["Country"].dropna().astype(str).unique()))
    selected_country_row = final_df[final_df["Country"].astype(str) == selected_country].iloc[0]

    country_index = final_df[final_df["Country"].astype(str) == selected_country].index[0]
    selected_country_point = pca_plot[country_index]

    st.success(
        f"{selected_country} is classified as: {selected_country_row['Cluster_Name']} "
        f"(Cluster {selected_country_row['Cluster']})"
    )

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader(f"{selected_model} cluster map")
    plot_clusters_2d(
        pca_plot,
        selected_labels,
        f"{selected_model} clusters on 2D PCA view",
        selected_country_point=selected_country_point,
    )

with col_right:
    st.subheader("Cluster distribution")
    cluster_counts = pd.Series(selected_labels).value_counts().sort_index()
    st.bar_chart(cluster_counts)

st.subheader("Cluster profiles")
profiles = cluster_profiles(final_df, "Cluster")
profiles["Cluster Name"] = profiles.index.map(cluster_name_mapping)
st.dataframe(profiles, use_container_width=True)

st.subheader("World Map by Development Category")
plot_world_map(final_df)

if selected_country_row is not None:
    st.subheader(f"{selected_country} Dashboard")

    cluster_num = int(selected_country_row["Cluster"])
    cluster_name = selected_country_row["Cluster_Name"]
    cluster_avg = final_df[final_df["Cluster"] == cluster_num].mean(numeric_only=True)
    overall_avg = final_df.mean(numeric_only=True)

    metric_cols = get_country_dashboard_columns(final_df)
    top1, top2, top3 = st.columns(3)

    if len(metric_cols) >= 1:
        top1.metric(metric_cols[0], f"{selected_country_row[metric_cols[0]]:.2f}")
    if len(metric_cols) >= 2:
        top2.metric(metric_cols[1], f"{selected_country_row[metric_cols[1]]:.2f}")
    if len(metric_cols) >= 3:
        top3.metric(metric_cols[2], f"{selected_country_row[metric_cols[2]]:.2f}")

    top4, top5, top6 = st.columns(3)
    if len(metric_cols) >= 4:
        top4.metric(metric_cols[3], f"{selected_country_row[metric_cols[3]]:.2f}")
    if len(metric_cols) >= 5:
        top5.metric(metric_cols[4], f"{selected_country_row[metric_cols[4]]:.2f}")
    if len(metric_cols) >= 6:
        top6.metric(metric_cols[5], f"{selected_country_row[metric_cols[5]]:.2f}")

    st.markdown(
        f"""
        **Development Category:** {cluster_name}  
        **Cluster Number:** {cluster_num}
        """
    )

    compare_cols = metric_cols[:5]
    compare_df = pd.DataFrame({
        "Indicator": compare_cols,
        selected_country: [selected_country_row[col] for col in compare_cols],
        "Cluster Average": [cluster_avg[col] for col in compare_cols],
        "Overall Average": [overall_avg[col] for col in compare_cols],
    })

    st.subheader("Country vs Cluster Average")
    st.dataframe(compare_df.round(2), use_container_width=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(compare_cols))
    width = 0.25

    ax.bar(x - width, compare_df[selected_country], width, label=selected_country)
    ax.bar(x, compare_df["Cluster Average"], width, label="Cluster Avg")
    ax.bar(x + width, compare_df["Overall Average"], width, label="Overall Avg")

    ax.set_xticks(x)
    ax.set_xticklabels(compare_cols, rotation=25, ha="right")
    ax.set_title(f"{selected_country} vs Cluster vs Overall")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    st.pyplot(fig)

    detail_cols = ["Country", "Cluster", "Cluster_Name"] + metric_cols
    detail_cols = [c for c in detail_cols if c in final_df.columns]
    st.subheader(f"{selected_country} Details")
    st.dataframe(pd.DataFrame([selected_country_row[detail_cols]]), use_container_width=True)

st.subheader("Hierarchical dendrogram")
plot_dendrogram(pca_data)

download_df = final_df.copy()
csv_bytes = download_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download clustered data as CSV",
    data=csv_bytes,
    file_name="country_development_clustered.csv",
    mime="text/csv",
)

with st.expander("Why this app looks strong", expanded=False):
    st.markdown(
        """
        - It cleans raw data automatically.
        - It aggregates repeated country rows into one record per country.
        - It chooses PCA components using explained variance.
        - It compares KMeans, DBSCAN, and Hierarchical clustering.
        - It labels clusters into development categories.
        - It gives a dashboard for the selected country.
        - It includes a world map, cluster profiles, visualization, and CSV download.
        """
    )
