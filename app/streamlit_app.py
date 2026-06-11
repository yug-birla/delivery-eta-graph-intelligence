# ============================================================
# Streamlit App: Delivery ETA Graph Intelligence
# ============================================================

import os
import pandas as pd
import streamlit as st
import plotly.express as px


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title="Delivery ETA Graph Intelligence",
    page_icon="🚚",
    layout="wide"
)

st.title("🚚 Delivery ETA Optimization with Graph-Based Network Intelligence")

st.markdown(
    """
    This dashboard demonstrates a graph-based logistics intelligence system for:

    - Hub-to-hub ETA estimation
    - ETA prediction analysis
    - Bottleneck hub detection
    - Corridor delay risk analysis
    - FTL vs Carting recommendation
    - Business impact estimation
    """
)


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "tables")

TRIP_DATA_PATH = os.path.join(DATA_DIR, "trip_corridor_cleaned.csv")
CORRIDOR_SUMMARY_PATH = os.path.join(DATA_DIR, "corridor_summary.csv")
NODE_FEATURES_PATH = os.path.join(DATA_DIR, "node_features_bottleneck.csv")
GRAPH_TEST_PATH = os.path.join(DATA_DIR, "graph_test_enriched.csv")

RECOMMENDATION_PATH = os.path.join(OUTPUT_DIR, "ftl_carting_recommendations_v2.csv")
IMPACT_PATH = os.path.join(OUTPUT_DIR, "business_impact_scenarios.csv")
RISK_IMPACT_PATH = os.path.join(OUTPUT_DIR, "risk_impact_summary.csv")
REC_IMPACT_PATH = os.path.join(OUTPUT_DIR, "recommendation_impact_summary.csv")


# ============================================================
# Load data
# ============================================================

@st.cache_data
def load_csv(path):
    return pd.read_csv(path)


try:
    trip_df = load_csv(TRIP_DATA_PATH)
    corridor_summary = load_csv(CORRIDOR_SUMMARY_PATH)
    node_features = load_csv(NODE_FEATURES_PATH)
    graph_test_enriched = load_csv(GRAPH_TEST_PATH)

    recommendations = load_csv(RECOMMENDATION_PATH)
    impact_summary = load_csv(IMPACT_PATH)
    risk_impact_summary = load_csv(RISK_IMPACT_PATH)
    recommendation_impact_summary = load_csv(REC_IMPACT_PATH)

except FileNotFoundError as e:
    st.error(f"Missing file: {e}")
    st.stop()


# ============================================================
# Helper functions
# ============================================================

def safe_round(value, digits=2):
    try:
        return round(float(value), digits)
    except Exception:
        return value


def get_existing_columns(df, cols):
    return [col for col in cols if col in df.columns]


def build_center_label_maps(df):
    """
    Creates readable hub labels:
    Hub Name | Hub ID
    """

    source_map = pd.DataFrame()
    destination_map = pd.DataFrame()

    if {"source_center", "source_name"}.issubset(df.columns):
        source_map = (
            df[["source_center", "source_name"]]
            .dropna()
            .drop_duplicates()
            .rename(columns={"source_center": "center", "source_name": "name"})
        )

    if {"destination_center", "destination_name"}.issubset(df.columns):
        destination_map = (
            df[["destination_center", "destination_name"]]
            .dropna()
            .drop_duplicates()
            .rename(columns={"destination_center": "center", "destination_name": "name"})
        )

    center_map_df = pd.concat([source_map, destination_map], axis=0)

    if center_map_df.empty:
        all_centers = pd.concat(
            [
                corridor_summary["source_center"],
                corridor_summary["destination_center"]
            ],
            axis=0
        ).dropna().unique()

        id_to_label = {center: str(center) for center in all_centers}

    else:
        center_map_df = center_map_df.drop_duplicates(subset=["center"])

        id_to_label = {
            row["center"]: f"{row['name']} | {row['center']}"
            for _, row in center_map_df.iterrows()
        }

    label_to_id = {
        label: center_id
        for center_id, label in id_to_label.items()
    }

    return id_to_label, label_to_id


id_to_label, label_to_id = build_center_label_maps(trip_df)


def center_label(center_id):
    return id_to_label.get(center_id, str(center_id))


def make_corridor_label(source_center, destination_center):
    return f"{center_label(source_center)}  →  {center_label(destination_center)}"


def classify_historical_risk(delay_ratio):
    if delay_ratio >= 2:
        return "High"
    elif delay_ratio >= 1.2:
        return "Medium"
    else:
        return "Low"


def fallback_recommendation(delay_ratio, distance):
    risk = classify_historical_risk(delay_ratio)
    median_distance = corridor_summary["median_distance"].median()

    if risk == "High":
        return "FTL", risk
    elif risk == "Medium" and distance >= median_distance:
        return "FTL", risk
    else:
        return "Carting", risk


# ============================================================
# Add readable labels to dataframes
# ============================================================

# Corridor summary labels
if {"source_center", "destination_center"}.issubset(corridor_summary.columns):
    corridor_summary["source_hub"] = corridor_summary["source_center"].apply(center_label)
    corridor_summary["destination_hub"] = corridor_summary["destination_center"].apply(center_label)
    corridor_summary["corridor_label"] = corridor_summary.apply(
        lambda row: make_corridor_label(row["source_center"], row["destination_center"]),
        axis=1
    )

# Recommendation labels
if {"source_center", "destination_center"}.issubset(recommendations.columns):
    recommendations["source_hub"] = recommendations["source_center"].apply(center_label)
    recommendations["destination_hub"] = recommendations["destination_center"].apply(center_label)
    recommendations["corridor_label"] = recommendations.apply(
        lambda row: make_corridor_label(row["source_center"], row["destination_center"]),
        axis=1
    )

# Graph test labels
if {"source_center", "destination_center"}.issubset(graph_test_enriched.columns):
    graph_test_enriched["source_hub"] = graph_test_enriched["source_center"].apply(center_label)
    graph_test_enriched["destination_hub"] = graph_test_enriched["destination_center"].apply(center_label)
    graph_test_enriched["corridor_label"] = graph_test_enriched.apply(
        lambda row: make_corridor_label(row["source_center"], row["destination_center"]),
        axis=1
    )

# Node labels
if "center" in node_features.columns:
    node_features["hub"] = node_features["center"].apply(center_label)


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Project Overview",
        "Hub-to-Hub ETA Estimator",
        "ETA Prediction Demo",
        "FTL vs Carting Recommendations",
        "Bottleneck Hubs",
        "Corridor Risk",
        "Business Impact"
    ]
)


# ============================================================
# Page 1: Project Overview
# ============================================================

if page == "Project Overview":
    st.header("📌 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Graph Nodes", "1,657")

    with col2:
        st.metric("Graph Edges", "2,783")

    with col3:
        st.metric("Baseline MAE", "42.58")

    with col4:
        st.metric("Graph Model MAE", "35.90")

    st.markdown("### Key Result")

    st.success(
        "Graph-enhanced XGBoost reduced ETA MAE from 42.58 to 35.90, "
        "an approximate 15.7% improvement over the strongest baseline."
    )

    st.markdown("### Project Summary")

    st.write(
        """
        This project models a logistics delivery system as a directed graph.

        - Nodes represent logistics hubs or centers.
        - Edges represent source-destination corridors.
        - Graph features capture corridor delay, hub centrality, bottleneck exposure, and route risk.
        - The final output supports ETA estimation, FTL vs Carting decisions, and operational prioritization.
        """
    )


# ============================================================
# Page 2: Hub-to-Hub ETA Estimator
# ============================================================

elif page == "Hub-to-Hub ETA Estimator":
    st.header("📍 Hub-to-Hub ETA Estimator")

    st.markdown(
        """
        Select a source hub and destination hub to estimate delivery time.

        The app first checks whether graph-enhanced prediction records exist for the selected corridor.
        If not, it falls back to historical corridor-level ETA using median actual time.
        """
    )

    source_hub_ids = sorted(corridor_summary["source_center"].dropna().unique())
    source_hub_labels = [center_label(center_id) for center_id in source_hub_ids]

    selected_source_label = st.selectbox(
        "Select source hub",
        source_hub_labels
    )

    selected_source = source_hub_ids[source_hub_labels.index(selected_source_label)]

    possible_destination_ids = sorted(
        corridor_summary[
            corridor_summary["source_center"] == selected_source
        ]["destination_center"].dropna().unique()
    )

    if len(possible_destination_ids) == 0:
        st.warning("No destination hubs found for this source hub.")
        st.stop()

    possible_destination_labels = [
        center_label(center_id) for center_id in possible_destination_ids
    ]

    selected_destination_label = st.selectbox(
        "Select destination hub",
        possible_destination_labels
    )

    selected_destination = possible_destination_ids[
        possible_destination_labels.index(selected_destination_label)
    ]

    selected_corridor_data = corridor_summary[
        (corridor_summary["source_center"] == selected_source) &
        (corridor_summary["destination_center"] == selected_destination)
    ].copy()

    if selected_corridor_data.empty:
        st.error("No direct corridor found between these two hubs.")
        st.stop()

    corridor_row = selected_corridor_data.iloc[0]
    selected_corridor = corridor_row["corridor"]

    st.markdown("### Selected Corridor")

    st.info(
        make_corridor_label(selected_source, selected_destination)
    )

    historical_eta = corridor_row["median_actual_time"]
    historical_osrm_time = corridor_row["median_osrm_time"]
    historical_delay_ratio = corridor_row["median_delay_ratio"]
    historical_distance = corridor_row["median_distance"]
    trips = corridor_row["trips"]

    matching_predictions = recommendations[
        recommendations["corridor"] == selected_corridor
    ].copy()

    prediction_available = not matching_predictions.empty

    if prediction_available:
        estimated_time = matching_predictions["predicted_actual_time"].median()
        eta_source = "Graph-enhanced prediction"
    else:
        estimated_time = historical_eta
        eta_source = "Historical corridor median"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Estimated Delivery Time", safe_round(estimated_time, 2))

    with col2:
        st.metric("OSRM Time", safe_round(historical_osrm_time, 2))

    with col3:
        st.metric("Delay Ratio", safe_round(historical_delay_ratio, 2))

    with col4:
        st.metric("Historical Trips", int(trips))

    st.caption(f"ETA source: {eta_source}")

    st.markdown("### Prediction and Recommendation")

    if prediction_available:
        median_predicted_time = matching_predictions["predicted_actual_time"].median()
        median_actual_time = matching_predictions["actual_time"].median()
        median_distance = matching_predictions["actual_distance_to_destination"].median()
        median_risk_score = matching_predictions["delay_risk_score"].median()

        risk_category = matching_predictions["delay_risk_category_v2"].mode()[0]
        recommended_route = matching_predictions["recommended_route_type_v2"].mode()[0]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Median Predicted Time", safe_round(median_predicted_time, 2))

        with col2:
            st.metric("Median Actual Time", safe_round(median_actual_time, 2))

        with col3:
            st.metric("Median Distance", safe_round(median_distance, 2))

        with col4:
            st.metric("Risk Score", safe_round(median_risk_score, 3))

        if recommended_route == "FTL":
            st.warning(
                f"Recommended Route Type: **{recommended_route}** | Risk Category: **{risk_category}**"
            )
        else:
            st.success(
                f"Recommended Route Type: **{recommended_route}** | Risk Category: **{risk_category}**"
            )

    else:
        recommended_route, historical_risk = fallback_recommendation(
            historical_delay_ratio,
            historical_distance
        )

        st.warning(
            "Graph-enhanced prediction record is not available for this corridor in the saved test prediction output. "
            "The estimate below uses historical corridor behavior."
        )

        if recommended_route == "FTL":
            st.warning(
                f"Fallback Recommendation: **{recommended_route}** | Historical Risk: **{historical_risk}**"
            )
        else:
            st.success(
                f"Fallback Recommendation: **{recommended_route}** | Historical Risk: **{historical_risk}**"
            )

    st.markdown("### Corridor Details")

    detail_cols = [
        "corridor_label",
        "source_hub",
        "destination_hub",
        "corridor",
        "trips",
        "median_actual_time",
        "mean_actual_time",
        "median_osrm_time",
        "mean_osrm_time",
        "median_distance",
        "median_osrm_distance",
        "median_delay_ratio",
        "mean_delay_ratio",
        "delay_rate",
        "delayed_trips"
    ]

    available_detail_cols = get_existing_columns(selected_corridor_data, detail_cols)

    st.dataframe(
        selected_corridor_data[available_detail_cols],
        use_container_width=True
    )

    st.markdown("### Interpretation")

    if historical_delay_ratio > 1.2:
        st.write(
            f"This corridor is historically delayed. Its median delay ratio is "
            f"{safe_round(historical_delay_ratio, 2)}, meaning actual travel time is usually higher than OSRM time."
        )
    else:
        st.write(
            f"This corridor is relatively stable. Its median delay ratio is "
            f"{safe_round(historical_delay_ratio, 2)}."
        )

    st.write(
        "This is a historical/graph-based estimator. It is not connected to live GPS, weather, or traffic APIs."
    )


# ============================================================
# Page 3: ETA Prediction Demo
# ============================================================

elif page == "ETA Prediction Demo":
    st.header("⏱️ ETA Prediction Demo")

    st.write(
        "Select a saved test movement to view graph-enhanced ETA prediction and route context."
    )

    sample_size = len(recommendations)

    selected_index = st.slider(
        "Select test record index",
        min_value=0,
        max_value=sample_size - 1,
        value=0
    )

    selected_row = recommendations.iloc[[selected_index]]

    predicted_time = selected_row["predicted_actual_time"].values[0]
    actual_time = selected_row["actual_time"].values[0]
    osrm_time = selected_row["osrm_time"].values[0]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("OSRM Time", safe_round(osrm_time, 2))

    with col2:
        st.metric("Predicted Actual Time", safe_round(predicted_time, 2))

    with col3:
        st.metric("Actual Time", safe_round(actual_time, 2))

    st.markdown("### Selected Movement Details")

    show_cols = [
        "source_hub",
        "destination_hub",
        "corridor_label",
        "source_center",
        "destination_center",
        "corridor",
        "route_type",
        "recommended_route_type_v2",
        "delay_risk_category_v2",
        "delay_risk_score",
        "actual_distance_to_destination",
        "corridor_median_delay_ratio",
        "corridor_delay_rate",
        "source_bottleneck_score",
        "destination_bottleneck_score"
    ]

    available_cols = get_existing_columns(selected_row, show_cols)

    st.dataframe(selected_row[available_cols], use_container_width=True)

    delay_gap = predicted_time - osrm_time

    st.markdown("### Prediction Interpretation")

    if delay_gap > 0:
        st.warning(
            f"The predicted actual time is {safe_round(delay_gap, 2)} units higher than OSRM time, "
            "indicating expected delay beyond map-based travel time."
        )
    else:
        st.success(
            "The predicted actual time is close to or below OSRM time for this movement."
        )


# ============================================================
# Page 4: FTL vs Carting Recommendations
# ============================================================

elif page == "FTL vs Carting Recommendations":
    st.header("🚦 FTL vs Carting Recommendation Framework")

    st.markdown(
        """
        The recommendation framework uses delay risk, predicted ETA, distance,
        corridor history, and hub bottleneck exposure.
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        rec_counts = recommendations["recommended_route_type_v2"].value_counts().reset_index()
        rec_counts.columns = ["Recommended Route Type", "Count"]

        fig = px.bar(
            rec_counts,
            x="Recommended Route Type",
            y="Count",
            title="Recommended Route Type Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        risk_counts = recommendations["delay_risk_category_v2"].value_counts().reset_index()
        risk_counts.columns = ["Risk Category", "Count"]

        fig = px.bar(
            risk_counts,
            x="Risk Category",
            y="Count",
            title="Delay Risk Category Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Recommendation Table")

    recommendation_cols = [
        "source_hub",
        "destination_hub",
        "corridor_label",
        "route_type",
        "recommended_route_type_v2",
        "delay_risk_category_v2",
        "delay_risk_score",
        "predicted_actual_time",
        "actual_time",
        "osrm_time",
        "actual_distance_to_destination"
    ]

    available_recommendation_cols = get_existing_columns(
        recommendations,
        recommendation_cols
    )

    st.dataframe(
        recommendations[available_recommendation_cols].head(200),
        use_container_width=True
    )

    st.info(
        "High-risk movements are recommended for FTL. Low-risk movements are recommended for Carting. "
        "Medium-risk movements are split using distance."
    )


# ============================================================
# Page 5: Bottleneck Hubs
# ============================================================

elif page == "Bottleneck Hubs":
    st.header("🧭 Bottleneck Hub Analysis")

    required_cols = [
        "hub",
        "center",
        "weighted_total_degree",
        "betweenness_centrality",
        "pagerank",
        "hub_delay_rate",
        "total_delayed_trips_handled",
        "bottleneck_score"
    ]

    available_cols = get_existing_columns(node_features, required_cols)

    top_hubs = node_features.sort_values(
        "bottleneck_score",
        ascending=False
    ).head(10)

    st.markdown("### Top 10 Bottleneck Hubs")

    st.dataframe(
        top_hubs[available_cols],
        use_container_width=True
    )

    fig = px.bar(
        top_hubs,
        x="bottleneck_score",
        y="hub",
        orientation="h",
        title="Top 10 Bottleneck Hubs by Bottleneck Score"
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})

    st.plotly_chart(fig, use_container_width=True)

    top_hub_name = top_hubs.iloc[0]["hub"]

    st.info(
        f"{top_hub_name} was identified as the strongest bottleneck hub based on "
        "traffic volume, centrality, PageRank, and delayed-trip exposure."
    )


# ============================================================
# Page 6: Corridor Risk
# ============================================================

elif page == "Corridor Risk":
    st.header("🛣️ Corridor Delay Risk Analysis")

    min_trips = st.slider(
        "Minimum trips for corridor ranking",
        min_value=1,
        max_value=100,
        value=20
    )

    filtered_corridors = corridor_summary[
        corridor_summary["trips"] >= min_trips
    ].copy()

    if filtered_corridors.empty:
        st.warning("No corridors found for this minimum trip threshold.")
    else:
        top_corridors = filtered_corridors.sort_values(
            ["delayed_trips", "median_delay_ratio"],
            ascending=[False, False]
        ).head(15)

        st.markdown("### Top Delay-Contributing Corridors")

        corridor_cols = [
            "corridor_label",
            "source_hub",
            "destination_hub",
            "trips",
            "median_actual_time",
            "median_osrm_time",
            "median_delay_ratio",
            "delay_rate",
            "delayed_trips"
        ]

        available_corridor_cols = get_existing_columns(
            top_corridors,
            corridor_cols
        )

        st.dataframe(
            top_corridors[available_corridor_cols],
            use_container_width=True
        )

        fig = px.bar(
            top_corridors,
            x="delayed_trips",
            y="corridor_label",
            orientation="h",
            title="Top Corridors by Delayed Trips"
        )

        fig.update_layout(yaxis={"categoryorder": "total ascending"})

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page 7: Business Impact
# ============================================================

elif page == "Business Impact":
    st.header("💼 Business Impact Estimation")

    st.markdown(
        """
        Business impact is estimated using scenario-based assumptions.
        These are not guaranteed savings, but operational what-if estimates.
        """
    )

    st.markdown("### Scenario-Based Impact")

    st.dataframe(impact_summary, use_container_width=True)

    if "estimated_delay_time_saved" in impact_summary.columns:
        fig = px.bar(
            impact_summary,
            x="scenario",
            y="estimated_delay_time_saved",
            title="Estimated Delay Time Saved by Scenario"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Impact by Risk Category")
    st.dataframe(risk_impact_summary, use_container_width=True)

    st.markdown("### Impact by Recommended Route Type")
    st.dataframe(recommendation_impact_summary, use_container_width=True)

    if "scenario" in impact_summary.columns:
        moderate_rows = impact_summary[
            impact_summary["scenario"] == "Moderate"
        ]

        if len(moderate_rows) > 0:
            moderate_row = moderate_rows.iloc[0]

            st.success(
                f"Under the moderate 10% improvement scenario, the system estimates "
                f"{safe_round(moderate_row['estimated_delay_time_saved'], 2)} delay-time units saved "
                f"and around {safe_round(moderate_row['estimated_sla_breaches_reduced'], 0)} SLA breaches reduced."
            )