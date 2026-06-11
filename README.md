# 🚚 Delivery ETA Optimization with Graph-Based Network Intelligence

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20Random%20Forest-green?style=for-the-badge)
![Graph Analytics](https://img.shields.io/badge/Graph%20Analytics-NetworkX-orange?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge)

### A graph-based logistics intelligence system for ETA prediction, bottleneck detection, corridor risk analysis, and FTL vs Carting decision support.

<br>

🔗 **Live Demo:** [delivery-eta-graph-intelligence.streamlit.app](https://delivery-eta-graph-intelligence.streamlit.app)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Business Problem](#-business-problem)
- [Core Idea](#-core-idea)
- [Live Streamlit Dashboard](#-live-streamlit-dashboard)
- [Key Results](#-key-results)
- [Repository Structure](#-repository-structure)
- [Dataset Description](#-dataset-description)
- [Methodology](#-methodology)
- [Modeling Results](#-modeling-results)
- [Graph Intelligence](#-graph-intelligence)
- [FTL vs Carting Recommendation Framework](#-ftl-vs-carting-recommendation-framework)
- [Business Impact Estimation](#-business-impact-estimation)
- [How to Run Locally](#-how-to-run-locally)
- [Tech Stack](#-tech-stack)
- [Data Leakage Prevention](#-data-leakage-prevention)
- [Dashboard Pages](#-dashboard-pages)
- [Future Improvements](#-future-improvements)

---

## 🚀 Project Overview

Delivery logistics companies operate through complex hub-and-spoke networks. Shipments move across multiple logistics centers and corridors before reaching the final destination.

Traditional ETA systems often depend heavily on map-based estimates like OSRM time. However, real logistics operations are affected by:

- Hub congestion
- Corridor-specific delays
- Route-type constraints
- Operational bottlenecks
- Long-distance uncertainty
- High-volume movement pressure
- SLA breach risk

This project builds a **graph-based logistics intelligence system** that improves delivery ETA prediction and supports operational decision-making.

The logistics network is modeled as a directed graph:

```text
Node = Logistics hub / delivery center
Edge = Source → destination corridor
```

The final system includes:

- Data cleaning and trip-corridor preparation
- Directed graph construction
- Bottleneck hub detection
- Corridor risk analysis
- Baseline ETA modeling
- Graph-enhanced ETA modeling
- Ablation testing
- FTL vs Carting recommendation framework
- Business impact estimation
- Streamlit dashboard deployment

---

## 🎯 Business Problem

A logistics operations team needs to answer:

1. Which hubs are acting as bottlenecks?
2. Which corridors are repeatedly delayed?
3. Can ETA prediction be improved using network-level features?
4. Which movements should be routed through FTL vs Carting?
5. What is the potential business impact of acting on high-risk routes?

This project converts raw trip data into a decision-support system for logistics operations.

---

## 🧠 Core Idea

A normal ETA model learns from trip-level features:

```text
OSRM time
Distance
Route type
Time features
```

This project adds graph intelligence:

```text
Corridor delay ratio
Corridor delay rate
Corridor trip volume
Hub centrality
PageRank
Betweenness centrality
Bottleneck score
Source and destination hub risk
```

This allows the model to capture operational patterns that normal tabular features miss.

---

## 🌐 Live Streamlit Dashboard

The project includes a deployed Streamlit dashboard.

🔗 **Live App:** [delivery-eta-graph-intelligence.streamlit.app](https://delivery-eta-graph-intelligence.streamlit.app)

The dashboard includes:

- Project overview
- Hub-to-hub ETA estimator
- ETA prediction demo
- FTL vs Carting recommendation dashboard
- Bottleneck hub ranking
- Corridor delay risk analysis
- Business impact estimation

---

## 📊 Key Results

| Metric | Baseline Model | Graph-Enhanced Model |
|---|---:|---:|
| Best Model | Random Forest | XGBoost |
| MAE | 42.58 | 35.90 |
| RMSE | 108.53 | 103.48 |
| R² | 0.923 | 0.930 |
| MAE Improvement | - | **15.7%** |

### Final Result

The graph-enhanced ETA model reduced average prediction error from **42.58 to 35.90**, achieving an approximate **15.7% reduction in MAE** compared to the strongest non-graph baseline.

---

## 🗂 Repository Structure

```text
delivery-eta-graph-intelligence/
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── delivery_data.csv
│   │
│   └── processed/
│       ├── trip_corridor_cleaned.csv
│       ├── corridor_summary.csv
│       ├── graph_test_enriched.csv
│       ├── node_features_basic.csv
│       ├── node_features_bottleneck.csv
│       └── X_test_ablation.csv
│
├── notebooks/
│   ├── 01_data_understanding_and_cleaning.ipynb
│   ├── 02_graph_construction_and_bottleneck_analysis.ipynb
│   ├── 03_eta_baseline_model.ipynb
│   ├── 04_graph_enhanced_eta_model.ipynb
│   └── 05_ftl_carting_recommendation_and_impact.ipynb
│
├── outputs/
│   ├── figures/
│   │
│   └── tables/
│       ├── baseline_model_results.csv
│       ├── baseline_error_by_route_type.csv
│       ├── graph_model_results.csv
│       ├── graph_vs_baseline_comparison.csv
│       ├── ablation_results.csv
│       ├── ftl_carting_recommendations_v2.csv
│       ├── ftl_carting_recommendation_summary_v2.csv
│       ├── business_impact_scenarios.csv
│       ├── risk_impact_summary.csv
│       └── recommendation_impact_summary.csv
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🗃 Dataset Description

The dataset contains logistics trip and route-level information.

Important columns include:

| Column | Description |
|---|---|
| `trip_uuid` | Unique trip identifier |
| `source_center` | Source hub ID |
| `source_name` | Source hub name |
| `destination_center` | Destination hub ID |
| `destination_name` | Destination hub name |
| `route_type` | FTL or Carting |
| `actual_time` | Actual observed movement time |
| `osrm_time` | OSRM/map-based estimated time |
| `actual_distance_to_destination` | Actual movement distance |
| `osrm_distance` | OSRM estimated distance |
| `delay_ratio` | actual_time / osrm_time |
| `is_delayed` | Delay label based on delay ratio |

The raw dataset contained repeated checkpoint-level records for the same trip. Therefore, it was transformed into a cleaned trip-corridor dataset.

Final cleaned unit:

```text
one row = one final trip movement between source hub and destination hub
```

---

## ⚙ Methodology

### 1. Data Understanding and Cleaning

The raw data was not one-row-per-trip. The same trip appeared multiple times due to checkpoint-level records.

To avoid overcounting, I created a cleaned trip-corridor dataset by grouping on:

```text
trip_uuid + source_center + destination_center
```

Then I selected the final cumulative movement record using maximum `actual_time`.

This produced a clean dataset suitable for graph construction and ETA modeling.

---

### 2. Graph Construction

A directed graph was built using NetworkX.

```text
Node = logistics hub
Edge = source → destination corridor
```

Each edge stores:

- Trip count
- Median actual time
- Median OSRM time
- Median distance
- Median delay ratio
- Delay rate
- Delayed trip count

Graph summary:

| Graph Metric | Value |
|---|---:|
| Nodes | 1,657 |
| Edges | 2,783 |
| Graph type | Directed |
| Graph density | 0.0010 |
| Isolated nodes | 0 |

The graph is sparse, which is expected in a hub-and-spoke logistics network.

---

### 3. Bottleneck Hub Analysis

The following graph metrics were calculated:

- In-degree
- Out-degree
- Weighted degree
- Betweenness centrality
- PageRank
- Hub delay rate
- Delayed trips handled
- Bottleneck score

The bottleneck score combines:

```text
traffic volume
centrality
PageRank
delay rate
delayed-trip exposure
```

Top bottleneck hub:

```text
IND000000ACB
```

This hub had the highest combination of traffic volume, connectivity, betweenness centrality, PageRank, and delayed-trip exposure.

---

### 4. Baseline ETA Modeling

Baseline models were trained using only non-graph features:

- OSRM time
- OSRM distance
- Actual distance
- Route type
- Hour
- Day of week
- Month

Models tested:

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost

Best baseline model:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 42.58 | 108.53 | 0.923 |

---

### 5. Graph-Enhanced ETA Modeling

Graph features were added to the modeling dataset:

- Corridor median delay ratio
- Corridor delay rate
- Corridor trip volume
- Source hub centrality
- Destination hub centrality
- Source bottleneck score
- Destination bottleneck score
- PageRank
- Betweenness centrality

To prevent data leakage, graph features were calculated using training data only and then merged into train and test data.

Best safer graph-enhanced model:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| XGBoost | 35.90 | 103.48 | 0.930 |

---

### 6. Ablation Study

An ablation test was performed by removing strong historical actual-time features:

```text
corridor_median_actual_time
corridor_mean_actual_time
```

This ensured the improvement was not only due to historical target aggregates.

Even after removing these features, graph-enhanced XGBoost outperformed the strongest baseline.

This confirmed that broader graph features such as corridor delay rate, corridor volume, hub bottleneck scores, PageRank, and centrality added meaningful predictive value.

---

## 📈 Modeling Results

### Baseline Models

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| Random Forest | 42.58 | 108.53 | 0.923 | 36.88% |
| XGBoost | 47.83 | 116.76 | 0.911 | 39.24% |
| Gradient Boosting | 49.75 | 119.53 | 0.907 | 41.84% |
| Linear Regression | 54.32 | 126.98 | 0.895 | 49.46% |

### Graph-Enhanced Ablation Models

| Model | MAE | RMSE | R² | MAPE |
|---|---:|---:|---:|---:|
| XGBoost | 35.90 | 103.48 | 0.930 | 26.84% |
| Random Forest | 36.31 | 98.14 | 0.937 | 26.61% |
| Gradient Boosting | 38.06 | 98.04 | 0.937 | 30.11% |
| Linear Regression | 48.48 | 110.12 | 0.921 | 44.18% |

Final selected model:

```text
Graph-enhanced XGBoost
```

Reason:

- Best MAE among safer ablation models
- Strong improvement over baseline
- Avoids over-reliance on historical actual-time target aggregates

---

## 🧭 Graph Intelligence

### Why graph features matter

A delivery movement does not exist in isolation. It belongs to a logistics network.

Two routes with the same OSRM time may behave differently if:

- One starts from a congested hub
- One ends at a bottleneck hub
- One corridor has high historical delay
- One corridor carries high trip volume
- One route is structurally important in the network

Graph intelligence captures these network-level patterns.

---

## 🚦 FTL vs Carting Recommendation Framework

A recommendation framework was built to support route-type decisions.

Inputs used:

- Predicted ETA
- Corridor delay risk
- Corridor delay rate
- Distance
- Source hub bottleneck score
- Destination hub bottleneck score

Recommendation logic:

| Condition | Recommendation |
|---|---|
| High risk | FTL |
| Low risk | Carting |
| Medium risk + long distance | FTL |
| Medium risk + short distance | Carting |

This creates a practical decision framework that balances:

```text
reliability vs cost efficiency
```

---

## 💼 Business Impact Estimation

The test set contained:

| Metric | Value |
|---|---:|
| Test movements | 5,274 |
| SLA breaches | 4,958 |
| SLA breach rate | 94.01% |
| Action-required movements | 2,834 |
| Action-required share | 53.74% |

Action-required movements were identified based on:

```text
high delay risk OR route-type recommendation change
```

Scenario-based impact estimates:

| Scenario | Delay Reduction Assumption | Delay Time Saved | SLA Breaches Reduced |
|---|---:|---:|---:|
| Conservative | 5% | 18,978 | 133 |
| Moderate | 10% | 37,955 | 266 |
| Optimistic | 15% | 56,933 | 399 |

These are scenario-based estimates, not guaranteed financial savings.

---

## 🖥 Streamlit Dashboard

The deployed dashboard contains the following pages:

### 1. Project Overview

Shows:

- Graph size
- Model results
- Key project outcome

### 2. Hub-to-Hub ETA Estimator

Allows users to select:

```text
Source hub
Destination hub
```

The app returns:

- Estimated delivery time
- OSRM time
- Delay ratio
- Historical trip count
- FTL vs Carting recommendation

If graph-enhanced prediction records are unavailable for a selected corridor, the app falls back to historical corridor median ETA.

### 3. ETA Prediction Demo

Displays graph-enhanced predicted ETA for saved test movements.

### 4. FTL vs Carting Recommendations

Shows:

- Recommended route-type distribution
- Delay risk distribution
- Sample recommendation table

### 5. Bottleneck Hubs

Ranks hubs by bottleneck score using graph and delay metrics.

### 6. Corridor Risk

Shows high-delay and high-volume delay-contributing corridors.

### 7. Business Impact

Shows conservative, moderate, and optimistic impact scenarios.

---

## 🧪 Tech Stack

| Category | Tools |
|---|---|
| Programming | Python |
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn, XGBoost |
| Graph Analytics | NetworkX |
| Visualization | Matplotlib, Plotly |
| Dashboard | Streamlit |
| Experimentation | Jupyter Notebook |
| Deployment | Streamlit Community Cloud |

---

## 🚀 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/delivery-eta-graph-intelligence.git
cd delivery-eta-graph-intelligence
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate on Windows:

```bash
venv\Scripts\activate
```

Activate on Mac/Linux:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

---

## 📦 requirements.txt

For deployment, the dashboard requires:

```text
streamlit
pandas
plotly
```

For running the full notebook pipeline, install:

```text
pandas
numpy
matplotlib
scikit-learn
xgboost
networkx
streamlit
plotly
joblib
```

---

## 🔒 Data Leakage Prevention

This project explicitly checks for data leakage.

Graph features were calculated only after the time-based train-test split.

Correct process:

```text
1. Split data by time
2. Build graph features using training data only
3. Merge train-derived graph features into train and test
4. Evaluate on future test data
```

This ensures future test-period delay information is not used during training.

An ablation study was also performed to remove historical actual-time features and verify that graph-based features still improved ETA prediction.

---

## 📌 Important Notes

- The deployed Streamlit app uses precomputed prediction outputs instead of loading a `.pkl` model.  
- This avoids model serialization issues across different scikit-learn versions.
- The hub-to-hub ETA estimator uses graph-enhanced predictions when available.
- If graph-enhanced predictions are unavailable, the app falls back to historical corridor-level estimates.
- Business impact values are scenario-based and should not be interpreted as guaranteed savings.
- The dashboard is designed as a decision-support prototype, not a live production ETA API.

---

## 🧾 Final Business Insights

1. `IND000000ACB` was identified as the most critical bottleneck hub.
2. High-volume delayed corridors should be prioritized over low-volume extreme-delay outliers.
3. Graph-based features improved ETA prediction beyond normal trip-level features.
4. FTL recommendations were concentrated on higher-risk and longer-distance movements.
5. Under a moderate 10% improvement scenario, the system could save approximately 37,955 delay-time units and reduce around 266 SLA breaches.

---

## 🔮 Future Improvements

- Add live traffic API integration
- Add weather and holiday features
- Add real shipment cost and SLA penalty data
- Build a real-time ETA prediction API
- Add alternative route simulation
- Add geospatial hub visualization
- Experiment with Node2Vec embeddings
- Experiment with Graph Neural Networks
- Add model monitoring and drift detection

---

### ⭐ If you found this project useful, consider starring the repository.

</div>
