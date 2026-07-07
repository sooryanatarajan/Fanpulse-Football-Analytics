"""
Modeling layer:
1. Demand forecasting - RandomForestRegressor predicting occupancy_rate/tickets_sold
2. Fan segmentation - KMeans clustering fans by behavior
3. Content engagement analysis
4. Campaign response propensity (simple classifier)
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
import json

D = "/home/claude/fifa_project/data"
matches = pd.read_csv(f"{D}/matches.csv")
tickets = pd.read_csv(f"{D}/ticket_sales.csv")
fans = pd.read_csv(f"{D}/fan_profiles.csv")
social = pd.read_csv(f"{D}/social_engagement.csv")
campaigns = pd.read_csv(f"{D}/campaign_performance.csv")
clickstream = pd.read_csv(f"{D}/clickstream.csv")

# ============================================================
# 1. DEMAND FORECASTING MODEL
# ============================================================
df = matches.merge(tickets.drop(columns=["capacity"]), on="match_id")
stage_map = {"Group Stage": 0, "Round of 16": 1, "Quarter Final": 2, "Semi Final": 3, "Final": 4}
dow_map = {d: i for i, d in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])}
df["stage_num"] = df["stage"].map(stage_map)
df["dow_num"] = df["day_of_week"].map(dow_map)

# social buzz per match as a feature
buzz = social.groupby("match_id").agg(
    total_engagement=("likes", "sum"),
    total_impressions=("impressions", "sum"),
    post_count=("post_id", "count")
).reset_index()
df = df.merge(buzz, on="match_id", how="left").fillna(0)

features = ["popularity_index", "capacity", "stage_num", "dow_num",
            "avg_ticket_price_usd", "total_engagement", "total_impressions", "post_count"]
target = "occupancy_rate"

X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
pred_test = rf.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
r2 = r2_score(y_test, pred_test)

# Predict demand for ALL matches (simulate "upcoming" forecast)
df["predicted_occupancy"] = rf.predict(X)
df["predicted_tickets"] = (df["predicted_occupancy"] * df["capacity"]).astype(int)
df["demand_tier"] = pd.cut(df["predicted_occupancy"], bins=[0, 0.6, 0.8, 0.92, 1.01],
                             labels=["Low", "Moderate", "High", "Very High"])

feature_importance = dict(zip(features, rf.feature_importances_.round(4)))

demand_output = df[["match_id", "home_team", "away_team", "stage", "match_date", "city",
                     "capacity", "tickets_sold", "occupancy_rate", "predicted_occupancy",
                     "predicted_tickets", "demand_tier"]].sort_values("predicted_occupancy", ascending=False)
demand_output.to_csv(f"{D}/demand_forecast.csv", index=False)

# ============================================================
# 2. FAN SEGMENTATION (behavior-based clustering)
# ============================================================
# Build behavior features per fan from clickstream + profile
click_agg = clickstream.groupby("fan_id").agg(
    sessions=("session_id", "count"),
    avg_session_seconds=("session_seconds", "mean"),
    conversions=("converted_to_purchase", "sum"),
).reset_index()

fan_feat = fans.merge(click_agg, on="fan_id", how="left").fillna(0)
fan_feat["conversion_rate"] = fan_feat["conversions"] / fan_feat["sessions"].replace(0, 1)

cluster_features = ["age", "years_following_fifa", "loyalty_points", "sessions",
                     "avg_session_seconds", "conversion_rate"]
Xc = fan_feat[cluster_features].fillna(0)
scaler = StandardScaler()
Xc_scaled = scaler.fit_transform(Xc)

k = 5
km = KMeans(n_clusters=k, random_state=42, n_init=10)
fan_feat["cluster"] = km.fit_predict(Xc_scaled)

# Label clusters by their characteristics (rank by engagement score)
cluster_profile = fan_feat.groupby("cluster")[cluster_features].mean()
cluster_profile["engagement_score"] = (
    cluster_profile["sessions"].rank() + cluster_profile["loyalty_points"].rank() +
    cluster_profile["conversion_rate"].rank() + cluster_profile["avg_session_seconds"].rank()
)
ranked = cluster_profile["engagement_score"].sort_values(ascending=False).index.tolist()

labels_by_rank = ["Super Fans", "High Engagers", "Casual Followers", "Digital Browsers", "Dormant / At-Risk"]
cluster_to_label = {cluster: labels_by_rank[i] for i, cluster in enumerate(ranked)}
fan_feat["segment"] = fan_feat["cluster"].map(cluster_to_label)

segment_summary = fan_feat.groupby("segment").agg(
    fan_count=("fan_id", "count"),
    avg_age=("age", "mean"),
    avg_sessions=("sessions", "mean"),
    avg_loyalty_points=("loyalty_points", "mean"),
    avg_conversion_rate=("conversion_rate", "mean"),
    avg_years_following=("years_following_fifa", "mean"),
).reset_index().round(2)
segment_summary["pct_of_base"] = (segment_summary["fan_count"] / len(fan_feat) * 100).round(1)

fan_feat[["fan_id", "city", "favorite_team", "age", "segment", "loyalty_points",
          "sessions", "conversion_rate"]].to_csv(f"{D}/fan_segments.csv", index=False)
segment_summary.to_csv(f"{D}/segment_summary.csv", index=False)

# City x segment for geo heatmap
geo_segment = fan_feat.groupby(["city", "segment"]).size().reset_index(name="count")
geo_segment.to_csv(f"{D}/geo_segment.csv", index=False)

# ============================================================
# 3. CONTENT ENGAGEMENT ANALYSIS
# ============================================================
content_perf = social.groupby(["content_type", "platform"]).agg(
    avg_engagement_rate=("engagement_rate", "mean"),
    total_impressions=("impressions", "sum"),
    total_likes=("likes", "sum"),
    total_shares=("shares", "sum"),
    post_count=("post_id", "count"),
).reset_index().round(4)
content_perf.to_csv(f"{D}/content_performance.csv", index=False)

content_summary = social.groupby("content_type").agg(
    avg_engagement_rate=("engagement_rate", "mean"),
    total_impressions=("impressions", "sum"),
).reset_index().sort_values("avg_engagement_rate", ascending=False).round(4)
content_summary.to_csv(f"{D}/content_summary.csv", index=False)

platform_summary = social.groupby("platform").agg(
    avg_engagement_rate=("engagement_rate", "mean"),
    total_impressions=("impressions", "sum"),
    total_video_views=("video_views", "sum"),
).reset_index().sort_values("avg_engagement_rate", ascending=False).round(4)
platform_summary.to_csv(f"{D}/platform_summary.csv", index=False)

# ============================================================
# 4. CAMPAIGN / OFFER RESPONSE PROPENSITY
# ============================================================
# Simulate whether a fan responds to offers, based on segment + past conversion
np.random.seed(1)
fan_feat["responds_to_offers"] = (
    (fan_feat["segment"].isin(["Super Fans", "High Engagers"])) &
    (np.random.rand(len(fan_feat)) < 0.75)
) | (np.random.rand(len(fan_feat)) < 0.08)

le_seg = LabelEncoder()
fan_feat["segment_enc"] = le_seg.fit_transform(fan_feat["segment"])
clf_features = ["age", "years_following_fifa", "loyalty_points", "sessions",
                "avg_session_seconds", "conversion_rate", "segment_enc"]
Xr = fan_feat[clf_features].fillna(0)
yr = fan_feat["responds_to_offers"].astype(int)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
clf.fit(Xr_tr, yr_tr)
clf_acc = accuracy_score(yr_te, clf.predict(Xr_te))
fan_feat["offer_response_probability"] = clf.predict_proba(Xr)[:, 1].round(3)
fan_feat[["fan_id", "segment", "offer_response_probability"]].sort_values(
    "offer_response_probability", ascending=False
).to_csv(f"{D}/offer_propensity.csv", index=False)

# ============================================================
# METRICS SUMMARY (for dashboard + PPT)
# ============================================================
metrics = {
    "demand_model": {
        "mae": round(float(mae), 4),
        "r2": round(float(r2), 3),
        "feature_importance": {k: float(v) for k, v in feature_importance.items()},
        "n_matches": int(len(df)),
    },
    "segmentation": {
        "n_fans": int(len(fan_feat)),
        "n_segments": k,
        "silhouette_note": "5 behavioral segments derived from engagement + loyalty features",
    },
    "offer_propensity_model": {
        "accuracy": round(float(clf_acc), 3)
    },
    "top_demand_matches": demand_output.head(5)[["match_id","home_team","away_team","stage","predicted_occupancy"]].to_dict("records"),
    "top_content_types": content_summary.head(3).to_dict("records"),
    "top_platforms": platform_summary.head(3).to_dict("records"),
    "segment_summary": segment_summary.to_dict("records"),
}
with open(f"{D}/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, default=str)

print("Demand model  -> MAE:", round(mae,4), " R2:", round(r2,3))
print("Offer model   -> Accuracy:", round(clf_acc,3))
print("Segments:")
print(segment_summary[["segment","fan_count","pct_of_base"]])
print("\nTop content types by engagement rate:")
print(content_summary.head(3))
print("\nTop platforms by engagement rate:")
print(platform_summary.head(3))
print("\nSaved all outputs to", D)
