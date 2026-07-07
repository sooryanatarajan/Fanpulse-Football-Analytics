"""
Synthetic data generator for FIFA Fan Analytics Platform.
Produces: matches, ticket_sales, attendance, social_engagement,
campaign_performance, fan_profiles, clickstream.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

np.random.seed(42)

TEAMS = ["Brazil", "Argentina", "France", "Germany", "England", "Spain",
         "Portugal", "Netherlands", "Italy", "Belgium", "Croatia", "Morocco",
         "USA", "Japan", "Mexico", "India"]  # India included for local flavor

STADIUMS = [
    ("Lusail Stadium", 88000, "Lusail"),
    ("Al Bayt Stadium", 60000, "Al Khor"),
    ("Ahmad Bin Ali", 44740, "Al Rayyan"),
    ("Education City", 44667, "Al Rayyan"),
    ("Al Thumama", 44400, "Doha"),
    ("974 Stadium", 44089, "Doha"),
]

STAGES = ["Group Stage", "Round of 16", "Quarter Final", "Semi Final", "Final"]

CITIES_INDIA = [
    ("Bengaluru", 12.97, 77.59), ("Mumbai", 19.07, 72.87), ("Delhi", 28.61, 77.20),
    ("Chennai", 13.08, 80.27), ("Kolkata", 22.57, 88.36), ("Hyderabad", 17.38, 78.48),
    ("Pune", 18.52, 73.85), ("Ahmedabad", 23.02, 72.57)
]

CONTENT_TYPES = ["Highlight Reel", "Player Interview", "Behind the Scenes",
                  "Match Preview", "Live Score Update", "Fan Poll", "Meme/GIF",
                  "Stadium Tour", "Tactical Analysis", "Ticket Promo"]

PLATFORMS = ["Instagram", "TikTok", "X (Twitter)", "YouTube", "Facebook", "App Push"]

N_MATCHES = 48
N_FANS = 6000

# ---------------------------------------------------------------- MATCHES
def gen_matches():
    rows = []
    start = datetime(2026, 6, 15)
    for i in range(N_MATCHES):
        home, away = np.random.choice(TEAMS, 2, replace=False)
        stadium, capacity, city = STADIUMS[i % len(STADIUMS)]
        stage = STAGES[0] if i < 32 else STAGES[min(1 + (i - 32) // 8, 4)]
        popularity = np.random.uniform(0.4, 1.0)
        # Big teams get popularity boost
        big_teams = {"Brazil", "Argentina", "France", "Germany", "England", "Spain", "Portugal"}
        if home in big_teams or away in big_teams:
            popularity = min(1.0, popularity + 0.25)
        match_date = start + timedelta(days=int(i // 2), hours=int(np.random.choice([15, 18, 21])))
        rows.append({
            "match_id": f"M{i+1:03d}",
            "home_team": home,
            "away_team": away,
            "stadium": stadium,
            "capacity": capacity,
            "city": city,
            "stage": stage,
            "match_date": match_date.strftime("%Y-%m-%d"),
            "match_datetime": match_date.isoformat(),
            "day_of_week": match_date.strftime("%A"),
            "popularity_index": round(popularity, 3),
        })
    return pd.DataFrame(rows)

matches = gen_matches()

# ---------------------------------------------------------------- TICKET SALES / DEMAND
def gen_ticket_sales(matches):
    rows = []
    for _, m in matches.iterrows():
        base_demand = m["popularity_index"] * m["capacity"]
        stage_boost = {"Group Stage": 1.0, "Round of 16": 1.15, "Quarter Final": 1.3,
                        "Semi Final": 1.5, "Final": 1.8}[m["stage"]]
        noise = np.random.normal(1.0, 0.08)
        demand = base_demand * stage_boost * noise
        tickets_sold = int(min(m["capacity"], max(0.2 * m["capacity"], demand)))
        days_before_sellout = max(1, int(np.random.exponential(10) / stage_boost))
        avg_price = round(np.random.uniform(60, 150) * stage_boost, 2)
        resale_price = round(avg_price * np.random.uniform(0.9, 2.5), 2)
        rows.append({
            "match_id": m["match_id"],
            "tickets_sold": tickets_sold,
            "capacity": m["capacity"],
            "occupancy_rate": round(tickets_sold / m["capacity"], 3),
            "avg_ticket_price_usd": avg_price,
            "resale_avg_price_usd": resale_price,
            "days_to_sellout": days_before_sellout,
            "international_buyer_pct": round(np.random.uniform(0.2, 0.75), 2),
        })
    return pd.DataFrame(rows)

ticket_sales = gen_ticket_sales(matches)

# ---------------------------------------------------------------- FAN PROFILES
def gen_fan_profiles(n):
    rows = []
    segments_seed = np.random.choice(
        ["superfan", "casual", "digital_only", "local_loyalist", "family", "dormant"],
        n, p=[0.12, 0.28, 0.18, 0.15, 0.17, 0.10]
    )
    for i in range(n):
        city, lat, lon = CITIES_INDIA[np.random.randint(len(CITIES_INDIA))]
        age = int(np.clip(np.random.normal(29, 9), 16, 65))
        seg = segments_seed[i]
        favorite_team = np.random.choice(TEAMS, p=_team_probs())
        rows.append({
            "fan_id": f"F{i+1:05d}",
            "age": age,
            "gender": np.random.choice(["M", "F", "Other"], p=[0.58, 0.4, 0.02]),
            "city": city,
            "lat": lat + np.random.uniform(-0.05, 0.05),
            "lon": lon + np.random.uniform(-0.05, 0.05),
            "favorite_team": favorite_team,
            "true_segment": seg,
            "years_following_fifa": int(np.clip(np.random.normal(8, 5), 0, 30)),
            "app_installed": bool(np.random.rand() < (0.85 if seg != "dormant" else 0.3)),
            "loyalty_points": int(np.random.exponential(200) if seg == "superfan" else np.random.exponential(50)),
        })
    return pd.DataFrame(rows)

def _team_probs():
    big = {"Brazil", "Argentina", "France", "Germany", "England", "Spain", "Portugal"}
    p = np.array([2.5 if t in big else 1.0 for t in TEAMS])
    return p / p.sum()

fan_profiles = gen_fan_profiles(N_FANS)

# ---------------------------------------------------------------- SOCIAL ENGAGEMENT
def gen_social_engagement(matches, n_posts=1500):
    rows = []
    for i in range(n_posts):
        m = matches.sample(1).iloc[0]
        platform = np.random.choice(PLATFORMS, p=[0.24, 0.22, 0.18, 0.16, 0.12, 0.08])
        content = np.random.choice(CONTENT_TYPES)
        # engagement varies by content type & platform
        content_mult = {"Highlight Reel": 1.8, "Player Interview": 1.3, "Behind the Scenes": 1.2,
                         "Match Preview": 1.0, "Live Score Update": 1.5, "Fan Poll": 0.9,
                         "Meme/GIF": 1.9, "Stadium Tour": 0.8, "Tactical Analysis": 0.6,
                         "Ticket Promo": 0.5}[content]
        base = m["popularity_index"] * content_mult * np.random.uniform(500, 5000)
        impressions = int(base * 10)
        likes = int(base * np.random.uniform(0.4, 0.8))
        shares = int(base * np.random.uniform(0.05, 0.2))
        comments = int(base * np.random.uniform(0.03, 0.15))
        video_views = int(base * np.random.uniform(2, 6)) if content in ("Highlight Reel", "Player Interview", "Stadium Tour") else 0
        engagement_rate = round((likes + shares + comments) / max(impressions, 1), 4)
        post_date = pd.to_datetime(m["match_date"]) - timedelta(days=int(np.random.uniform(-1, 5)))
        rows.append({
            "post_id": f"P{i+1:05d}",
            "match_id": m["match_id"],
            "platform": platform,
            "content_type": content,
            "post_date": post_date.strftime("%Y-%m-%d"),
            "impressions": impressions,
            "likes": likes,
            "shares": shares,
            "comments": comments,
            "video_views": video_views,
            "engagement_rate": engagement_rate,
        })
    return pd.DataFrame(rows)

social_engagement = gen_social_engagement(matches)

# ---------------------------------------------------------------- CAMPAIGN PERFORMANCE
def gen_campaigns(n=40):
    types = ["Early Bird Ticket Offer", "Merch Discount", "App Push - Match Reminder",
             "Fan Loyalty Points Boost", "Referral Bonus", "Retargeting - Cart Abandon",
             "Newsletter - Weekly Digest", "SMS - Flash Sale"]
    rows = []
    for i in range(n):
        t = np.random.choice(types)
        sent = int(np.random.uniform(5000, 200000))
        open_rate = np.clip(np.random.normal(0.32, 0.1), 0.05, 0.75)
        ctr = np.clip(np.random.normal(0.08, 0.04), 0.005, 0.35)
        conv_rate = np.clip(np.random.normal(0.03, 0.02), 0.001, 0.2)
        opens = int(sent * open_rate)
        clicks = int(opens * ctr)
        conversions = int(clicks * conv_rate * 5)
        revenue = round(conversions * np.random.uniform(20, 180), 2)
        rows.append({
            "campaign_id": f"C{i+1:03d}",
            "campaign_type": t,
            "channel": np.random.choice(["Email", "App Push", "SMS", "Social Retarget"]),
            "audience_size": sent,
            "open_rate": round(open_rate, 3),
            "click_through_rate": round(ctr, 3),
            "conversion_rate": round(conv_rate, 4),
            "conversions": conversions,
            "revenue_usd": revenue,
        })
    return pd.DataFrame(rows)

campaigns = gen_campaigns()

# ---------------------------------------------------------------- CLICKSTREAM
def gen_clickstream(fan_profiles, matches, n=20000):
    pages = ["home", "match_schedule", "ticket_checkout", "team_profile",
             "merch_store", "live_scores", "highlights", "fan_zone", "account"]
    rows = []
    fan_ids = fan_profiles["fan_id"].values
    for i in range(n):
        fan = np.random.choice(fan_ids)
        m = matches.sample(1).iloc[0]
        session_len = max(5, int(np.random.exponential(180)))
        pages_viewed = np.random.choice(pages, size=np.random.randint(1, 6), replace=False)
        device = np.random.choice(["mobile", "desktop", "tablet"], p=[0.72, 0.22, 0.06])
        converted = np.random.rand() < 0.06
        rows.append({
            "session_id": f"S{i+1:06d}",
            "fan_id": fan,
            "match_id": m["match_id"],
            "session_seconds": session_len,
            "pages_viewed": ", ".join(pages_viewed),
            "device": device,
            "converted_to_purchase": bool(converted),
            "timestamp": (pd.to_datetime(m["match_date"]) - timedelta(days=int(np.random.uniform(0, 14)))).strftime("%Y-%m-%d"),
        })
    return pd.DataFrame(rows)

clickstream = gen_clickstream(fan_profiles, matches)

# ---------------------------------------------------------------- SAVE
out = "/home/claude/fifa_project/data"
import os
os.makedirs(out, exist_ok=True)
matches.to_csv(f"{out}/matches.csv", index=False)
ticket_sales.to_csv(f"{out}/ticket_sales.csv", index=False)
fan_profiles.to_csv(f"{out}/fan_profiles.csv", index=False)
social_engagement.to_csv(f"{out}/social_engagement.csv", index=False)
campaigns.to_csv(f"{out}/campaign_performance.csv", index=False)
clickstream.to_csv(f"{out}/clickstream.csv", index=False)

print("Rows generated:")
for name, df in [("matches", matches), ("ticket_sales", ticket_sales),
                  ("fan_profiles", fan_profiles), ("social_engagement", social_engagement),
                  ("campaigns", campaigns), ("clickstream", clickstream)]:
    print(f"  {name}: {len(df)}")
