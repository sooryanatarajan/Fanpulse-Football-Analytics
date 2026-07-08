"""
Fan Mood Index
--------------
Generates realistic synthetic fan comments per match (with a deliberate
"controversy" injection on a handful of matches), scores them with VADER
sentiment analysis, and builds:
  1. A daily mood trend per match (sentiment in the run-up to kickoff)
  2. An overall mood index per match (weighted toward the last 3 days)
  3. An Engagement-Sentiment Divergence flag: matches where engagement
     is rising but sentiment is falling -> an early-warning signal for
     organizers (something is brewing: price backlash, injury news,
     refereeing controversy, travel/logistics complaints, etc.)
"""
import numpy as np
import pandas as pd
from datetime import timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json

np.random.seed(7)
analyzer = SentimentIntensityAnalyzer()

D = "/home/claude/fifa_project/data"
matches = pd.read_csv(f"{D}/matches.csv")
social = pd.read_csv(f"{D}/social_engagement.csv")


POSITIVE = [
    "Can't wait for {home} vs {away}, this is going to be electric!",
    "{home} are looking unstoppable this tournament, so proud!",
    "Just got my tickets for {home} vs {away}, best day ever!",
    "The atmosphere at {stadium} is going to be insane, so hyped!",
    "{away} fans travelling in huge numbers, respect for the support!",
    "This lineup for {home} is exactly what we needed, love it!",
    "Watching {home} vs {away} with the whole family, can't wait!",
    "That highlight reel from the last match gave me chills, let's go {home}!",
    "Best ticket price I've seen all tournament for {home} vs {away}!",
    "{home} to win it all, calling it now!",
]
NEGATIVE_PRICE = [
    "Ticket prices for {home} vs {away} are honestly outrageous, pricing out real fans.",
    "Resale prices for {home} vs {away} are out of control, scalpers ruining it for everyone.",
    "Can barely afford tickets anymore, prices have gone way too high this year.",
    "Why does {home} vs {away} cost double the group stage matches, feels like a cash grab.",
]
NEGATIVE_LOGISTICS = [
    "Transit to {stadium} was a nightmare last time, hope they fix it for {home} vs {away}.",
    "No clear signage getting to {stadium}, missed kickoff because of it.",
    "Queues to get into {stadium} took over an hour, really needs sorting out.",
]
NEGATIVE_CONTROVERSY = [
    "Not happy about the news on {home}'s squad ahead of {away} match, worrying signs.",
    "That refereeing decision in the last {home} game was a disgrace, still fuming.",
    "Reports of a key {home} player injury are concerning right before this game.",
    "Feels like {home} are being treated unfairly by officials again this tournament.",
    "Rumors swirling about {home}'s lineup changes, fans are not happy about it.",
]
NEUTRAL = [
    "Anyone know the kickoff time for {home} vs {away}?",
    "What's the weather looking like for {home} vs {away} at {stadium}?",
    "Watching the {home} vs {away} preview show right now.",
    "Curious how {home} lines up against {away} this time.",
    "Heading to {stadium} this weekend, first time at a match like this.",
]

def fill(t, m):
    return t.format(home=m["home_team"], away=m["away_team"], stadium=m["stadium"])


controversy_matches = np.random.choice(matches["match_id"], size=6, replace=False)

rows = []
comment_id = 1
for _, m in matches.iterrows():
    is_controversy = m["match_id"] in controversy_matches
    n_comments = np.random.randint(35, 70)
    for _ in range(n_comments):
        days_before = np.random.choice([7, 6, 5, 4, 3, 2, 1, 0], p=[0.05, 0.07, 0.09, 0.12, 0.14, 0.17, 0.18, 0.18])
       
        if is_controversy and days_before <= 3:
            weights = {"pos": 0.20, "price": 0.15, "logi": 0.10, "contro": 0.45, "neu": 0.10}
        elif is_controversy:
            weights = {"pos": 0.45, "price": 0.15, "logi": 0.10, "contro": 0.15, "neu": 0.15}
        else:
            weights = {"pos": 0.62, "price": 0.10, "logi": 0.08, "contro": 0.05, "neu": 0.15}
        cat = np.random.choice(list(weights.keys()), p=list(weights.values()))
        if cat == "pos":
            text = fill(np.random.choice(POSITIVE), m)
        elif cat == "price":
            text = fill(np.random.choice(NEGATIVE_PRICE), m)
        elif cat == "logi":
            text = fill(np.random.choice(NEGATIVE_LOGISTICS), m)
        elif cat == "contro":
            text = fill(np.random.choice(NEGATIVE_CONTROVERSY), m)
        else:
            text = fill(np.random.choice(NEUTRAL), m)
        platform = np.random.choice(["Instagram", "TikTok", "X (Twitter)", "YouTube", "Facebook"])
        comment_date = pd.to_datetime(m["match_date"]) - timedelta(days=int(days_before))
        rows.append({
            "comment_id": f"CM{comment_id:06d}",
            "match_id": m["match_id"],
            "platform": platform,
            "days_before_match": int(days_before),
            "comment_date": comment_date.strftime("%Y-%m-%d"),
            "text": text,
            "category": cat,
            "is_controversy_match": bool(is_controversy),
        })
        comment_id += 1

comments = pd.DataFrame(rows)


scores = comments["text"].apply(lambda t: analyzer.polarity_scores(t)["compound"])
comments["sentiment_score"] = scores.round(4)
comments["sentiment_class"] = pd.cut(
    comments["sentiment_score"], bins=[-1.01, -0.05, 0.05, 1.01], labels=["Negative", "Neutral", "Positive"]
)
comments.to_csv(f"{D}/fan_comments.csv", index=False)


trend = comments.groupby(["match_id", "days_before_match"]).agg(
    avg_sentiment=("sentiment_score", "mean"),
    n_comments=("comment_id", "count"),
).reset_index().sort_values(["match_id", "days_before_match"], ascending=[True, False])
trend.to_csv(f"{D}/mood_trend.csv", index=False)


def weighted_mood(g):
    w = np.where(g["days_before_match"] <= 3, 2.0, 1.0)
    return np.average(g["sentiment_score"], weights=w)

mood_index = comments.groupby("match_id").apply(weighted_mood).reset_index(name="mood_score")
mood_index["mood_index_0_100"] = ((mood_index["mood_score"] + 1) / 2 * 100).round(1)

pos_share = comments[comments["sentiment_class"] == "Positive"].groupby("match_id").size()
neg_share = comments[comments["sentiment_class"] == "Negative"].groupby("match_id").size()
total = comments.groupby("match_id").size()
mood_index = mood_index.set_index("match_id")
mood_index["pct_positive"] = (pos_share / total * 100).round(1)
mood_index["pct_negative"] = (neg_share / total * 100).round(1)
mood_index = mood_index.reset_index()


early = comments[comments["days_before_match"] >= 4].groupby("match_id")["sentiment_score"].mean()
late = comments[comments["days_before_match"] <= 2].groupby("match_id")["sentiment_score"].mean()
mood_index = mood_index.merge(early.rename("early_sentiment"), on="match_id", how="left")
mood_index = mood_index.merge(late.rename("late_sentiment"), on="match_id", how="left")
mood_index["sentiment_delta"] = (mood_index["late_sentiment"] - mood_index["early_sentiment"]).round(3)

mood_index = mood_index.merge(
    matches[["match_id", "home_team", "away_team", "stage", "match_date"]], on="match_id", how="left"
)


buzz = social.groupby("match_id").agg(total_engagement=("likes", "sum")).reset_index()
buzz["engagement_z"] = (buzz["total_engagement"] - buzz["total_engagement"].mean()) / buzz["total_engagement"].std()
mood_index = mood_index.merge(buzz[["match_id", "total_engagement", "engagement_z"]], on="match_id", how="left")


mood_index["divergence_flag"] = (mood_index["engagement_z"] > 0.15) & (mood_index["sentiment_delta"] < -0.05)

mood_index = mood_index.sort_values("mood_index_0_100")
mood_index.to_csv(f"{D}/mood_index.csv", index=False)


top_positive = comments.sort_values("sentiment_score", ascending=False).drop_duplicates("match_id").head(6)
top_negative = comments.sort_values("sentiment_score", ascending=True).drop_duplicates("match_id").head(6)

sample_comments = {
    "positive": top_positive[["match_id", "text", "sentiment_score", "platform"]].to_dict("records"),
    "negative": top_negative[["match_id", "text", "sentiment_score", "platform"]].to_dict("records"),
}
with open(f"{D}/mood_samples.json", "w") as f:
    json.dump(sample_comments, f, indent=2, default=str)


print(f"Generated {len(comments)} comments across {comments['match_id'].nunique()} matches")
print("\nOverall sentiment distribution:")
print(comments["sentiment_class"].value_counts(normalize=True).round(3) * 100)
print(f"\nMatches flagged with Engagement-Sentiment Divergence: {mood_index['divergence_flag'].sum()}")
print(mood_index[mood_index["divergence_flag"]][["match_id", "home_team", "away_team", "mood_index_0_100", "sentiment_delta", "total_engagement"]])
print("\nLowest mood-index matches:")
print(mood_index[["match_id", "home_team", "away_team", "mood_index_0_100", "pct_negative"]].head(5))
