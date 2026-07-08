# FanPulse — AI-Driven Fan Analytics for FIFA Events

FanPulse unifies ticketing, attendance, social media, digital campaign, and fan
profile data into a single AI-driven platform to predict ticket demand,
understand fan engagement, segment fans by behavior, and personalize
communication and offers — while supporting crowd planning for organizers.

Built for the FIFA Fan Analytics Hackathon Challenge.

## 🎥 Demo Video
[Watch the demo on YouTube]https://youtu.be/VC8gZzlitQI

## 🚀 Live Dashboard
No install needed — just download `dashboard/FanPulse_Dashboard.html` and open
it in any browser.

## 🧠 What's Inside

| Feature | Method | Result |
|---|---|---|
| Demand Forecasting | Random Forest Regressor | R² = 0.78, predicts occupancy per match |
| Fan Segmentation | K-Means Clustering (k=5) | 5 behavioral segments across 6,000 fans |
| Fan Mood Index | VADER Sentiment Analysis | Flags engagement-vs-sentiment divergence as an early warning signal |
| Offer Propensity | Random Forest Classifier | 88.5% accuracy on predicting offer response |

## 📊 Dashboard Views
- **Overview** — headline KPIs, top-demand matches, segment mix, platform engagement
- **Demand Forecast** — match-by-match predicted occupancy and demand tiers
- **Fan Segments** — segment sizes, behavior profile, recommended actions per segment
- **Engagement** — content-type and platform performance, campaign channel results
- **Fan Mood Index** — sentiment trend per match and engagement-sentiment divergence alerts
- **Recommendations** — marketing/operations guidance and ranked fan offer-propensity list

## 🛠️ Tech Stack
Python (pandas, scikit-learn, VADER Sentiment) for data generation and modeling;
HTML + Chart.js for the interactive dashboard.

## 📁 Repo Structure
```
fanpulse-fifa-analytics/
├── dashboard/
│   └── FanPulse_Dashboard.html
├── docs/
│   ├── FanPulse_Solution_Architecture.docx
│   └── FanPulse_Pitch_Deck.pptx
├── src/
│   ├── generate_data.py
│   ├── model.py
│   └── fan_mood_index.py
└── data/
    └── (generated CSVs + metrics.json)
```

## ⚠️ Data Disclosure
All data in this prototype is simulated to mirror the structure of real
ticketing, social, and fan-profile systems. No real FIFA data, real fans, or
real transactions are used. See `src/generate_data.py` for how the synthetic
data is structured.

## 🔧 Running It Yourself
```bash
pip install pandas numpy scikit-learn vaderSentiment
python src/generate_data.py
python src/model.py
python src/fan_mood_index.py
```
Then open `dashboard/FanPulse_Dashboard.html` in a browser.

## 👥 Team
Soorya Anvekar Natrajan · Namitha G
