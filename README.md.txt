IPL 2026 Match Prediction Model

Predicting IPL 2026 match winners using historical IPL data and 2026 team squads.

About

This project uses machine learning to predict IPL 2026 match outcomes.
It considers:

- Player performance (batting, bowling, all-rounders)
- Team power scores
- Head-to-head stats
- Recent form
- Venue advantage
- Toss impact

The model uses an XGBoost ensemble to generate win probabilities for each team.

Folder Structure

ipl-match-prediction/
├── data/                  
│   ├── deliveries_updated_ipl_upto_2025.csv
│   ├── ipl_2026_full_squads.csv
│   └── ipl.csv
├── src/
│   └── ipl_prediction_model.py
├── README.md

- data/: Place all CSV files here.
- src/: Python scripts for the model.