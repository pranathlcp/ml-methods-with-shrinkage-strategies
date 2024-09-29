import streamlit as st
import pickle
from joblib import load
import pandas as pd
import sklearn
import numpy as np

config = {
    "balls_per_over": 6,
    "to_win": 0,
    "model_path": "models/logistic_regression_model.joblib",
    "model_path_lr": "models/logistic_regression_model.joblib",
    "model_path_lr_lasso": "models/logistic_regression_lasso_model.joblib",
    "model_path_gb": "models/gb_model.joblib",
    "model_path_mars": "models/gb_model.joblib",
    "teams": ['New Zealand', 'Australia', 'South Africa', 'England', 'Pakistan',
       'India', 'Sri Lanka', 'Bangladesh', 'Scotland', 'Zimbabwe',
       'Kenya', 'West Indies', 'Ireland', 'Netherlands', 'Afghanistan',
       'Canada', 'Hong Kong', 'Nepal', 'Papua New Guinea', 'Oman',
       'United Arab Emirates', 'ICC World XI', 'United States of America',
       'Philippines', 'Vanuatu', 'Namibia', 'Botswana', 'Ghana', 'Uganda',
       'Italy', 'Germany', 'Denmark', 'Guernsey', 'Norway', 'Malaysia',
       'Maldives', 'Thailand', 'Qatar', 'Kuwait', 'Singapore',
       'Cayman Islands', 'Bermuda', 'Nigeria', 'Jersey', 'Spain',
       'Portugal', 'Bhutan', 'Saudi Arabia', 'Bahrain', 'Luxembourg',
       'Czech Republic', 'Romania', 'Bulgaria', 'Austria', 'Greece',
       'Serbia', 'Malta', 'Belgium', 'France', 'Sweden', 'Rwanda',
       'Gibraltar', 'Finland', 'Hungary', 'Cyprus', 'Isle of Man',
       'Estonia', 'Lesotho', 'Seychelles', 'Malawi', 'Sierra Leone',
       'Switzerland', 'Mozambique', 'Cameroon', 'Tanzania', 'Belize',
       'Argentina', 'Panama', 'Bahamas', 'Israel', 'Turkey', 'Croatia',
       'Slovenia', 'Fiji', 'Cook Islands', 'Samoa', 'Indonesia',
       'South Korea', 'Japan', 'Mali', 'St Helena', 'Eswatini', 'Gambia',
       'Myanmar', 'China'],
    "beta_s": [-0.038725950,  0.01251763,  0.02231607, -0.02947968,  0.66086090,
               -0.015489640,  0.02311430, -0.00475097,  0.00719932, -0.00767896,
                0.009880790, -0.01111963,  0.01635897],
    "beta_ps": [-0.03872595,  0.01251763,  0.02231607, -0.02947968,  0.6608609,
                -0.01548964,  0.0231143 , -0.00475097,  0.00719932, -0.00767896,
                0.00988079, -0.01111963,  0.01635897],
    "beta_ls": [-0.05331159, 0.02145901, 0.01005249, -0.0302106 ,  0.54438553,
                0.        ,  0.01020999, -0.00438091,  0.00532614, 0.        ,
                0.00522598,  0.        ,  0.        ],
    "beta_pt": [-0.03840893,  0.01232329,  0.02258262, -0.02946379,  0.66339247,
                -0.01582631,  0.02339478, -0.00475902,  0.00724003, -0.00784586,
                0.00998196, -0.01136131,  0.01671453],
    "beta_spt": [-0.03840893,  0.01232329,  0.02258262, -0.02946379,  0.66339247,
                 -0.01582631,  0.02339478, -0.00475902,  0.00724003, -0.00784586,
                 0.00998196, -0.01136131,  0.01671453]
}

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

model_loaded = load(config["model_path"])

def logistic_function(x):
    return 1/(1 + np.exp(-x))

def predict_shrinkage(X, coefficients, threshold=0.5):
    z = np.dot(X, coefficients)
    probabilities = logistic_function(z)
    predictions = (probabilities >= threshold).astype(int)
    return probabilities

def main():
    st.set_page_config(layout="wide")

    # Set the title of the main page
    st.title("T20 Cricket Winner Prediction (Second Innings)")

    # Using the sidebar for input
    st.sidebar.header("Main Configurations")

    with st.sidebar:
        # Collect inputs
        config["model"] = st.selectbox("Select Model", ["Logistic Regression",
                                                        "Logistic Regression (with LASSO)",
                                                        "Shrinkage",
                                                        "Positive Shrinkage",
                                                        "Linear Shrinkage",
                                                        "Pretest",
                                                        "Shrinkage Pretest",
                                                        "GBM"], index=0)
        config["team_A"] = st.selectbox("Select Team Batting First", config["teams"], index=1)
        config["team_B"] = st.selectbox("Select Team Batting Second", config["teams"], index=0)

        config["model_path"] = (
            config["model_path_lr"] if config["model"] == "Logistic Regression" else
            config["model_path_lr_lasso"] if config["model"] == "Logistic Regression (with LASSO)" else
            config["model_path_gb"] if config["model"] == "GBM" else
            config["model_path_mars"] if config["model"] == "MARS" else
            config["model_path_lr_lasso"]
        )
        model_loaded = load(config["model_path"])
        innings_runs = st.sidebar.number_input("Innings Runs", min_value=0, value=38, step=1)
        innings_balls = st.sidebar.number_input("Innings Balls", min_value=0, value=20, step=1)
        innings_wickets = st.sidebar.number_input("Innings Wickets", min_value=0, value=1, max_value=10, step=1)
        target_score = st.sidebar.number_input("Target Score", min_value=0, value=168, step=1)    
        
        config["to_win"] = target_score - innings_runs
        config["balls_remaining"] = 120 - innings_balls
        config["innings_balls"] = innings_balls
        config["crr"] = round(innings_runs/(innings_balls/6), 2)
        config["rrr"] = round(config["to_win"] / (config["balls_remaining"]/6), 2)
        config["wickets_remaining"] = 10 - innings_wickets

    st.markdown('###')
    
    st.subheader("Other Inputs", divider='rainbow')
    c_input_1, c_input_2, c_input_3, c_input_4, c_input_5  = st.columns(5)
    with c_input_1:
        config["strikers_runs"] = st.number_input("Striker's Runs", min_value=0, value=14, max_value=innings_runs, step=1)
    with c_input_2:
        config["balls_faced_by_striker"] = st.number_input("Balls Faced by Striker", min_value=0, 
                                                           value=8, max_value=innings_balls, step=1)
    with c_input_3:
        config["non_strikers_runs"] = st.number_input("Non Striker's Runs", min_value=0, max_value=innings_runs,
                                                      value=6, step=1)
    with c_input_4:
        config["balls_faced_by_non_striker"] = st.number_input("Balls Faced by Non Striker", min_value=0,
                                                               max_value=innings_balls, value=6, step=1)
    with c_input_5:
        config["runs_conceded_by_bowler"] = st.number_input("Runs Conceded by Bowler", min_value=0, value=7, step=1)

    st.markdown('##')
    
    ## Prediction
    X_df = pd.DataFrame({
        'Runs to Get': config["to_win"], 
        'Balls Remaining': config["balls_remaining"], 
        'Innings Runs': innings_runs, 
        'Innings Balls': innings_balls, 
        'Wickets Remaining': config["wickets_remaining"],
        'Target Score': target_score, 
        'CRR': config["crr"],
        'RRR': config["rrr"],
        'Total Batter Runs': config["strikers_runs"],
        'Batter Balls Faced': config["balls_faced_by_striker"],
        'Total Non Striker Runs': config["non_strikers_runs"], 
        'Non Striker Balls Faced': config["balls_faced_by_non_striker"],
        'Bowler Runs Conceded': config["runs_conceded_by_bowler"]
    }, index=[0])

    loaded_scaler.transform(X_df)

    probability = 0

    if config["model"] == "Shrinkage":
        probability = predict_shrinkage(X_df, config["beta_s"])[0]
    elif config["model"] == "Positive Shrinkage":
        probability = predict_shrinkage(X_df, config["beta_ps"])[0]
    elif config["model"] == "Linear Shrinkage":
        probability = predict_shrinkage(X_df, config["beta_ls"])[0]
    elif config["model"] == "Pretest":
        probability = predict_shrinkage(X_df, config["beta_pt"])[0]
    elif config["model"] == "Shrinkage Pretest":
        probability = predict_shrinkage(X_df, config["beta_spt"])[0]
    else:
        probability = model_loaded.predict_proba(X_df)[0][1]

    probability = round(probability * 100, 2)

    st.subheader(f'Winning Probability - {config["model"]}', divider='rainbow')
    col1, col2, col3, col4 = st.columns([0.25, 0.25, 0.25, 0.25])

    with col1:
        bulb = "🟢" if probability < 50 else "🔴"
        st.metric(label=f'{config["team_A"]}', value=f'{bulb}  {round(100 - probability, 2)} %')

    with col2:
        bulb = "🟢" if probability > 50 else "🔴"
        st.metric(label=f'{config["team_B"]}', value=f'{bulb}  {probability} %')

    st.markdown('##')

    # Configuration Information Panel
    st.subheader("Current Match Situation (Second Innings)", divider='rainbow')
    input_1, input_2, input_3, input_4  = st.columns(4)
    with input_1:
        st.metric(label="Innings Runs", value=f'{innings_runs}')
    with input_2:
        overs = config["innings_balls"] // config["balls_per_over"]
        over_balls = config["innings_balls"] % config["balls_per_over"]
        st.metric(label="Overs Bowled", value=f'{overs}.{over_balls}')
    with input_3:
        overs = config["balls_remaining"] // config["balls_per_over"]
        over_balls = config["balls_remaining"] % config["balls_per_over"]
        st.metric(label="Overs Remaining", value=f'{overs}.{over_balls}')
    with input_4:
        st.metric(label="Wickets Remaining", value=f'{config["wickets_remaining"]}')
    
    input_5, input_6, input_7, input_8  = st.columns(4)
    with input_5:
        st.metric(label="Target", value=f'{target_score}')
    with input_6:
        st.metric(label="To Win", value=f'{config["to_win"]}')
    with input_7:
        st.metric(label="Current Run Rate", value=f'{config["crr"]}')
    with input_8:
        st.metric(label="Required Run Rate", value=f'{config["rrr"]}')
    
    st.markdown('##')

if __name__ == '__main__':
    main()