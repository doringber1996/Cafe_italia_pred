import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
import altair as alt
from urllib.parse import quote
from io import BytesIO
import requests
from sklearn.ensemble import StackingRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Path to the folder containing the models and images
models_path = 'https://raw.githubusercontent.com/doringber1996/Cafe_italia_pred/main/'

# Load the dataset containing model information
optimal_models_df = pd.read_csv(f'{models_path}optimal_models_results.csv')

# Load images from GitHub
logo_url = f'{models_path}logo.png'
restaurant_url = f'{models_path}cafe-italia.jpg'

# Define the list of dishes
dish_columns = optimal_models_df['Dish'].unique()

# Load train data from GitHub
train_data_url = 'https://raw.githubusercontent.com/doringber1996/Cafe_italia_pred/main/train_data.csv'
train_data = pd.read_csv(train_data_url)

features = train_data.columns.drop(dish_columns)
features=features.drop('תאריך')

# Create MinMaxScaler based on train_data
scaler_X = MinMaxScaler()
scaler_y_dict = {dish: MinMaxScaler() for dish in dish_columns}

# Normalize X in train_data
scaled_X_train = scaler_X.fit_transform(train_data[features])

# Normalize y in train_data for each dish
for dish in dish_columns:
    scaler_y_dict[dish].fit(train_data[[dish]])

# Define the features to be used by each model
features_rf = [feature for feature in features if feature not in ['לקוחות יחס יומי', 'לקוחות יחס חודשי', 'סוף שבוע']]
features_svr = features
features_stacking_rf = features 

# Compute average customers per day and month
average_customers_per_day= {1: 264.2421052631579, 2: 284.775, 3: 294.87704918032784, 4: 296.3606557377049, 5: 352.64516129032256, 6: 354.008064516129, 7: 357.3414634146341}
average_customers_per_month = {1: 334.28409090909093, 2: 350.51851851851853, 3: 337.4623655913978, 4: 313.58024691358025, 5: 309.14606741573033, 6: 307.3448275862069, 7: 312.4561403508772, 8: 336.41379310344826, 9: 310.47058823529414, 10: 216.94545454545454, 11: 282.43859649122805, 12: 352.66129032258067}

# Calculate correlation matrix for menu items (excluding metadata columns)
corr_matrix = train_data.loc[:,dish_columns].corr()
# Create a list to store pairs of highly correlated dishes
high_corr_pairs = []


# Identify highly correlated pairs of dishes
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))

# Create new features based on correlation values
for dish1, dish2, corr_value in high_corr_pairs:
    new_feature_name = f"{dish1}_corr_{dish2}"
    train_data[new_feature_name] = corr_value
    
# Function to add derived features to the dataset
def add_features(data, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    data['תאריך'] = pd.to_datetime(data['תאריך'])
    data['יום בשבוע'] = data['תאריך'].dt.dayofweek + 1
    data['יום בשבוע'] = data['יום בשבוע'].apply(lambda x: 1 if x == 7 else x + 1)
    data['חודש'] = data['תאריך'].dt.month
    data['שנה'] = data['תאריך'].dt.year

    data['לקוחות יחס יומי'] = data.apply(lambda x: x['מספר לקוחות'] / average_customers_per_day[x['יום בשבוע']], axis=1)
    data['לקוחות יחס חודשי'] = data.apply(lambda x: x['מספר לקוחות'] / average_customers_per_month[x['חודש']], axis=1)

    data.rename(columns={'לקחות יחס יומי': 'לקוחות יחס יומי', 'לקחות יחס חודשי': 'לקוחות יחס חודשי'}, inplace=True)
    data['סוף שבוע'] = data['יום בשבוע'].isin([5, 6, 7])

    new_features = {}
    for dish1, dish2, corr_value in high_corr_pairs:
        new_feature_name = f"{dish1}_corr_{dish2}"
        new_features[new_feature_name] = corr_value
    
    new_features_df = pd.DataFrame(new_features, index=data.index)
    data = pd.concat([data, new_features_df], axis=1)
    
    return data

# Preprocessing function for SVR
def preprocess_input_svr(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs, scaler_X):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({'תאריך': dates})
    data['מספר לקוחות'] = num_customers
    data = add_features(data, average_customers_per_day, average_customers_per_month, high_corr_pairs)
    data[features_svr] = scaler_X.transform(data[features_svr])
    
    return data

# Preprocessing function for RF and Stacking RF
def preprocess_input_rf(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({'תאריך': dates})
    data['מספר לקוחות'] = num_customers
    data = add_features(data, average_customers_per_day, average_customers_per_month, high_corr_pairs)
    return data

def predict_dishes(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs, scaler_X, scaler_y_dict):
    results = {}
    input_data_svr = preprocess_input_svr(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs, scaler_X)
    input_data_rf = preprocess_input_rf(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs)

    for dish in dish_columns:
        best_model_type = optimal_models_df.loc[optimal_models_df['Dish'] == dish, 'Model'].values[0]
        if best_model_type == 'SVR':
            predictions = load_model_and_predict(dish, input_data_svr, best_model_type, scaler_y_dict)
        else:
            predictions = load_model_and_predict(dish, input_data_rf, best_model_type)
        results[dish] = predictions

    return results
    
def load_model_and_predict(dish, input_data, model_type, scaler_y_dict=None):
    model_type = model_type.lower()
    if model_type == 'svr':
        model_file = f'{models_path}best_svr_model_{dish}.pkl'
        features = input_data[features_svr]
    elif model_type == 'stacking rf':
        model_file = f'{models_path}best_stacking_rf_model_{dish}.pkl'
        features = input_data[features_stacking_rf]
    elif model_type == 'random forest':
        model_file = f'{models_path}best_rf_model_{dish}.pkl'
        features = input_data[features_rf]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    try:
        response = requests.get(model_file)
        response.raise_for_status()
        model = joblib.load(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Model file not found or error in loading: {model_file}, Error: {e}")
        return np.array([])
    except Exception as e:
        st.error(f"Error in loading the model: {model_file}, Error: {e}")
        return np.array([])

    predictions_scaled = model.predict(features)
    
    if model_type == 'svr':
        predictions = scaler_y_dict[dish].inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()
    else:
        predictions = predictions_scaled
    
    predictions = np.ceil(predictions).astype(int)
    return predictions
    

st.markdown(
    f"""
    <style>
    .main {{
        background-image: url("{restaurant_url}");
        background-size: cover;
        position: relative;
        z-index: 1;
        color: white;
    }}
    .stDownloadButton button {{
            background-color: rgba(255, 255, 255, 0.8);
            color: black;
            border-radius: 12px;
        }}
    .stButton button {{
        background-color: rgba(255, 255, 255, 0.8);
        color: black;
        border-radius: 12px;
    }}
    .stTextInput, .stNumberInput input {{
        color: black;
    }}
    .title {{
        background-color: rgba(255, 255, 255, 0.8); /* רקע בהיר */
        color: black !important;
        padding: 10px;
        border-radius: 10px;
    }}
    .header {{
        background-color: rgba(255, 255, 255, 0.8); /* רקע בהיר */
        color: black !important;
        padding: 5px;
        border-radius: 5px;
    }}
    .css-10trblm, .css-1v3fvcr p {{
        color: white !important;
    }}
    .stTitle, .stHeader, .stSubheader, .stMarkdown, .stText, .stNumberInput label, .stDateInput label {{
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.image(logo_url, width=200)

st.markdown('<h1 class="title">Dish Prediction Application</h1>', unsafe_allow_html=True)

st.markdown('<h2 class="header">Input Parameters</h2>', unsafe_allow_html=True)

# Input fields
start_date = st.date_input("Start Date", datetime.now())
end_date = st.date_input("End Date", datetime.now() + timedelta(days=1))
num_customers = st.number_input("Number of Customers", min_value=1, step=1)

if st.button("Predict"):
    results = predict_dishes(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs, scaler_X, scaler_y_dict)

    results_text = "Predicted Dishes:\n"
    predictions_data = []
    for dish, prediction in results.items():
        results_text += f"{dish}: {prediction.sum()}\n"
        predictions_data.append({"Dish": dish, "Prediction": prediction.sum()})

    # Display results as a table
    st.markdown('<h1 class="title">Prediction Results</h1>', unsafe_allow_html=True)
    predictions_df = pd.DataFrame(predictions_data)
    st.dataframe(predictions_df, use_container_width=True, height=40 * predictions_df.shape[0])

    # Display results as a bar chart
    st.markdown('<h2 class="title">Prediction Bar Chart</2>', unsafe_allow_html=True)

    chart = alt.Chart(predictions_df).mark_bar().encode(
        x=alt.X('Dish', sort=None, axis=alt.Axis(labelAngle=0)), 
        y='Prediction',
        color=alt.Color('Dish', scale=alt.Scale(scheme='tableau20')),  # צבעים ייחודיים לכל מנה
        tooltip=['Dish', 'Prediction']
    ).properties(width=700, height=400)
    st.altair_chart(chart)

    # Provide option to download results
    st.markdown('<h2 class="title">Download Results</h2>', unsafe_allow_html=True)
    csv = predictions_df.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='predictions.csv', mime='text/csv')
