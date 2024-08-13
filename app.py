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

#Load the dataset containing model information
optimal_models_df = pd.read_csv(f'{models_path}optimal_models_results.csv')

# Load images from GitHub
logo_url = f'{models_path}logo.png'
restaurant_url = f'{models_path}cafe-italia.jpg'

# Define the list of dishes
dish_columns = optimal_models_df['Dish'].unique()

features = ["מספר לקוחות",
"יום בשבוע",
"חודש",
"שנה",
"חציל פרמז'ן_corr_פוקצ'ת הבית",
"שרימפס אליו פפרונצ'ינו_corr_חציל פרמז'ן",
"קרפצ'יו בקר אורוגולה ופרמז'ן_corr_פוקצ'ת הבית",
"קרפצ'יו בקר אורוגולה ופרמז'ן_corr_חציל פרמז'ן",
"קרפצ'יו בקר אורוגולה ופרמז'ן_corr_שרימפס אליו פפרונצ'ינו",
"סלט חסה גדול_corr_פוקצ'ת הבית",
"סלט חסה גדול_corr_חציל פרמז'ן",
"סלט חסה גדול_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"סלט קיסר_corr_פוקצ'ת הבית",
"סלט קיסר_corr_חציל פרמז'ן",
"לינגוויני ירקות_corr_פוקצ'ת הבית",
"לינגוויני ירקות_corr_חציל פרמז'ן",
"לינגוויני ירקות_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"לינגוויני ירקות_corr_סלט חסה גדול",
"לינגוויני ארביאטה_corr_פוקצ'ת הבית",
"לינגוויני ארביאטה_corr_חציל פרמז'ן",
"לינגוויני ארביאטה_corr_סלט חסה גדול",
"פפרדלה פטריות ושמנת_corr_פוקצ'ת הבית",
"פפרדלה פטריות ושמנת_corr_חציל פרמז'ן",
"פפרדלה פטריות ושמנת_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"פפרדלה פטריות ושמנת_corr_סלט חסה גדול",
"פפרדלה פטריות ושמנת_corr_סלט קיסר",
"פטוצ'יני תרד גורגונזולה_corr_פוקצ'ת הבית",
"פטוצ'יני תרד גורגונזולה_corr_חציל פרמז'ן",
"פטוצ'יני תרד גורגונזולה_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"פטוצ'יני תרד גורגונזולה_corr_פפרדלה פטריות ושמנת",
"פסטה בולונז_corr_פוקצ'ת הבית",
"פסטה בולונז_corr_חציל פרמז'ן",
"פסטה בולונז_corr_שרימפס אליו פפרונצ'ינו",
"פסטה בולונז_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"פסטה בולונז_corr_סלט חסה גדול",
"פסטה בולונז_corr_לינגוויני ירקות",
"פסטה בולונז_corr_לינגוויני ארביאטה",
"פסטה בולונז_corr_פפרדלה פטריות ושמנת",
"פסטה בולונז_corr_פטוצ'יני תרד גורגונזולה",
"פנה קרבונרה_corr_פוקצ'ת הבית",
"פנה קרבונרה_corr_חציל פרמז'ן",
"פנה קרבונרה_corr_שרימפס אליו פפרונצ'ינו",
"פנה קרבונרה_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"פנה קרבונרה_corr_סלט חסה גדול",
"פנה קרבונרה_corr_סלט קיסר",
"פנה קרבונרה_corr_לינגוויני ירקות",
"פנה קרבונרה_corr_לינגוויני ארביאטה",
"פנה קרבונרה_corr_פפרדלה פטריות ושמנת",
"פנה קרבונרה_corr_פטוצ'יני תרד גורגונזולה",
"פנה קרבונרה_corr_פסטה בולונז",
"מאצי רוזה אפונה ובייקון_corr_פוקצ'ת הבית",
"מאצי רוזה אפונה ובייקון_corr_חציל פרמז'ן",
"מאצי רוזה אפונה ובייקון_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"מאצי רוזה אפונה ובייקון_corr_פסטה בולונז",
"מאצי רוזה אפונה ובייקון_corr_פנה קרבונרה",
"לזניה בולונז_corr_פוקצ'ת הבית",
"לזניה בולונז_corr_חציל פרמז'ן",
"לזניה בולונז_corr_שרימפס אליו פפרונצ'ינו",
"לזניה בולונז_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"לזניה בולונז_corr_סלט חסה גדול",
"לזניה בולונז_corr_לינגוויני ירקות",
"לזניה בולונז_corr_פפרדלה פטריות ושמנת",
"לזניה בולונז_corr_פטוצ'יני תרד גורגונזולה",
"לזניה בולונז_corr_פסטה בולונז",
"לזניה בולונז_corr_פנה קרבונרה",
"לזניה בולונז_corr_מאצי רוזה אפונה ובייקון",
"טורטלוני_corr_פוקצ'ת הבית",
"פסטה פירות ים_corr_פוקצ'ת הבית",
"פסטה פירות ים_corr_חציל פרמז'ן",
"פסטה פירות ים_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"פסטה פירות ים_corr_סלט חסה גדול",
"פסטה פירות ים_corr_לינגוויני ירקות",
"פסטה פירות ים_corr_פסטה בולונז",
"פסטה פירות ים_corr_פנה קרבונרה",
"פסטה פירות ים_corr_לזניה בולונז",
"פילה דג_corr_פוקצ'ת הבית",
"פילה דג_corr_חציל פרמז'ן",
"פילה דג_corr_שרימפס אליו פפרונצ'ינו",
"פילה דג_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"פילה דג_corr_סלט חסה גדול",
"פילה דג_corr_לינגוויני ירקות",
"פילה דג_corr_פטוצ'יני תרד גורגונזולה",
"פילה דג_corr_פסטה בולונז",
"פילה דג_corr_פנה קרבונרה",
"פילה דג_corr_מאצי רוזה אפונה ובייקון",
"פילה דג_corr_לזניה בולונז",
"פילה דג_corr_פסטה פירות ים",
"כבדי עוף ובצל מטוגן_corr_פוקצ'ת הבית",
"כבדי עוף ובצל מטוגן_corr_חציל פרמז'ן",
"כבדי עוף ובצל מטוגן_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"כבדי עוף ובצל מטוגן_corr_סלט חסה גדול",
"כבדי עוף ובצל מטוגן_corr_לינגוויני ירקות",
"כבדי עוף ובצל מטוגן_corr_פסטה בולונז",
"כבדי עוף ובצל מטוגן_corr_פנה קרבונרה",
"כבדי עוף ובצל מטוגן_corr_מאצי רוזה אפונה ובייקון",
"כבדי עוף ובצל מטוגן_corr_לזניה בולונז",
"כבדי עוף ובצל מטוגן_corr_פסטה פירות ים",
"כבדי עוף ובצל מטוגן_corr_פילה דג",
"פרגיות_corr_פוקצ'ת הבית",
"פרגיות_corr_חציל פרמז'ן",
"פרגיות_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"פרגיות_corr_סלט חסה גדול",
"פרגיות_corr_לינגוויני ירקות",
"פרגיות_corr_פסטה בולונז",
"פרגיות_corr_פנה קרבונרה",
"פרגיות_corr_מאצי רוזה אפונה ובייקון",
"פרגיות_corr_לזניה בולונז",
"פרגיות_corr_פסטה פירות ים",
"פרגיות_corr_פילה דג",
"פרגיות_corr_כבדי עוף ובצל מטוגן",
"טליאטה די מנזו_corr_פוקצ'ת הבית",
"טליאטה די מנזו_corr_חציל פרמז'ן",
"טליאטה די מנזו_corr_קרפצ'יו בקר אורוגולה ופרמז'ן",
"טליאטה די מנזו_corr_לינגוויני ירקות",
"טליאטה די מנזו_corr_פסטה בולונז",
"טליאטה די מנזו_corr_פנה קרבונרה",
"טליאטה די מנזו_corr_לזניה בולונז",
"טליאטה די מנזו_corr_פסטה פירות ים",
"טליאטה די מנזו_corr_פילה דג",
"טליאטה די מנזו_corr_כבדי עוף ובצל מטוגן",
"טליאטה די מנזו_corr_פרגיות",
"טירמיסו_corr_חציל פרמז'ן",
"טירמיסו_corr_לינגוויני ירקות",
"טירמיסו_corr_פסטה בולונז",
"טירמיסו_corr_פנה קרבונרה",
"טירמיסו_corr_פילה דג",
"טירמיסו_corr_פרגיות",
"לקוחות יחס יומי",
"לקוחות יחס חודשי",
"סוף שבוע"
]

# Define the features to be used by each model
features_rf = [feature for feature in features if feature not in ['לקוחות יחס יומי', 'לקוחות יחס חודשי', 'סוף שבוע']]
features_svr = features  # Assuming SVR uses all features

# פונקציה להוספת פיצ'רים נגזרים לסט האימון
def add_features(data, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    data['תאריך'] = pd.to_datetime(data['תאריך'])
    data['יום בשבוע'] = data['תאריך'].dt.dayofweek + 1
    data['יום בשבוע'] = data['יום בשבוע'].apply(lambda x: 1 if x == 7 else x + 1)
    data['חודש'] = data['תאריך'].dt.month
    data['שנה'] = data['תאריך'].dt.year

    # Create new columns in data
    data['לקוחות יחס יומי'] = data.apply(lambda x: x['מספר לקוחות'] / average_customers_per_day[x['יום בשבוע']], axis=1)
    data['לקוחות יחס חודשי'] = data.apply(lambda x: x['מספר לקוחות'] / average_customers_per_month[x['חודש']], axis=1)

    # Update the feature names in both datasets to match
    data.rename(columns={'לקחות יחס יומי': 'לקוחות יחס יומי', 'לקחות יחס חודשי': 'לקוחות יחס חודשי'}, inplace=True)

    # סופש
    data['סוף שבוע'] = data['יום בשבוע'].isin([5, 6, 7])

    # Create new features based on correlation values
    for dish1, dish2, corr_value in high_corr_pairs:
        new_feature_name = f"{dish1}_corr_{dish2}"
        data[new_feature_name] = corr_value

    return data

# Preprocessing function for SVR
def preprocess_input_svr(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({'תאריך': dates})
    data['מספר לקוחות'] = num_customers
    data = add_features(data, average_customers_per_day, average_customers_per_month, high_corr_pairs)
    scaler = MinMaxScaler()
    data['מספר לקוחות מנורמל'] = scaler.fit_transform(data[['מספר לקוחות']])
    return data

# Preprocessing function for RF and Stacking RF
def preprocess_input_rf(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({'תאריך': dates})
    data['מספר לקוחות'] = num_customers
    data = add_features(data, average_customers_per_day, average_customers_per_month, high_corr_pairs)
    return data

# Prediction function
def predict_dishes(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    results = {}
    input_data_svr = preprocess_input_svr(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs)
    input_data_rf = preprocess_input_rf(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs)

    for dish in dish_columns:
        best_model_type = optimal_models_df.loc[optimal_models_df['Dish'] == dish, 'Model'].values[0]
        if best_model_type == 'SVR':
            predictions = load_model_and_predict(dish, input_data_svr, best_model_type)
        else:
            predictions = load_model_and_predict(dish, input_data_rf, best_model_type)
        results[dish] = predictions

    return results

# Define function to load model and make predictions
def load_model_and_predict(dish, input_data, model_type):
    model_type = model_type.lower()
    if model_type == 'svr':
        model_type = 'svr'
    elif model_type == 'stacking rf':
        model_type = 'stacking_rf'
    elif model_type == 'random forest':
        model_type = 'rf'
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model_file = f'{models_path}best_{model_type}_model_{dish}.pkl'
    
    # Download the model file from the given URL
    try:
        response = requests.get(model_file)
        response.raise_for_status()  # Check if the request was successful
        model = joblib.load(BytesIO(response.content))
    except requests.exceptions.RequestException as e:
        st.error(f"Model file not found or error in loading: {model_file}, Error: {e}")
        return np.array([])
    except Exception as e:
        st.error(f"Error in loading the model: {model_file}, Error: {e}")
        return np.array([])

    if model_type == 'svr':
        features = input_data[features_svr]
    else:
        features = input_data[features_rf]

    predictions = model.predict(features)

    # המרה למספרים שלמים בעזרת np.ceil
    predictions = np.ceil(predictions).astype(int)

    return predictions

# Compute average customers per day and month
average_customers_per_day = merged_df.groupby('יום בשבוע')['מספר לקוחות'].mean().to_dict()
average_customers_per_month = merged_df.groupby('חודש')['מספר לקוחות'].mean().to_dict()

# Calculate correlation matrix for menu items (excluding metadata columns)
corr_matrix = merged_df.loc[:, dish_columns].corr()

# Threshold for high correlation
threshold = 0.5

# Create a list to store pairs of highly correlated dishes
high_corr_pairs = []

# Identify highly correlated pairs of dishes
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        corr_value = corr_matrix.iloc[i, j]
        if abs(corr_value) > threshold:
            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_value))

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

st.image(logo_url, width=200, use_container_width=False)

st.markdown('<h1 class="title">Dish Prediction Application</h1>', unsafe_allow_html=True)

st.markdown('<h2 class="header">Input Parameters</h2>', unsafe_allow_html=True)

# Input fields
start_date = st.date_input("Start Date", datetime.now())
end_date = st.date_input("End Date", datetime.now() + timedelta(days=1))
num_customers = st.number_input("Number of Customers", min_value=1, step=1)

if st.button("Predict"):
    results = predict_dishes(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs)

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
    st.markdown('<h2 class="title">Prediction Bar Chart</h2>', unsafe_allow_html=True)

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
