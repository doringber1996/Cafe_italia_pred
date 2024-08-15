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

# Define the features to be used by each model
features = ['מספר לקוחות',
'יום בשבוע',
'חודש',
'שנה',
'חציל פרמז'ן_corr_פוקצ'ת הבית',
'מבחר פטריות_corr_חציל פרמז'ן',
'שרימפס אליו פפרונצ'ינו_corr_חציל פרמז'ן',
'קרפצ'יו בקר אורוגולה ופרמז'ן_corr_פוקצ'ת הבית',
'קרפצ'יו בקר אורוגולה ופרמז'ן_corr_חציל פרמז'ן',
'קרפצ'יו בקר אורוגולה ופרמז'ן_corr_שרימפס אליו פפרונצ'ינו',
'סלט חסה גדול_corr_פוקצ'ת הבית',
'סלט חסה גדול_corr_חציל פרמז'ן',
'סלט חסה גדול_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'סלט קיסר_corr_פוקצ'ת הבית',
'סלט קיסר_corr_חציל פרמז'ן',
'לינגוויני ירקות_corr_פוקצ'ת הבית',
'לינגוויני ירקות_corr_חציל פרמז'ן',
'לינגוויני ירקות_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'לינגוויני ירקות_corr_סלט חסה גדול',
'לינגוויני ארביאטה_corr_פוקצ'ת הבית',
'לינגוויני ארביאטה_corr_חציל פרמז'ן',
'לינגוויני ארביאטה_corr_סלט חסה גדול',
'פפרדלה פטריות ושמנת_corr_פוקצ'ת הבית',
'פפרדלה פטריות ושמנת_corr_חציל פרמז'ן',
'פפרדלה פטריות ושמנת_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'פפרדלה פטריות ושמנת_corr_סלט קיסר',
'פטוצ'יני תרד גורגונזולה_corr_פוקצ'ת הבית',
'פטוצ'יני תרד גורגונזולה_corr_חציל פרמז'ן',
'פטוצ'יני תרד גורגונזולה_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'פטוצ'יני תרד גורגונזולה_corr_סלט חסה גדול',
'פטוצ'יני תרד גורגונזולה_corr_פפרדלה פטריות ושמנת',
'פסטה בולונז_corr_פוקצ'ת הבית',
'פסטה בולונז_corr_חציל פרמז'ן',
'פסטה בולונז_corr_שרימפס אליו פפרונצ'ינו',
'פסטה בולונז_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'פסטה בולונז_corr_סלט חסה גדול',
'פסטה בולונז_corr_לינגוויני ירקות',
'פסטה בולונז_corr_לינגוויני ארביאטה',
'פסטה בולונז_corr_פפרדלה פטריות ושמנת',
'פסטה בולונז_corr_פטוצ'יני תרד גורגונזולה',
'פנה קרבונרה_corr_פוקצ'ת הבית',
'פנה קרבונרה_corr_חציל פרמז'ן',
'פנה קרבונרה_corr_שרימפס אליו פפרונצ'ינו',
'פנה קרבונרה_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'פנה קרבונרה_corr_סלט חסה גדול',
'פנה קרבונרה_corr_לינגוויני ירקות',
'פנה קרבונרה_corr_לינגוויני ארביאטה',
'פנה קרבונרה_corr_פפרדלה פטריות ושמנת',
'פנה קרבונרה_corr_פטוצ'יני תרד גורגונזולה',
'פנה קרבונרה_corr_פסטה בולונז',
'מאצי רוזה אפונה ובייקון_corr_פוקצ'ת הבית',
'מאצי רוזה אפונה ובייקון_corr_חציל פרמז'ן',
'מאצי רוזה אפונה ובייקון_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'מאצי רוזה אפונה ובייקון_corr_פסטה בולונז',
'מאצי רוזה אפונה ובייקון_corr_פנה קרבונרה',
'לזניה בולונז_corr_פוקצ'ת הבית',
'לזניה בולונז_corr_חציל פרמז'ן',
'לזניה בולונז_corr_שרימפס אליו פפרונצ'ינו',
'לזניה בולונז_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'לזניה בולונז_corr_סלט חסה גדול',
'לזניה בולונז_corr_לינגוויני ירקות',
'לזניה בולונז_corr_פפרדלה פטריות ושמנת',
'לזניה בולונז_corr_פטוצ'יני תרד גורגונזולה',
'לזניה בולונז_corr_פסטה בולונז',
'לזניה בולונז_corr_פנה קרבונרה',
'לזניה בולונז_corr_מאצי רוזה אפונה ובייקון',
'טורטלוני_corr_פוקצ'ת הבית',
'טורטלוני_corr_חציל פרמז'ן',
'טורטלוני_corr_לזניה בולונז',
'פסטה פירות ים_corr_פוקצ'ת הבית',
'פסטה פירות ים_corr_חציל פרמז'ן',
'פסטה פירות ים_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'פסטה פירות ים_corr_סלט חסה גדול',
'פסטה פירות ים_corr_לינגוויני ירקות',
'פסטה פירות ים_corr_פסטה בולונז',
'פסטה פירות ים_corr_פנה קרבונרה',
'פסטה פירות ים_corr_מאצי רוזה אפונה ובייקון',
'פסטה פירות ים_corr_לזניה בולונז',
'פילה דג_corr_פוקצ'ת הבית',
'פילה דג_corr_חציל פרמז'ן',
'פילה דג_corr_שרימפס אליו פפרונצ'ינו',
'פילה דג_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'פילה דג_corr_סלט חסה גדול',
'פילה דג_corr_לינגוויני ירקות',
'פילה דג_corr_פטוצ'יני תרד גורגונזולה',
'פילה דג_corr_פסטה בולונז',
'פילה דג_corr_פנה קרבונרה',
'פילה דג_corr_מאצי רוזה אפונה ובייקון',
'פילה דג_corr_לזניה בולונז',
'פילה דג_corr_פסטה פירות ים',
'כבדי עוף ובצל מטוגן_corr_פוקצ'ת הבית',
'כבדי עוף ובצל מטוגן_corr_חציל פרמז'ן',
'כבדי עוף ובצל מטוגן_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'כבדי עוף ובצל מטוגן_corr_סלט חסה גדול',
'כבדי עוף ובצל מטוגן_corr_לינגוויני ירקות',
'כבדי עוף ובצל מטוגן_corr_פסטה בולונז',
'כבדי עוף ובצל מטוגן_corr_פנה קרבונרה',
'כבדי עוף ובצל מטוגן_corr_לזניה בולונז',
'כבדי עוף ובצל מטוגן_corr_פסטה פירות ים',
'כבדי עוף ובצל מטוגן_corr_פילה דג',
'פרגיות_corr_פוקצ'ת הבית',
'פרגיות_corr_חציל פרמז'ן',
'פרגיות_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'פרגיות_corr_סלט חסה גדול',
'פרגיות_corr_לינגוויני ירקות',
'פרגיות_corr_פסטה בולונז',
'פרגיות_corr_פנה קרבונרה',
'פרגיות_corr_מאצי רוזה אפונה ובייקון',
'פרגיות_corr_לזניה בולונז',
'פרגיות_corr_פסטה פירות ים',
'פרגיות_corr_פילה דג',
'פרגיות_corr_כבדי עוף ובצל מטוגן',
'טליאטה די מנזו_corr_פוקצ'ת הבית',
'טליאטה די מנזו_corr_חציל פרמז'ן',
'טליאטה די מנזו_corr_קרפצ'יו בקר אורוגולה ופרמז'ן',
'טליאטה די מנזו_corr_לינגוויני ירקות',
'טליאטה די מנזו_corr_פסטה בולונז',
'טליאטה די מנזו_corr_פנה קרבונרה',
'טליאטה די מנזו_corr_לזניה בולונז',
'טליאטה די מנזו_corr_פסטה פירות ים',
'טליאטה די מנזו_corr_פילה דג',
'טליאטה די מנזו_corr_כבדי עוף ובצל מטוגן',
'טליאטה די מנזו_corr_פרגיות',
'נמסיס_corr_פסטה בולונז',
'טירמיסו_corr_פוקצ'ת הבית',
'טירמיסו_corr_פסטה בולונז',
'טירמיסו_corr_פנה קרבונרה',
'טירמיסו_corr_לזניה בולונז',
'לקוחות יחס יומי',
'לקוחות יחס חודשי',
'סוף שבוע']

features_stacking_rf = features
features_rf = features

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
    new_features = {}
    for dish1, dish2, corr_value in high_corr_pairs:
        new_feature_name = f"{dish1}_corr_{dish2}"
        new_features[new_feature_name] = corr_value
    
    new_features_df = pd.DataFrame(new_features, index=data.index)
    data = pd.concat([data, new_features_df], axis=1)
    
    return data


# Preprocessing function for RF and Stacking RF
def preprocess_input_rf(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    data = pd.DataFrame({'תאריך': dates})
    data['מספר לקוחות'] = num_customers
    data = add_features(data, average_customers_per_day, average_customers_per_month, high_corr_pairs)
    return data

def predict_dishes(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs):
    results = {}
    input_data_rf = preprocess_input_rf(start_date, end_date, num_customers, average_customers_per_day, average_customers_per_month, high_corr_pairs)

    for dish in dish_columns:
        best_model_type = optimal_models_df.loc[optimal_models_df['Dish'] == dish, 'Model'].values[0]
        predictions = load_model_and_predict(dish, input_data_rf, best_model_type)
        results[dish] = predictions

    return results
    
def load_model_and_predict(dish, input_data, model_type):
    model_type = model_type.lower()
    if model_type == 'stacking rf':
        model_file = f'{models_path}best_stacking_rf_model_{dish}.pkl'
        features = input_data[features_stacking_rf]
    elif model_type == 'random forest':
        model_file = f'{models_path}best_rf_model_{dish}.pkl'
        features = input_data[features_rf]
    else:
        raise ValueError(f"Unknown model type: {model_type}")

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

    predictions = model.predict(features)
    predictions = np.ceil(predictions).astype(int)

    return predictions
    
# Compute average customers per day and month
average_customers_per_day= {1: 264.2421052631579, 2: 284.775, 3: 294.87704918032784, 4: 296.3606557377049, 5: 352.64516129032256, 6: 354.008064516129, 7: 357.3414634146341}
average_customers_per_month = {1: 334.28409090909093, 2: 350.51851851851853, 3: 337.4623655913978, 4: 313.58024691358025, 5: 309.14606741573033, 6: 307.3448275862069, 7: 312.4561403508772, 8: 336.41379310344826, 9: 310.47058823529414, 10: 216.94545454545454, 11: 282.43859649122805, 12: 352.66129032258067}

# Create a list to store pairs of highly correlated dishes
high_corr_pairs=[("חציל פרמז'ן", "פוקצ'ת הבית", 0.6568337984040734),
 ('מבחר פטריות', "חציל פרמז'ן", 0.5077749102121348),
 ("שרימפס אליו פפרונצ'ינו", "חציל פרמז'ן", 0.538642804066582),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן", "פוקצ'ת הבית", 0.6322719162386367),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן", "חציל פרמז'ן", 0.6593154527711517),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן","שרימפס אליו פפרונצ'ינו",0.5103282673404597),
 ('סלט חסה גדול', "פוקצ'ת הבית", 0.5683993444929052),
 ('סלט חסה גדול', "חציל פרמז'ן", 0.5676932533052638),
 ('סלט חסה גדול', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5508409850253048),
 ('סלט קיסר', "פוקצ'ת הבית", 0.5449902723315474),
 ('סלט קיסר', "חציל פרמז'ן", 0.5219419203511221),
 ('לינגוויני ירקות', "פוקצ'ת הבית", 0.5293301277900869),
 ('לינגוויני ירקות', "חציל פרמז'ן", 0.6085315945929876),
 ('לינגוויני ירקות', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5620332087544316),
 ('לינגוויני ירקות', 'סלט חסה גדול', 0.5316583294692059),
 ('לינגוויני ארביאטה', "פוקצ'ת הבית", 0.5034830351004693),
 ('לינגוויני ארביאטה', "חציל פרמז'ן", 0.54985579207627),
 ('לינגוויני ארביאטה', 'סלט חסה גדול', 0.5409726193229318),
 ('פפרדלה פטריות ושמנת', "פוקצ'ת הבית", 0.608212901070681),
 ('פפרדלה פטריות ושמנת', "חציל פרמז'ן", 0.5480887681473657),
 ('פפרדלה פטריות ושמנת', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5073436706011916),
 ('פפרדלה פטריות ושמנת', 'סלט קיסר', 0.5322459660391311),
 ("פטוצ'יני תרד גורגונזולה", "פוקצ'ת הבית", 0.6029687847225681),
 ("פטוצ'יני תרד גורגונזולה", "חציל פרמז'ן", 0.5962489384635405),
 ("פטוצ'יני תרד גורגונזולה","קרפצ'יו בקר אורוגולה ופרמז'ן",0.5746398318998693),
 ("פטוצ'יני תרד גורגונזולה", 'סלט חסה גדול', 0.5029080729306342),
 ("פטוצ'יני תרד גורגונזולה", 'פפרדלה פטריות ושמנת', 0.5804810860718181),
 ('פסטה בולונז', "פוקצ'ת הבית", 0.620234824415326),
 ('פסטה בולונז', "חציל פרמז'ן", 0.7022306350007375),
 ('פסטה בולונז', "שרימפס אליו פפרונצ'ינו", 0.5454641144722612),
 ('פסטה בולונז', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6438648118616215),
 ('פסטה בולונז', 'סלט חסה גדול', 0.5949398841509792),
 ('פסטה בולונז', 'לינגוויני ירקות', 0.617225062792499),
 ('פסטה בולונז', 'לינגוויני ארביאטה', 0.5381653246159598),
 ('פסטה בולונז', 'פפרדלה פטריות ושמנת', 0.5087045880955757),
 ('פסטה בולונז', "פטוצ'יני תרד גורגונזולה", 0.5576052721076455),
 ('פנה קרבונרה', "פוקצ'ת הבית", 0.6238095865295545),
 ('פנה קרבונרה', "חציל פרמז'ן", 0.6755967256327364),
 ('פנה קרבונרה', "שרימפס אליו פפרונצ'ינו", 0.5327292204437987),
 ('פנה קרבונרה', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6471330766349769),
 ('פנה קרבונרה', 'סלט חסה גדול', 0.5978415253917421),
 ('פנה קרבונרה', 'לינגוויני ירקות', 0.5959503997904237),
 ('פנה קרבונרה', 'לינגוויני ארביאטה', 0.5444900067970028),
 ('פנה קרבונרה', 'פפרדלה פטריות ושמנת', 0.5307575135593271),
 ('פנה קרבונרה', "פטוצ'יני תרד גורגונזולה", 0.5600489689815721),
 ('פנה קרבונרה', 'פסטה בולונז', 0.6745066092536932),
 ('מאצי רוזה אפונה ובייקון', "פוקצ'ת הבית", 0.5092721867392906),
 ('מאצי רוזה אפונה ובייקון', "חציל פרמז'ן", 0.5419151070030906),
 ('מאצי רוזה אפונה ובייקון',"קרפצ'יו בקר אורוגולה ופרמז'ן",0.534267939974032),
 ('מאצי רוזה אפונה ובייקון', 'פסטה בולונז', 0.5775209348773254),
 ('מאצי רוזה אפונה ובייקון', 'פנה קרבונרה', 0.5367526755419962),
 ('לזניה בולונז', "פוקצ'ת הבית", 0.6031509155618798),
 ('לזניה בולונז', "חציל פרמז'ן", 0.7074179916386176),
 ('לזניה בולונז', "שרימפס אליו פפרונצ'ינו", 0.5266125795402976),
 ('לזניה בולונז', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6869139808117352),
 ('לזניה בולונז', 'סלט חסה גדול', 0.5604448004143872),
 ('לזניה בולונז', 'לינגוויני ירקות', 0.6000436265504676),
 ('לזניה בולונז', 'פפרדלה פטריות ושמנת', 0.5067477649817782),
 ('לזניה בולונז', "פטוצ'יני תרד גורגונזולה", 0.597217827495053),
 ('לזניה בולונז', 'פסטה בולונז', 0.7096621640402699),
 ('לזניה בולונז', 'פנה קרבונרה', 0.6721317959323688),
 ('לזניה בולונז', 'מאצי רוזה אפונה ובייקון', 0.5665966342622929),
 ('טורטלוני', "פוקצ'ת הבית", 0.5092296898074763),
 ('טורטלוני', "חציל פרמז'ן", 0.5204074113571779),
 ('טורטלוני', 'לזניה בולונז', 0.5073351111762855),
 ('פסטה פירות ים', "פוקצ'ת הבית", 0.5471973382570239),
 ('פסטה פירות ים', "חציל פרמז'ן", 0.6290497327725796),
 ('פסטה פירות ים', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5709362840213662),
 ('פסטה פירות ים', 'סלט חסה גדול', 0.5769469341357238),
 ('פסטה פירות ים', 'לינגוויני ירקות', 0.5444125637922037),
 ('פסטה פירות ים', 'פסטה בולונז', 0.6356129665375524),
 ('פסטה פירות ים', 'פנה קרבונרה', 0.6268286775220714),
 ('פסטה פירות ים', 'מאצי רוזה אפונה ובייקון', 0.5266763681686005),
 ('פסטה פירות ים', 'לזניה בולונז', 0.6061648462463428),
 ('פילה דג', "פוקצ'ת הבית", 0.590039087236486),
 ('פילה דג', "חציל פרמז'ן", 0.6859461398616314),
 ('פילה דג', "שרימפס אליו פפרונצ'ינו", 0.5202279718513025),
 ('פילה דג', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6650772277845262),
 ('פילה דג', 'סלט חסה גדול', 0.5609388258449611),
 ('פילה דג', 'לינגוויני ירקות', 0.5931200353944255),
 ('פילה דג', "פטוצ'יני תרד גורגונזולה", 0.5664437274819032),
 ('פילה דג', 'פסטה בולונז', 0.6905438589672412),
 ('פילה דג', 'פנה קרבונרה', 0.6617488241476286),
 ('פילה דג', 'מאצי רוזה אפונה ובייקון', 0.5160838293673813),
 ('פילה דג', 'לזניה בולונז', 0.7157529656361171),
 ('פילה דג', 'פסטה פירות ים', 0.6361593124217784),
 ('כבדי עוף ובצל מטוגן', "פוקצ'ת הבית", 0.522475521608485),
 ('כבדי עוף ובצל מטוגן', "חציל פרמז'ן", 0.5820941231353166),
 ('כבדי עוף ובצל מטוגן', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5665356577057544),
 ('כבדי עוף ובצל מטוגן', 'סלט חסה גדול', 0.5104963906179226),
 ('כבדי עוף ובצל מטוגן', 'לינגוויני ירקות', 0.5309470677036174),
 ('כבדי עוף ובצל מטוגן', 'פסטה בולונז', 0.6156096840705065),
 ('כבדי עוף ובצל מטוגן', 'פנה קרבונרה', 0.5504989213577978),
 ('כבדי עוף ובצל מטוגן', 'לזניה בולונז', 0.5881126681799829),
 ('כבדי עוף ובצל מטוגן', 'פסטה פירות ים', 0.5605467421010687),
 ('כבדי עוף ובצל מטוגן', 'פילה דג', 0.589531640630601),
 ('פרגיות', "פוקצ'ת הבית", 0.5538961626523163),
 ('פרגיות', "חציל פרמז'ן", 0.6606467160022924),
 ('פרגיות', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6013856891506398),
 ('פרגיות', 'סלט חסה גדול', 0.549900537565337),
 ('פרגיות', 'לינגוויני ירקות', 0.5670853538675623),
 ('פרגיות', 'פסטה בולונז', 0.6899623323784217),
 ('פרגיות', 'פנה קרבונרה', 0.6214646456254598),
 ('פרגיות', 'מאצי רוזה אפונה ובייקון', 0.5345608327071687),
 ('פרגיות', 'לזניה בולונז', 0.6781291048016839),
 ('פרגיות', 'פסטה פירות ים', 0.6204779416937027),
 ('פרגיות', 'פילה דג', 0.6727745277350367),
 ('פרגיות', 'כבדי עוף ובצל מטוגן', 0.6177137609791438),
 ('טליאטה די מנזו', "פוקצ'ת הבית", 0.5508450254827048),
 ('טליאטה די מנזו', "חציל פרמז'ן", 0.6019896735786916),
 ('טליאטה די מנזו', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5909553890046991),
 ('טליאטה די מנזו', 'לינגוויני ירקות', 0.5130159252359634),
 ('טליאטה די מנזו', 'פסטה בולונז', 0.5978611328963324),
 ('טליאטה די מנזו', 'פנה קרבונרה', 0.6088559238735584),
 ('טליאטה די מנזו', 'לזניה בולונז', 0.6206079220612539),
 ('טליאטה די מנזו', 'פסטה פירות ים', 0.5652758049447213),
 ('טליאטה די מנזו', 'פילה דג', 0.604648680423571),
 ('טליאטה די מנזו', 'כבדי עוף ובצל מטוגן', 0.5553178723389186),
 ('טליאטה די מנזו', 'פרגיות', 0.5888395616416938),
 ('נמסיס', 'פסטה בולונז', 0.5093502571041615),
 ('טירמיסו', "פוקצ'ת הבית", 0.5053188649502104),
 ('טירמיסו', 'פסטה בולונז', 0.5415369953576205),
 ('טירמיסו', 'פנה קרבונרה', 0.5167671953839011),
 ('טירמיסו', 'לזניה בולונז', 0.5106018270401965)]

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
