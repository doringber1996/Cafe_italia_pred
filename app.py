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

high_corr_pairs= [("חציל פרמז'ן", "פוקצ'ת הבית", 0.6565827190858701),
 ("שרימפס אליו פפרונצ'ינו", "פוקצ'ת הבית", 0.5248384812797835),
 ("שרימפס אליו פפרונצ'ינו", "חציל פרמז'ן", 0.515993581226792),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן", "פוקצ'ת הבית", 0.6436292223677104),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן", "חציל פרמז'ן", 0.6593618567085427),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן",
  "שרימפס אליו פפרונצ'ינו",
  0.5277064876906393),
 ('סלט חסה גדול', "פוקצ'ת הבית", 0.5343944232635433),
 ('סלט חסה גדול', "חציל פרמז'ן", 0.5544257682609077),
 ('סלט חסה גדול', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5215650900146656),
 ('סלט קיסר', "פוקצ'ת הבית", 0.5757594089379622),
 ('לינגוויני ירקות', "פוקצ'ת הבית", 0.5576848896545049),
 ('לינגוויני ירקות', "חציל פרמז'ן", 0.6009414943137296),
 ('לינגוויני ירקות', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5506080503505856),
 ('לינגוויני ירקות', 'סלט חסה גדול', 0.5215693998600409),
 ('לינגוויני ארביאטה', "פוקצ'ת הבית", 0.5393605109823749),
 ('לינגוויני ארביאטה', "חציל פרמז'ן", 0.5018731290166865),
 ('לינגוויני ארביאטה', 'סלט חסה גדול', 0.5485374295500287),
 ('פפרדלה פטריות ושמנת', "פוקצ'ת הבית", 0.6429618421425549),
 ('פפרדלה פטריות ושמנת', "חציל פרמז'ן", 0.5764528573651745),
 ('פפרדלה פטריות ושמנת', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5334163525252209),
 ('פפרדלה פטריות ושמנת', 'סלט קיסר', 0.5762320890266133),
 ('פפרדלה פטריות ושמנת', 'לינגוויני ירקות', 0.5103338255006614),
 ("פטוצ'יני תרד גורגונזולה", "פוקצ'ת הבית", 0.6478898030849269),
 ("פטוצ'יני תרד גורגונזולה", "חציל פרמז'ן", 0.588904576255679),
 ("פטוצ'יני תרד גורגונזולה",
  "קרפצ'יו בקר אורוגולה ופרמז'ן",
  0.5562236493310385),
 ("פטוצ'יני תרד גורגונזולה", 'סלט קיסר', 0.50615870299973),
 ("פטוצ'יני תרד גורגונזולה", 'פפרדלה פטריות ושמנת', 0.6014616584153057),
 ('פסטה בולונז', "פוקצ'ת הבית", 0.6272319316617186),
 ('פסטה בולונז', "חציל פרמז'ן", 0.6974235431367888),
 ('פסטה בולונז', 'מבחר פטריות', 0.5137840641734933),
 ('פסטה בולונז', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6369625035220927),
 ('פסטה בולונז', 'סלט חסה גדול', 0.5999447006773511),
 ('פסטה בולונז', 'לינגוויני ירקות', 0.6110078909686497),
 ('פסטה בולונז', 'לינגוויני ארביאטה', 0.5241915518821497),
 ('פסטה בולונז', 'פפרדלה פטריות ושמנת', 0.558394342445489),
 ('פסטה בולונז', "פטוצ'יני תרד גורגונזולה", 0.5529545771268348),
 ('פנה קרבונרה', "פוקצ'ת הבית", 0.6450344496119355),
 ('פנה קרבונרה', "חציל פרמז'ן", 0.6802694316937159),
 ('פנה קרבונרה', "שרימפס אליו פפרונצ'ינו", 0.5050822002561902),
 ('פנה קרבונרה', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6412943678759895),
 ('פנה קרבונרה', 'סלט חסה גדול', 0.5547214360496243),
 ('פנה קרבונרה', 'לינגוויני ירקות', 0.5918279346756121),
 ('פנה קרבונרה', 'לינגוויני ארביאטה', 0.5069199421733191),
 ('פנה קרבונרה', 'פפרדלה פטריות ושמנת', 0.5606041284288834),
 ('פנה קרבונרה', "פטוצ'יני תרד גורגונזולה", 0.5565234300511093),
 ('פנה קרבונרה', 'פסטה בולונז', 0.6857412883469566),
 ('מאצי רוזה אפונה ובייקון', "פוקצ'ת הבית", 0.5256022136786052),
 ('מאצי רוזה אפונה ובייקון', "חציל פרמז'ן", 0.5298685320950809),
 ('מאצי רוזה אפונה ובייקון',
  "קרפצ'יו בקר אורוגולה ופרמז'ן",
  0.5139858438488668),
 ('מאצי רוזה אפונה ובייקון', 'פסטה בולונז', 0.5631835763373715),
 ('מאצי רוזה אפונה ובייקון', 'פנה קרבונרה', 0.5398154975562333),
 ('לזניה בולונז', "פוקצ'ת הבית", 0.6174086841448755),
 ('לזניה בולונז', "חציל פרמז'ן", 0.6898410313715436),
 ('לזניה בולונז', "שרימפס אליו פפרונצ'ינו", 0.5168807608287183),
 ('לזניה בולונז', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6577143557429178),
 ('לזניה בולונז', 'סלט חסה גדול', 0.5318154845879475),
 ('לזניה בולונז', 'לינגוויני ירקות', 0.611254984765526),
 ('לזניה בולונז', 'פפרדלה פטריות ושמנת', 0.5502742927288252),
 ('לזניה בולונז', "פטוצ'יני תרד גורגונזולה", 0.5948261388914582),
 ('לזניה בולונז', 'פסטה בולונז', 0.6807865380128831),
 ('לזניה בולונז', 'פנה קרבונרה', 0.6788736102804237),
 ('לזניה בולונז', 'מאצי רוזה אפונה ובייקון', 0.5146724612396721),
 ('טורטלוני', "חציל פרמז'ן", 0.5249806002164218),
 ('טורטלוני', 'לזניה בולונז', 0.5266242811569384),
 ('פסטה פירות ים', "פוקצ'ת הבית", 0.535378777400553),
 ('פסטה פירות ים', "חציל פרמז'ן", 0.6076352528613673),
 ('פסטה פירות ים', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5442240663867249),
 ('פסטה פירות ים', 'סלט חסה גדול', 0.5285798634643856),
 ('פסטה פירות ים', 'לינגוויני ירקות', 0.5405386534070394),
 ('פסטה פירות ים', 'פסטה בולונז', 0.6275652298862847),
 ('פסטה פירות ים', 'פנה קרבונרה', 0.5930874635924093),
 ('פסטה פירות ים', 'לזניה בולונז', 0.5811638035327952),
 ('פילה דג', "פוקצ'ת הבית", 0.6070567762875468),
 ('פילה דג', "חציל פרמז'ן", 0.671403256571381),
 ('פילה דג', "שרימפס אליו פפרונצ'ינו", 0.5041182383482723),
 ('פילה דג', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6523017167622069),
 ('פילה דג', 'סלט חסה גדול', 0.5479662068136641),
 ('פילה דג', 'לינגוויני ירקות', 0.6014326693988462),
 ('פילה דג', 'פפרדלה פטריות ושמנת', 0.5000939226707158),
 ('פילה דג', "פטוצ'יני תרד גורגונזולה", 0.5412647700700772),
 ('פילה דג', 'פסטה בולונז', 0.6884737335623423),
 ('פילה דג', 'פנה קרבונרה', 0.6367598938070034),
 ('פילה דג', 'לזניה בולונז', 0.7103577565802528),
 ('פילה דג', 'פסטה פירות ים', 0.6001051811455674),
 ('כבדי עוף ובצל מטוגן', "פוקצ'ת הבית", 0.5645272858960989),
 ('כבדי עוף ובצל מטוגן', "חציל פרמז'ן", 0.6373370306852646),
 ('כבדי עוף ובצל מטוגן', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5823212631962509),
 ('כבדי עוף ובצל מטוגן', 'סלט חסה גדול', 0.536251748275619),
 ('כבדי עוף ובצל מטוגן', 'לינגוויני ירקות', 0.5377983887208598),
 ('כבדי עוף ובצל מטוגן', 'פפרדלה פטריות ושמנת', 0.5048982469474594),
 ('כבדי עוף ובצל מטוגן', 'פסטה בולונז', 0.630928073988651),
 ('כבדי עוף ובצל מטוגן', 'פנה קרבונרה', 0.6050181747411866),
 ('כבדי עוף ובצל מטוגן', 'לזניה בולונז', 0.6266151781381102),
 ('כבדי עוף ובצל מטוגן', 'פסטה פירות ים', 0.5582680506014548),
 ('כבדי עוף ובצל מטוגן', 'פילה דג', 0.6471516895492848),
 ('פרגיות', "פוקצ'ת הבית", 0.585742476298543),
 ('פרגיות', "חציל פרמז'ן", 0.667082507822566),
 ('פרגיות', 'מבחר פטריות', 0.5050076264525208),
 ('פרגיות', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.619268299831509),
 ('פרגיות', 'סלט חסה גדול', 0.5483798319087448),
 ('פרגיות', 'לינגוויני ירקות', 0.5610402083468606),
 ('פרגיות', 'פסטה בולונז', 0.7005767246830686),
 ('פרגיות', 'פנה קרבונרה', 0.6148132101821968),
 ('פרגיות', 'מאצי רוזה אפונה ובייקון', 0.5462428072613771),
 ('פרגיות', 'לזניה בולונז', 0.6669806922049658),
 ('פרגיות', 'פסטה פירות ים', 0.6089251453417761),
 ('פרגיות', 'פילה דג', 0.672409241956693),
 ('פרגיות', 'כבדי עוף ובצל מטוגן', 0.6269219293452397),
 ('טליאטה די מנזו', "פוקצ'ת הבית", 0.5628324782069245),
 ('טליאטה די מנזו', "חציל פרמז'ן", 0.5873282451948568),
 ('טליאטה די מנזו', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5970576227111678),
 ('טליאטה די מנזו', 'לינגוויני ירקות', 0.5084720052289979),
 ('טליאטה די מנזו', 'פסטה בולונז', 0.5630292296564107),
 ('טליאטה די מנזו', 'פנה קרבונרה', 0.5776781642642779),
 ('טליאטה די מנזו', 'לזניה בולונז', 0.5983587617183078),
 ('טליאטה די מנזו', 'פסטה פירות ים', 0.5359505033549944),
 ('טליאטה די מנזו', 'פילה דג', 0.6176317609604395),
 ('טליאטה די מנזו', 'כבדי עוף ובצל מטוגן', 0.5685115659799039),
 ('טליאטה די מנזו', 'פרגיות', 0.572610336252279),
 ('נמסיס', 'פסטה בולונז', 0.5060936614785484),
 ('טירמיסו', 'פסטה בולונז', 0.5111866769316015)]
    
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
