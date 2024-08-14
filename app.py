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

# Load train data from GitHub
train_data_url = 'https://raw.githubusercontent.com/doringber1996/Cafe_italia_pred/main/train_data.csv'
train_data = pd.read_csv(train_data_url)

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

# Create a list to store pairs of highly correlated dishes
high_corr_pairs=[("חציל פרמז'ן", "פוקצ'ת הבית", 0.6794204034027995),
 ("שרימפס אליו פפרונצ'ינו", "חציל פרמז'ן", 0.5301521280757641),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן", "פוקצ'ת הבית", 0.6280344388460439),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן", "חציל פרמז'ן", 0.6579865009472007),
 ("קרפצ'יו בקר אורוגולה ופרמז'ן",
  "שרימפס אליו פפרונצ'ינו",
  0.5166536148541543),
 ('סלט חסה גדול', "פוקצ'ת הבית", 0.5410454006666334),
 ('סלט חסה גדול', "חציל פרמז'ן", 0.5629834271538905),
 ('סלט חסה גדול', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5375521865806292),
 ('סלט קיסר', "פוקצ'ת הבית", 0.5745612814570632),
 ('סלט קיסר', "חציל פרמז'ן", 0.5231266230293472),
 ('לינגוויני ירקות', "פוקצ'ת הבית", 0.5461450093905766),
 ('לינגוויני ירקות', "חציל פרמז'ן", 0.5831427738124462),
 ('לינגוויני ירקות', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5527458487008483),
 ('לינגוויני ירקות', 'סלט חסה גדול', 0.5450249341093092),
 ('לינגוויני ארביאטה', "פוקצ'ת הבית", 0.527482702393315),
 ('לינגוויני ארביאטה', "חציל פרמז'ן", 0.5242285328431594),
 ('לינגוויני ארביאטה', 'סלט חסה גדול', 0.5242901761810056),
 ('פפרדלה פטריות ושמנת', "פוקצ'ת הבית", 0.6549089018422338),
 ('פפרדלה פטריות ושמנת', "חציל פרמז'ן", 0.5734400667870724),
 ('פפרדלה פטריות ושמנת', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5042464382393756),
 ('פפרדלה פטריות ושמנת', 'סלט חסה גדול', 0.5046809182740327),
 ('פפרדלה פטריות ושמנת', 'סלט קיסר', 0.5587692346393143),
 ("פטוצ'יני תרד גורגונזולה", "פוקצ'ת הבית", 0.6106123233284888),
 ("פטוצ'יני תרד גורגונזולה", "חציל פרמז'ן", 0.6018334628902513),
 ("פטוצ'יני תרד גורגונזולה",
  "קרפצ'יו בקר אורוגולה ופרמז'ן",
  0.5552737545191738),
 ("פטוצ'יני תרד גורגונזולה", 'פפרדלה פטריות ושמנת', 0.5824326540937602),
 ('פסטה בולונז', "פוקצ'ת הבית", 0.6115376682817356),
 ('פסטה בולונז', "חציל פרמז'ן", 0.7171825635824794),
 ('פסטה בולונז', "שרימפס אליו פפרונצ'ינו", 0.5298076122103216),
 ('פסטה בולונז', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6409031253365083),
 ('פסטה בולונז', 'סלט חסה גדול', 0.5771716661236195),
 ('פסטה בולונז', 'לינגוויני ירקות', 0.617689095187458),
 ('פסטה בולונז', 'לינגוויני ארביאטה', 0.5231956738110581),
 ('פסטה בולונז', 'פפרדלה פטריות ושמנת', 0.5352258609973652),
 ('פסטה בולונז', "פטוצ'יני תרד גורגונזולה", 0.5335596219034152),
 ('פנה קרבונרה', "פוקצ'ת הבית", 0.6616321479982459),
 ('פנה קרבונרה', "חציל פרמז'ן", 0.6832177052903644),
 ('פנה קרבונרה', "שרימפס אליו פפרונצ'ינו", 0.5305934805226464),
 ('פנה קרבונרה', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6672415373209374),
 ('פנה קרבונרה', 'סלט חסה גדול', 0.5728025057422158),
 ('פנה קרבונרה', 'סלט קיסר', 0.500599886183611),
 ('פנה קרבונרה', 'לינגוויני ירקות', 0.6194567182380913),
 ('פנה קרבונרה', 'לינגוויני ארביאטה', 0.5387331864425771),
 ('פנה קרבונרה', 'פפרדלה פטריות ושמנת', 0.567956094665199),
 ('פנה קרבונרה', "פטוצ'יני תרד גורגונזולה", 0.5781466422207676),
 ('פנה קרבונרה', 'פסטה בולונז', 0.6935278833259493),
 ('מאצי רוזה אפונה ובייקון', "פוקצ'ת הבית", 0.5177139619781851),
 ('מאצי רוזה אפונה ובייקון', "חציל פרמז'ן", 0.5605613479153516),
 ('מאצי רוזה אפונה ובייקון',
  "קרפצ'יו בקר אורוגולה ופרמז'ן",
  0.5325434409314381),
 ('מאצי רוזה אפונה ובייקון', 'פסטה בולונז', 0.5740857245505439),
 ('מאצי רוזה אפונה ובייקון', 'פנה קרבונרה', 0.5872379238712979),
 ('לזניה בולונז', "פוקצ'ת הבית", 0.6101439681575348),
 ('לזניה בולונז', "חציל פרמז'ן", 0.6828726850422728),
 ('לזניה בולונז', "שרימפס אליו פפרונצ'ינו", 0.518158727700968),
 ('לזניה בולונז', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6680051689900914),
 ('לזניה בולונז', 'סלט חסה גדול', 0.5287660178829815),
 ('לזניה בולונז', 'לינגוויני ירקות', 0.5918103997484588),
 ('לזניה בולונז', 'פפרדלה פטריות ושמנת', 0.5074023471420633),
 ('לזניה בולונז', "פטוצ'יני תרד גורגונזולה", 0.5605010852215706),
 ('לזניה בולונז', 'פסטה בולונז', 0.6772364578286771),
 ('לזניה בולונז', 'פנה קרבונרה', 0.6829311557717408),
 ('לזניה בולונז', 'מאצי רוזה אפונה ובייקון', 0.5214684983821148),
 ('טורטלוני', "פוקצ'ת הבית", 0.5222944808797815),
 ('פסטה פירות ים', "פוקצ'ת הבית", 0.5166159208035473),
 ('פסטה פירות ים', "חציל פרמז'ן", 0.5886571957412403),
 ('פסטה פירות ים', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.548136547658278),
 ('פסטה פירות ים', 'סלט חסה גדול', 0.520081158609227),
 ('פסטה פירות ים', 'לינגוויני ירקות', 0.514906937879105),
 ('פסטה פירות ים', 'פסטה בולונז', 0.6228841445939878),
 ('פסטה פירות ים', 'פנה קרבונרה', 0.6101535938517949),
 ('פסטה פירות ים', 'לזניה בולונז', 0.5764639801235586),
 ('פילה דג', "פוקצ'ת הבית", 0.6206400452274098),
 ('פילה דג', "חציל פרמז'ן", 0.6955725426882632),
 ('פילה דג', "שרימפס אליו פפרונצ'ינו", 0.5204687166297993),
 ('פילה דג', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6695851423795264),
 ('פילה דג', 'סלט חסה גדול', 0.5676938193088936),
 ('פילה דג', 'לינגוויני ירקות', 0.6232073895574112),
 ('פילה דג', "פטוצ'יני תרד גורגונזולה", 0.5441753324513122),
 ('פילה דג', 'פסטה בולונז', 0.6981785766492467),
 ('פילה דג', 'פנה קרבונרה', 0.6961030472870292),
 ('פילה דג', 'מאצי רוזה אפונה ובייקון', 0.5137599929976016),
 ('פילה דג', 'לזניה בולונז', 0.7187522275977571),
 ('פילה דג', 'פסטה פירות ים', 0.5872801477422178),
 ('כבדי עוף ובצל מטוגן', "פוקצ'ת הבית", 0.5480470314749896),
 ('כבדי עוף ובצל מטוגן', "חציל פרמז'ן", 0.6180104412422653),
 ('כבדי עוף ובצל מטוגן', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6017754541113155),
 ('כבדי עוף ובצל מטוגן', 'סלט חסה גדול', 0.5469321958097055),
 ('כבדי עוף ובצל מטוגן', 'לינגוויני ירקות', 0.5613704698285991),
 ('כבדי עוף ובצל מטוגן', 'פסטה בולונז', 0.6468101728937069),
 ('כבדי עוף ובצל מטוגן', 'פנה קרבונרה', 0.630010516654044),
 ('כבדי עוף ובצל מטוגן', 'מאצי רוזה אפונה ובייקון', 0.50887922395586),
 ('כבדי עוף ובצל מטוגן', 'לזניה בולונז', 0.6237024408423071),
 ('כבדי עוף ובצל מטוגן', 'פסטה פירות ים', 0.5579309028895644),
 ('כבדי עוף ובצל מטוגן', 'פילה דג', 0.6522255773688835),
 ('פרגיות', "פוקצ'ת הבית", 0.5674439841400465),
 ('פרגיות', "חציל פרמז'ן", 0.6771703917921598),
 ('פרגיות', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.6146450140303347),
 ('פרגיות', 'סלט חסה גדול', 0.5516387102279146),
 ('פרגיות', 'לינגוויני ירקות', 0.600584354243212),
 ('פרגיות', 'פסטה בולונז', 0.6625973170269633),
 ('פרגיות', 'פנה קרבונרה', 0.645879605104764),
 ('פרגיות', 'מאצי רוזה אפונה ובייקון', 0.5234780649269968),
 ('פרגיות', 'לזניה בולונז', 0.676777357215479),
 ('פרגיות', 'פסטה פירות ים', 0.6109159584362646),
 ('פרגיות', 'פילה דג', 0.6910621548735392),
 ('פרגיות', 'כבדי עוף ובצל מטוגן', 0.6482244993744979),
 ('טליאטה די מנזו', "פוקצ'ת הבית", 0.550722907669781),
 ('טליאטה די מנזו', "חציל פרמז'ן", 0.5989169398270107),
 ('טליאטה די מנזו', "קרפצ'יו בקר אורוגולה ופרמז'ן", 0.5880554034016106),
 ('טליאטה די מנזו', 'לינגוויני ירקות', 0.5251756117456499),
 ('טליאטה די מנזו', 'פסטה בולונז', 0.5729251548783911),
 ('טליאטה די מנזו', 'פנה קרבונרה', 0.6148807316854287),
 ('טליאטה די מנזו', 'לזניה בולונז', 0.5897116815608815),
 ('טליאטה די מנזו', 'פסטה פירות ים', 0.5168409527648044),
 ('טליאטה די מנזו', 'פילה דג', 0.6231760649430558),
 ('טליאטה די מנזו', 'כבדי עוף ובצל מטוגן', 0.5733339320350187),
 ('טליאטה די מנזו', 'פרגיות', 0.5966212609834128),
 ('טירמיסו', "חציל פרמז'ן", 0.5019856860194244),
 ('טירמיסו', 'לינגוויני ירקות', 0.5293910138015708),
 ('טירמיסו', 'פסטה בולונז', 0.539569095254237),
 ('טירמיסו', 'פנה קרבונרה', 0.5576175435814857),
 ('טירמיסו', 'פילה דג', 0.5155064783394511),
 ('טירמיסו', 'פרגיות', 0.5306007273916324)]

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
