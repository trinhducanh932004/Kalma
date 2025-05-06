import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore") 

use_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage',
    'Number_of_Ads', 'Listening_Time_minutes', 'Podcast_Name', 'Episode_Title',
    'Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment'
]
data = pd.read_csv(r"C:/Users/admin/Documents/Zalo Received Files/PTCTG/train.csv", usecols=use_cols)

num_imputer = SimpleImputer(strategy='median')
data[['Episode_Length_minutes', 'Guest_Popularity_percentage']] = num_imputer.fit_transform(
    data[['Episode_Length_minutes', 'Guest_Popularity_percentage']])
data['Number_of_Ads'] = data['Number_of_Ads'].fillna(0)
data['Listening_Time_minutes'] = data['Listening_Time_minutes'].bfill()

categorical_cols = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

data['Podcast_Name_Length'] = data['Podcast_Name'].str.len()
data['Episode_Title_Length'] = data['Episode_Title'].str.len()
data.drop(columns=['Podcast_Name', 'Episode_Title'], inplace=True)

numeric_cols = [
    'Episode_Length_minutes', 'Host_Popularity_percentage',
    'Guest_Popularity_percentage', 'Number_of_Ads',
    'Listening_Time_minutes', 'Podcast_Name_Length', 'Episode_Title_Length'
]
scaler = StandardScaler()
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

friday_label = label_encoders['Publication_Day'].transform(['Friday'])[0]
friday_data = data[data["Publication_Day"] == friday_label].copy()

if friday_data.empty:
    print("Error: No data found for Friday.")
    exit()

views = friday_data['Listening_Time_minutes'].ffill().values

kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
kf = kf.em(views, n_iter=10)
kalman_filtered, _ = kf.filter(views)

model = ARIMA(views, order=(2, 1, 2)) 
model_fit = model.fit()
arima_pred = model_fit.predict(start=1, end=len(views)-1, typ="levels")

views_trimmed = views[1:]
kalman_trimmed = kalman_filtered[1:]

def evaluate(true_values, predicted_values, name=""):
    mae = mean_absolute_error(true_values, predicted_values)
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    print(f"===== {name} =====")
    print(f"MAE : {mae:.4f}")
    print(f"MSE : {mse:.4f}")
    print(f"RMSE: {rmse:.4f}\n") 

evaluate(views_trimmed, kalman_trimmed, "Kalman Filter")
evaluate(views_trimmed, arima_pred, "ARIMA(2,1,2)")

plt.figure(figsize=(14, 8))
plt.plot(views_trimmed, label="Original Listening Time", alpha=0.4)
plt.plot(kalman_trimmed, label="Kalman Filter", linewidth=2)
plt.plot(arima_pred, label="ARIMA(2,1,2)", linewidth=2, linestyle="--")
plt.title("Listening Time on Fridays - Kalman Filter vs ARIMA")
plt.xlabel("Time Index")
plt.ylabel("Listening Time (minutes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()