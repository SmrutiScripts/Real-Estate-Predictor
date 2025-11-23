# import pandas as pd
# housing = pd.read_csv("melb_data.csv")
# housing.head()
# housing.info()
# housing['Rooms']
# housing.describe()
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20,15))
# import numpy as np
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
# train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {(len(train_set))}\nRows in test set: {(len(test_set))}\n")
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split = train_test_split(housing, test_size=0.2, random_state=42)
# print(f"Rows in train set: {(len(train_set))}\nRows in test set: {(len(test_set))}\n")
# from sklearn.model_selection import StratifiedShuffleSplit
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# housing["Landsize_cat"] = pd.qcut(housing["Landsize"], q=5, labels=False)
# for train_index, test_index in split.split(housing, housing['Landsize_cat']):
#     start_train_set = housing.loc[train_index]
#     start_test_set = housing.loc[test_index]
# start_test_set['Landsize_cat'].value_counts()
# start_train_set['Landsize_cat'].value_counts()
# housing.corr(numeric_only=True)
# corr_matrix = housing.corr(numeric_only=True)
# corr_matrix['Price'].sort_values(ascending=False)
# from pandas.plotting import scatter_matrix
# attributes = ["Price", "Bathroom", "Longtitude", "YearBuilt"]
# scatter_matrix(housing[attributes], figsize = (12,8))
# housing.plot(kind="scatter",x="Price", y="YearBuilt", alpha=0.8)
# housing["Tax"] = housing["Price"]/housing["BuildingArea"]
# housing.head()
# corr_matrix = housing.corr(numeric_only=True)
# corr_matrix['Price'].sort_values(ascending=False)
# housing.plot(kind="scatter",x="Price", y="Tax", alpha=0.8)
# print(start_train_set.columns)
# housing = start_test_set.drop("Distance", axis =1)
# housing_labels = start_test_set["Distance"].copy()
# housing.dropna(subset=["Car"]) #Option 1
# housing.shape
# housing.drop("Car", axis=1) #option 2
# median = housing["Car"].median()
# print(median)
# housing["Car"].fillna(median)
# print(housing.shape)
# import pandas as pd
# housing = housing.select_dtypes(include=['number'])
# print(housing.head())
# import numpy as np
# import pandas as pd
# housing = housing.replace([np.inf, -np.inf], np.nan)
# housing = housing.select_dtypes(include=['number'])
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="median")
# housing_imputed = imputer.fit_transform(housing)
# housing = pd.DataFrame(housing_imputed, columns=housing.columns)
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy = "median")
# imputer.fit(housing)
# imputer.statistics_
# imputer.statistics_.shape
# X = imputer.transform(housing)
# housing_tr = pd.DataFrame(X, columns=housing.columns)
# housing_tr.describe()
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler 
# my_pipeline = Pipeline([
#     ('imputer', SimpleImputer(strategy="median")),
#     ('std_scaler', StandardScaler()),
# ])
# housing_num_tr = my_pipeline.fit_transform(housing_tr)
# housing_num_tr
# housing_num_tr.shape
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
# model = RandomForestRegressor()
# model.fit(housing_num_tr,housing_labels)
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# prepared_data = my_pipeline.transform(some_data)
# model.predict(prepared_data)
# some_labels
# list(some_labels)
# from sklearn.metrics import mean_squared_error
# housing_predictions = model.predict(housing_num_tr)
# mse = mean_squared_error(housing_labels, housing_predictions)
# rmse = np.sqrt(mse)
# rmse
# from sklearn.model_selection import cross_val_score
# scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv = 10)
# rmse_scores = np.sqrt(-scores)
# rmse_scores
# def print_scores(scores):
#     print("Scores: ", scores)
#     print("Mean: ", scores.mean())
#     print("Standard deviation: ", scores.std())
# print_scores(rmse_scores)
# import joblib
# joblib.dump(model, 'Mehers.pkl') 
# from sklearn.metrics import mean_squared_error
# X_test = start_test_set.drop("Distance", axis=1).select_dtypes(include=['number'])
# X_test_prepared = my_pipeline.transform(X_test)
# Y_test = start_test_set["Distance"].copy()
# X_test_prepared = my_pipeline.transform(X_test)
# final_predictions = model.predict(X_test_prepared)
# final_mse = mean_squared_error(Y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)
# print(final_predictions, list(Y_test))
# final_rmse
# prepared_data[0]
# from joblib import dump, load
# import numpy as np
# model = load('Mehers.pkl')
# # features = np.array([[0.09456289,  0.27781026, -0.38110948,  0.12011576, -0.74974834,
# #        -0.63573023, -0.09055916, -0.03434547,  0.10656111,  0.59302247,
# #         0.13076916,  0.34424249,  0.00104159]])
# features = np.array([[0.09456289,  0.27781026, -0.38110948,  0.12011576]])
# print(model.predict(features))


# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "melb_data.csv"     # path to your dataset
TARGET_COL = "Price"
FEATURE_COLUMNS = ["Rooms", "Bedroom2", "Bathroom", "Car"]
RANDOM_STATE = 42

# ----------------------------
# 1. Load data
# ----------------------------
housing = pd.read_csv(CSV_PATH)

# Keep only rows where target and Landsize exist
housing = housing.dropna(subset=[TARGET_COL, "Landsize"])

# ----------------------------
# 2. Create stratified category for Landsize
# ----------------------------
housing["Landsize_cat"] = pd.qcut(
    housing["Landsize"],
    q=5,
    labels=False,
    duplicates="drop"  # safe if some bins collapse
)

# ----------------------------
# 3. Stratified train/test split
# ----------------------------
split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=RANDOM_STATE
)

for train_idx, test_idx in split.split(housing, housing["Landsize_cat"]):
    train_set = housing.loc[train_idx].copy()
    test_set = housing.loc[test_idx].copy()

# Drop helper column
train_set = train_set.drop("Landsize_cat", axis=1)
test_set = test_set.drop("Landsize_cat", axis=1)

# ----------------------------
# 4. Select features & target, drop rows with missing feature values
# ----------------------------
train_df = train_set[FEATURE_COLUMNS + [TARGET_COL]].dropna()
test_df = test_set[FEATURE_COLUMNS + [TARGET_COL]].dropna()

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df[TARGET_COL]

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df[TARGET_COL]

print("Training rows:", len(X_train))
print("Test rows     :", len(X_test))

# ----------------------------
# 5. Train RandomForestRegressor
# ----------------------------
model = RandomForestRegressor(
    random_state=RANDOM_STATE,
    n_estimators=200,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ----------------------------
# 6. Evaluate on test set
# ----------------------------
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Test RMSE: {rmse:,.2f}")

# ----------------------------
# 7. Save model + feature metadata
# ----------------------------
joblib.dump(
    (model, FEATURE_COLUMNS),
    "rf_model_4features.pkl"
)

print(" Model saved to rf_model_4features.pkl")





