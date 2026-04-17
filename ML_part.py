# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression


# # ================================
# # 2. PREPROCESSING
# # ================================
# def preprocess_data(df):
#     df['start_time'] = pd.to_datetime(df['start_time'])
#     df['end_time'] = pd.to_datetime(df['end_time'])

#     df['bed_availability'] = df['bed_availability'].fillna(-1)

#     bounds = df.groupby('department')['process_debt_mins'].apply(
#         lambda x: (
#             x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)),
#             x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))
#         )
#     ).to_dict()

#     df = df[df.apply(lambda r: bounds[r['department']][0] <= r['process_debt_mins'] <= bounds[r['department']][1], axis=1)].copy()

#     df['shift_encoded'] = df['shift_time'].map({'Morning': 1, 'Evening': 2, 'Night': 3})

#     df_ml = pd.get_dummies(df, columns=['department', 'Activity'], prefix=['dept', 'act'])
#     df_ml['is_severe_delay'] = (df_ml['process_debt_mins'] > 60).astype(int)

#     return df, df_ml


# # ================================
# # 3. VISUALIZATION (RETURN FIG)
# # ================================
# def generate_visuals(df):
#     sns.set_theme(style="whitegrid")
#     fig, axs = plt.subplots(2, 2, figsize=(16, 10))

#     sns.barplot(data=df, x='department', y='process_debt_mins', hue='shift_time', ax=axs[0,0])
#     axs[0,0].set_title("Shift Impact")

#     sns.regplot(data=df, x='queue_length', y='process_debt_mins', ax=axs[0,1])

#     sns.boxplot(data=df, x='priority_level', y='process_debt_mins', ax=axs[1,0])

#     bed_df = df[df['bed_availability'] >= 0]
#     sns.lineplot(data=bed_df, x='bed_availability', y='process_debt_mins', ax=axs[1,1])

#     plt.tight_layout()
#     return fig


# # ================================
# # 4. TRAIN MODEL
# # ================================
# def train_model(df_ml):
#     cols_to_drop = ['case_id', 'start_time', 'end_time', 'shift_time',
#                     'process_debt_mins', 'is_severe_delay', 'previous_activity']

#     X = df_ml.drop(columns=[c for c in cols_to_drop if c in df_ml.columns])
#     y = df_ml['process_debt_mins']

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#     model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
#     model.fit(X_train, y_train)

#     y_pred = model.predict(X_test)

#     return model, X_train.columns.tolist(), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)


# # ================================
# # 5. PREDICTION
# # ================================
# def predict_patient(patient_data, model, feature_cols):
#     df = pd.DataFrame([patient_data])

#     for col in feature_cols:
#         if col not in df.columns:
#             df[col] = 0

#     df = df[feature_cols]
#     pred = model.predict(df)[0]

#     if pred > 80:
#         status = "CRITICAL"
#     elif pred > 60:
#         status = "WARNING"
#     else:
#         status = "NORMAL"

#     return pred, status


# # ================================
# # 6. FEATURE IMPORTANCE
# # ================================
# def feature_importance_plot(model, feature_cols):
#     importances = model.feature_importances_
#     indices = np.argsort(importances)[::-1]

#     fig = plt.figure(figsize=(10,6))
#     plt.barh(range(10), importances[indices[:10]])
#     plt.yticks(range(10), [feature_cols[i] for i in indices[:10]])
#     plt.gca().invert_yaxis()

#     return fig


# # ================================
# # 7. MODEL TOURNAMENT
# # ================================
# def train_multiple_models(X_train, y_train):
#     model_xgb = xgb.XGBRegressor().fit(X_train, y_train)
#     model_rf = RandomForestRegressor().fit(X_train, y_train)
#     model_lr = LinearRegression().fit(X_train, y_train)

#     return model_xgb, model_rf, model_lr


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


# ================================
# 2. PREPROCESSING
# ================================
def preprocess_data(df):

    # ✅ Convert time columns (safe)
    if 'start_time' in df.columns:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['start_hour'] = df['start_time'].dt.hour   # NEW feature

    if 'end_time' in df.columns:
        df['end_time'] = pd.to_datetime(df['end_time'])

    # ✅ Fill missing values
    df['bed_availability'] = df['bed_availability'].fillna(-1)

    # ✅ Remove outliers (safe)
    if 'department' in df.columns:
        bounds = df.groupby('department')['process_debt_mins'].apply(
            lambda x: (
                x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)),
                x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))
            )
        ).to_dict()

        df = df[df.apply(
            lambda r: bounds[r['department']][0] <= r['process_debt_mins'] <= bounds[r['department']][1],
            axis=1
        )].copy()

    # ✅ Encode shift
    df['shift_encoded'] = df['shift_time'].map({
        'Morning': 1,
        'Evening': 2,
        'Night': 3
    })

    # ✅ One-hot encoding (FIXED + IMPROVED)
    df_ml = pd.get_dummies(
        df,
        columns=['department', 'Activity', 'previous_activity'],
        prefix=['dept', 'act', 'prev']
    )

    # ✅ Target classification (extra feature)
    df_ml['is_severe_delay'] = (df_ml['process_debt_mins'] > 60).astype(int)

    return df, df_ml


# ================================
# 3. VISUALIZATION (RETURN FIG)
# ================================
def generate_visuals(df):
    sns.set_theme(style="whitegrid")
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))

    sns.barplot(data=df, x='department', y='process_debt_mins', hue='shift_time', ax=axs[0,0])
    axs[0,0].set_title("Shift Impact")

    sns.regplot(data=df, x='queue_length', y='process_debt_mins', ax=axs[0,1])

    sns.boxplot(data=df, x='priority_level', y='process_debt_mins', ax=axs[1,0])

    bed_df = df[df['bed_availability'] >= 0]
    sns.lineplot(data=bed_df, x='bed_availability', y='process_debt_mins', ax=axs[1,1])

    plt.tight_layout()
    return fig


# ================================
# 4. TRAIN MODEL
# ================================
def train_model(df_ml):

    cols_to_drop = [
        'case_id', 'start_time', 'end_time',
        'shift_time', 'process_debt_mins',
        'is_severe_delay'
    ]

    X = df_ml.drop(columns=[c for c in cols_to_drop if c in df_ml.columns])
    y = df_ml['process_debt_mins']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBRegressor(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return model, X_train.columns.tolist(), mean_absolute_error(y_test, y_pred), r2_score(y_test, y_pred)


# ================================
# 5. PREDICTION
# ================================
def predict_patient(patient_data, model, feature_cols):
    df = pd.DataFrame([patient_data])

    # ✅ Ensure all columns exist
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_cols]
    pred = model.predict(df)[0]

    if pred > 80:
        status = "CRITICAL"
    elif pred > 60:
        status = "WARNING"
    else:
        status = "NORMAL"

    return pred, status


# ================================
# 6. FEATURE IMPORTANCE
# ================================
def feature_importance_plot(model, feature_cols):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig = plt.figure(figsize=(10,6))
    plt.barh(range(10), importances[indices[:10]])
    plt.yticks(range(10), [feature_cols[i] for i in indices[:10]])
    plt.gca().invert_yaxis()

    return fig


# ================================
# 7. MODEL TOURNAMENT
# ================================
def train_multiple_models(X_train, y_train):
    model_xgb = xgb.XGBRegressor().fit(X_train, y_train)
    model_rf = RandomForestRegressor().fit(X_train, y_train)
    model_lr = LinearRegression().fit(X_train, y_train)

    return model_xgb, model_rf, model_lr