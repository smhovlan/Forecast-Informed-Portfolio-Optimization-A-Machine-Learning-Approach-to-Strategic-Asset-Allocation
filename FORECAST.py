# [LIBRARIES]
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import optuna
from sklearn.preprocessing import MinMaxScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, log_loss, brier_score_loss,
    precision_recall_curve, auc, roc_curve
)
import warnings
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import shap
import random
import os
from datetime import timedelta
from scipy.interpolate import make_interp_spline


# [GLOBAL SEED]
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

warnings.filterwarnings("ignore")
shap.initjs()


# [TECHNICAL INDICATORS]
def compute_extra_features(df):

    df['trend_slope_20'] = df['Close'].rolling(window=20).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True).fillna(0)
    df['volatility_20'] = df['Close'].rolling(window=20).std().fillna(0)
    df['volume_trend'] = (df['Volume'].rolling(window=5).mean() / df['Volume'].rolling(window=20).mean()).fillna(1)
    return df

def compute_RSI(df, window=14):

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(0)

    return df


def compute_MACD(df, fast=12, slow=26, signal=9):

    df['EMA_fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    return df


def compute_Bollinger_Bands(df, window=20, num_std=2):

    df['BB_MA'] = df['Close'].rolling(window).mean()
    df['BB_std'] = df['Close'].rolling(window).std()
    df['BB_upper'] = df['BB_MA'] + num_std * df['BB_std']
    df['BB_lower'] = df['BB_MA'] - num_std * df['BB_std']

    return df


def compute_Stochastic(df, window=14):

    df['L14'] = df['Low'].rolling(window=window).min()
    df['H14'] = df['High'].rolling(window=window).max()
    df['Stochastic'] = 100 * (df['Close'] - df['L14']) / (df['H14'] - df['L14'])
    df['Stochastic'] = df['Stochastic'].fillna(0)

    return df


def compute_ATR(df, window=14):

    df['H-L'] = df['High'] - df['Low']
    df['H-C'] = abs(df['High'] - df['Close'].shift())
    df['L-C'] = abs(df['Low'] - df['Close'].shift())
    df['TR'] = df[['H-L', 'H-C', 'L-C']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=window).mean()
    df['ATR'] = df['ATR'].fillna(0)
    df.drop(['H-L', 'H-C', 'L-C', 'TR'], axis=1, inplace=True)

    return df


def compute_CCI(df, window=20):

    TP = (df['High'] + df['Low'] + df['Close']) / 3
    ma = TP.rolling(window=window).mean()
    md = TP.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (TP - ma) / (0.015 * md)
    df['CCI'] = df['CCI'].fillna(0)

    return df


def compute_technical_indicators(df):
    
    for i in range(1, 11):
        df[f"ln_close_diff_{i}"] = np.log(df["Close"] / df["Close"].shift(i)).fillna(0)

    for i in range(4):
        df[f"ln_high_open_{i}"] = np.log(df["High"].shift(i) / df["Open"].shift(i)).fillna(0)
        df[f"ln_low_open_{i}"] = np.log(df["Low"].shift(i) / df["Open"].shift(i)).fillna(0)
    df["ema_14"] = df["Close"].ewm(span=14, adjust=False).mean().fillna(0)
    df["ema_50"] = df["Close"].ewm(span=50, adjust=False).mean().fillna(0)
    df["momentum_10"] = (df["Close"] - df["Close"].shift(10)).fillna(0)

    return df


def compute_all_technical_indicators(df):

    df = compute_technical_indicators(df)
    df = compute_RSI(df)
    df = compute_MACD(df)
    df = compute_Bollinger_Bands(df)
    df = compute_Stochastic(df)
    df = compute_ATR(df)
    df = compute_CCI(df)
    df = compute_extra_features(df)

    return df



# [VISUALIZE TECHNICAL INDICATORS]
def plot_technical_indicators(df):

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.07, row_heights=[0.7, 0.3],
        subplot_titles=("Close Price with Technical Indicators", "RSI and Volatility")
    )

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["Close"], name="Close Price",
        line=dict(color="black", width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["ema_14"], name="EMA 14",
        line=dict(color="blue", width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["ema_50"], name="EMA 50",
        line=dict(color="orange", width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["BB_upper"], name="Bollinger Upper",
        line=dict(color="green", width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["BB_lower"], name="Bollinger Lower",
        line=dict(color="red", width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["RSI"], name="RSI",
        line=dict(color="black", width = 1)), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["volatility_20"], name="Volatility (20d)",
        line=dict(color="red", width=1)), row=2, col=1)

    fig.update_layout(
        height=750,
        font=dict(family="Arial", size=14),
        margin=dict(t=60, b=40),
        legend=dict(orientation="h", y=-0.25))
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI / Volatility", row=2, col=1)
    fig.show()



# [FUNCTIONS]
def shap_analysis(model, X, top_k = 15):

    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    shap_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP_Importance': shap_importance
    }).sort_values(by='SHAP_Importance', ascending=False)

    fig = go.Figure(
        go.Bar(
            x=shap_df['SHAP_Importance'].head(top_k),
            y=shap_df['Feature'].head(top_k),
            orientation='h'
        )
    )
    fig.update_layout(title=f"Top SHAP Feature Importances", height=500)
    fig.show()
    return shap_df


def find_best_k_shap(X_train, y_train, X_valid, y_valid, max_k = 20):
    base_model = XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=42)
    base_model.fit(X_train, y_train)

    shap_df = shap_analysis(base_model, X_train, top_k=max_k)
    best_auc = -1
    best_k = None
    auc_scores = []

    for k in range(3, max_k + 1):
        top_features = shap_df["Feature"].head(k).tolist()
        model = XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=42)
        model.fit(X_train[top_features], y_train)
        y_val_pred = model.predict_proba(X_valid[top_features])[:, 1]
        auc_score = roc_auc_score(y_valid, y_val_pred)
        auc_scores.append((k, auc_score))

        if auc_score > best_auc:
            best_auc = auc_score
            best_k = k

    print(f"\nBest k based on ROC AUC: {best_k} (AUC = {best_auc:.4f})")
    return best_k, shap_df


def plot_probability_distribution(y_proba, bins=10):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=y_proba,
        nbinsx=bins,
        marker_color='navy',
        marker_line_color='black',
        marker_line_width=1.2
    ))
    fig.update_layout(
        title="Distribution of Predicted Probabilities",
        xaxis=dict(
            title="Probability for Up (Class 1)",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        yaxis=dict(
            title="Frequency",
            title_font=dict(size=18),
            tickfont=dict(size=16)
        ),
        bargap=0.05,
        template="plotly_white"
    )
    fig.show()



    counts, bin_edges = np.histogram(y_proba, bins = bins)
    distribution_df = pd.DataFrame({
        "Interval": [f"{round(bin_edges[i], 2)} - {round(bin_edges[i+1], 2)}" for i in range(len(counts))],
        "Frequence": counts
    })
    print("Table for probability intervals")
    print(distribution_df.to_string(index=False))


def analyze_stock(stock, start_year=2015):
    df = pd.read_csv(stock)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[df["Date"].dt.year >= start_year].copy()
    df.index = range(len(df))
    
    threshold = 0.02
    df["future_return"] = (df["Close"].shift(-20) - df["Close"]) / df["Close"]
    df["Target_Class"] = np.where(

    df["future_return"] > threshold, 1,
    np.where(df["future_return"] < -threshold, 0, np.nan)

    )
    
    df = df.dropna(subset=["Target_Class"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.03,
                        subplot_titles=("OHLC Chart", "Volume"))
    
    fig.add_trace(
        go.Ohlc(x=df.Date, open=df.Open, high=df.High,
                low=df.Low, close=df.Close, name="OHLC Chart"),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df.Date, y=df.Volume, mode="lines", name="Volume"),
        row=2, col=1
    )
    
    fig.update_layout(
    height=700,
    title="OHLC og Volume",
    showlegend=True)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.show()

    return df



def evaluate_class_balance(y):

    counts = y.value_counts()
    majority_class = counts.idxmax()
    imbalance_ratio = counts.max() / counts.min()
    print(f"Class distribution: {counts.to_dict()} — Ratio: {imbalance_ratio:.2f}")

    if imbalance_ratio > 1.5:
        print(f"Significant imbalance detected. Majority class: {majority_class}")
    else:
        print("Class balance is acceptable.")

    return counts


def apply_resampling(X, y, method="SMOTE"):

    if method == "SMOTE":
        print("Applying SMOTE...")
        resampler = SMOTE(random_state=42)

    elif method == "undersample":
        print("Applying Random Undersampling...")
        resampler = RandomUnderSampler(random_state=42)

    elif method is None:
        print("No resampling applied.")
        return X, y
    else:
        raise ValueError("Unsupported resampling method")

    X_res, y_res = resampler.fit_resample(X, y)
    print(f"Resampled: {sum(y_res==0)} class 0 / {sum(y_res==1)} class 1")

    return X_res, y_res


def compare_imbalance_methods(X_train_raw, y_train, methods=[None, "SMOTE", "undersample"]):

    results = []
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
    
    for method in methods:
        X_resampled, y_resampled = apply_resampling(X_train_scaled, y_train, method)
        model = XGBClassifier(

            max_depth=3, learning_rate=0.01, n_estimators=300, 
            reg_alpha=0.1, reg_lambda=1, 
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )

        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X_resampled, y_resampled, cv=tscv, scoring="f1")
        results.append({

            "Method": method if method else "None",
            "Mean_F1": scores.mean(),
            "Std_F1": scores.std(),
            "Class_0": sum(y_resampled == 0),
            "Class_1": sum(y_resampled == 1)

        })

    results_df = pd.DataFrame(results).sort_values(by="Mean_F1", ascending=False)
    print("Resampling Method Comparison:")

    print(results_df)
    return results_df


def analyze_overfitting_cv(X, y, best_params, resample_method=None):
    tscv = TimeSeriesSplit(n_splits=5)
    train_f1, val_f1 = [], []
    train_auc, val_auc = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train_fold = X.iloc[train_idx].copy()
        X_val_fold = X.iloc[val_idx].copy()
        y_train_fold = y.iloc[train_idx].copy()
        y_val_fold = y.iloc[val_idx].copy()

        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_fold = pd.DataFrame(scaler.fit_transform(X_train_fold), columns=X.columns)
        X_val_fold = pd.DataFrame(scaler.transform(X_val_fold), columns=X.columns)

        if resample_method:
            X_train_fold, y_train_fold = apply_resampling(X_train_fold, y_train_fold, method=resample_method)

        model_fold = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="auc", random_state=42)
        model_fold.fit(X_train_fold, y_train_fold, early_stopping_rounds = 10, 
                       eval_set=[(X_val_fold, y_val_fold)], verbose=False)

        train_pred_class = model_fold.predict(X_train_fold)
        val_pred_class = model_fold.predict(X_val_fold)
        train_f1.append(f1_score(y_train_fold, train_pred_class))
        val_f1.append(f1_score(y_val_fold, val_pred_class))

        train_pred_proba = model_fold.predict_proba(X_train_fold)[:, 1]
        val_pred_proba = model_fold.predict_proba(X_val_fold)[:, 1]
        train_auc.append(roc_auc_score(y_train_fold, train_pred_proba))
        val_auc.append(roc_auc_score(y_val_fold, val_pred_proba))


    print("\nF1 Scores:")
    print(f"Train:      {np.round(train_f1, 4)}")
    print(f"Validation: {np.round(val_f1, 4)}")
    print(f"Train avg: {np.mean(train_f1):.4f} | Validation avg: {np.mean(val_f1):.4f}")

    print("\nROC AUC Scores:")
    print(f"Train:      {np.round(train_auc, 4)}")
    print(f"Validation: {np.round(val_auc, 4)}")
    print(f"Train avg: {np.mean(train_auc):.4f} | Validation avg: {np.mean(val_auc):.4f}")


def evaluate_model(X_train, y_train, X_valid, y_valid, X_test, y_test, test_dates, df_raw=None):

    shap_model = XGBClassifier(use_label_encoder=False, eval_metric="auc", random_state=42)
    shap_model.fit(X_train, y_train)

    best_k, shap_df = find_best_k_shap(X_train, y_train, X_valid, y_valid, max_k = 30)
    top_features = shap_df["Feature"].head(best_k).tolist()

    X_train_sel = X_train[top_features]
    X_valid_sel = X_valid[top_features]
    X_test_sel = X_test[top_features]

    scale_pos_weight = sum(y_train == 0) / sum(y_train == 1)

    def objective(trial):

        params = {
            "n_estimators": trial.suggest_categorical("n_estimators", [100, 200, 300, 400]),
            "learning_rate": trial.suggest_float("learning_rate", 0.002, 0.02, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 6),
            "subsample": trial.suggest_float("subsample", 0.5, 0.9, step=0.05),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9, step=0.1),
            "gamma": trial.suggest_float("gamma", 0.0, 0.4, step=0.05),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0, step=0.1),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0, step=0.5),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "use_label_encoder": False,
            "random_state": 42
        }
        model = XGBClassifier(**params)
        tscv = TimeSeriesSplit(n_splits=5)
        scores = cross_val_score(model, X_train_sel, y_train, cv=tscv, scoring="roc_auc")
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials = 50)

    print("Best Params:", study.best_params)
    print("Best ROC AUC value:", study.best_value)

    best_params = study.best_params.copy()
    best_params.pop("eval_metric", None)

    decay_rate = 0.001
    sample_weights = np.exp(-decay_rate * (len(X_train_sel) - np.arange(len(X_train_sel))))

    final_model = XGBClassifier(**best_params, use_label_encoder=False, eval_metric="auc", random_state=42)
    final_model.fit(
    X_train_sel, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_valid_sel, y_valid)],
    verbose=False
)

    y_pred_proba = final_model.predict_proba(X_test_sel)[:, 1]
    df_plot = df_raw.loc[X_test_sel.index].copy()

    roc = roc_auc_score(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)
    brier = brier_score_loss(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    print("\nSOFT CLASSIFICATION METRICS:")
    print(f"ROC AUC:      {roc:.4f}")
    print(f"Log Loss:     {logloss:.4f}")
    print(f"Brier Score:  {brier:.4f}")
    print(f"PR AUC:       {pr_auc:.4f}")

    plot_probability_distribution(y_pred_proba)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=test_dates,
        y=y_pred_proba,
        name="Predicted P(>2% return)",
        mode="lines",
        line=dict(color="#ff4c4c", width = 1, shape = "linear"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Probability: %{y:.2f}<extra></extra>"))

    fig.update_layout(
        title="<b>Predicted Probabilities for Positive Return (20-Day Horizon)</b>",
        xaxis_title="Date",
        yaxis_title="Predicted Probability",
        height=500,
        font=dict(family="Arial", size=14),
        margin=dict(l=40, r=40, t=60, b=40))
    fig.show()



    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=fpr,
        y=tpr,
        mode='lines',
        name='ROC Curve',
        line=dict(color='black')
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode='lines',
        name='Random Guess',
        line=dict(dash='dash')
        ))

    fig.update_layout(
        title='ROC Curve',
        xaxis=dict(
        title='False Positive Rate',
        title_font=dict(size=18),
        tickfont=dict(size=16)
        ),
        yaxis=dict(
        title='True Positive Rate',
        title_font=dict(size=18),
        tickfont=dict(size=16))
        )
    fig.show()


    return final_model, top_features, y_pred_proba, test_dates, best_params


# [BASELINE: DummyClassifier]
def evaluate_dummy_classifier(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    y_pred = dummy.predict(X_test)
    y_proba = dummy.predict_proba(X_test)[:, 1] if hasattr(dummy, "predict_proba") else np.full(len(y_test), 0.5)

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    return {
        "Model": "Dummy",
        "ROC AUC": roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else np.nan,
        "Log Loss": log_loss(y_test, y_proba, labels=[0, 1]),
        "Brier Score": brier_score_loss(y_test, y_proba),
        "PR AUC": auc(recall, precision)
    }


# [BASELINE: Logistic Regression]
def evaluate_logistic_baseline(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    return {
        "Model": "Logistic",
        "ROC AUC": roc_auc_score(y_test, y_proba),
        "Log Loss": log_loss(y_test, y_proba),
        "Brier Score": brier_score_loss(y_test, y_proba),
        "PR AUC": auc(recall, precision)
    }


# [COMPARE ALL MODELS]
def compare_models_table(*results):
    results_df = pd.DataFrame(results)
    print("COMPARISON OF MODELS:")
    print(results_df.round(4).to_string(index=False))
    return results_df


def rolling_scale(X, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X_scaled, scaler
    else:
        return pd.DataFrame(scaler.transform(X), columns=X.columns)


# [MAIN PIPELINE]
def main():
    
    #TECHNOLOGY
    AAPL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\AAPL-history-daily-ten-yrs.csv"
    MSFT = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\MSFT-history-daily-ten-yrs.csv"
    NVDA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\NVDA-history-daily-ten-yrs.csv"
    TSM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\TSM-history-daily-ten-yrs.csv"
    SAP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\SAP-history-daily-ten-yrs.csv"
    AVGO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\AVGO-history-daily-ten-yrs.csv"
    STM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\STM-history-daily-ten-yrs.csv"
    ORCL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\ORCL-history-daily-ten-yrs.csv"
    INTC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\INTC-history-daily-ten-yrs.csv"
    AMD = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\AMD-history-daily-ten-yrs.csv"
    PLTR = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Technology\PLTR-history-daily-ten-yrs.csv"

    #FINANCE
    BAC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\BAC-history-daily-ten-yrs.csv"
    DB = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\DB-history-daily-ten-yrs.csv"
    HSBC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\HSBC-history-daily-ten-yrs.csv"
    JPM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\JPM-history-daily-ten-yrs.csv"
    MS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\MS-history-daily-ten-yrs.csv"
    SAN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\SAN-history-daily-ten-yrs.csv"
    AXP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\AXP-history-daily-ten-yrs.csv"
    BLK = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\BLK-history-daily-ten-yrs.csv"
    BNPQF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\BNPQF-history-daily-ten-yrs.csv"
    GS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\GS-history-daily-ten-yrs.csv"
    ING = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\ING-history-daily-ten-yrs.csv"
    LYG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\LYG-history-daily-ten-yrs.csv"
    UBS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\UBS-history-daily-ten-yrs.csv"
    WFC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Finance\WFC-history-daily-ten-yrs.csv"

    #CONSUMER STAPLES
    AMZN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\AMZN-history-daily-ten-yrs.csv"
    BUD = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\BUD-history-daily-ten-yrs.csv"
    DEO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\DEO-history-daily-ten-yrs.csv"
    TM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\TM-history-daily-ten-yrs.csv"
    KO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\KO-history-daily-ten-yrs.csv"
    KR = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\KR-history-daily-ten-yrs.csv"
    MDLZ = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\MDLZ-history-daily-ten-yrs.csv"
    NKE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\NKE-history-daily-ten-yrs.csv"
    NSRGF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\NSRGF-history-daily-ten-yrs.csv"
    PEP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\PEP-history-daily-ten-yrs.csv"
    PG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\PG-history-daily-ten-yrs.csv"
    TSN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\TSN-history-daily-ten-yrs.csv"
    UL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Consumer Staples\UL-history-daily-ten-yrs.csv"

    #ENERGY
    BP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\BP-history-daily-ten-yrs.csv"
    COP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\COP-history-daily-ten-yrs.csv"
    CVX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\CVX-history-daily-ten-yrs.csv"
    SHEL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\SHEL-history-daily-ten-yrs.csv"
    SLB = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\SLB-history-daily-ten-yrs.csv"
    TTE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\TTE-history-daily-ten-yrs.csv"
    XOM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\XOM-history-daily-ten-yrs.csv"
    DVN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\DVN-history-daily-ten-yrs.csv"
    EOG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\EOG-history-daily-ten-yrs.csv"
    EQNR = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\EQNR-history-daily-ten-yrs.csv"
    HAL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\HAL-history-daily-ten-yrs.csv"
    MPC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\MPC-history-daily-ten-yrs.csv"
    OXY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\OXY-history-daily-ten-yrs.csv"
    VLO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Energy\VLO-history-daily-ten-yrs.csv"

    #INDUSTRIAL
    ABBNY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\ABBNY-history-daily-ten-yrs.csv"
    EADSF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\EADSF-history-daily-ten-yrs.csv"
    GE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\GE-history-daily-ten-yrs.csv"
    SBGSF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\SBGSF-history-daily-ten-yrs.csv"
    TSLA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\TSLA-history-daily-ten-yrs.csv"
    CAT = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\CAT-history-daily-ten-yrs.csv"
    ETN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\ETN-history-daily-ten-yrs.csv"
    MMM = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\MMM-history-daily-ten-yrs.csv"
    SDVKY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Industrial\SDVKY-history-daily-ten-yrs.csv"

    #COMMUNICATION SERVICES
    META = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\META-history-daily-ten-yrs.csv"
    CMCSA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\CMCSA-history-daily-ten-yrs.csv"
    DTEGY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\DTEGY-history-daily-ten-yrs.csv"
    GOOGL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\GOOGL-history-daily-ten-yrs.csv"
    NFLX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\NFLX-history-daily-ten-yrs.csv"
    VOD = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\VOD-history-daily-ten-yrs.csv"
    DIS = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\DIS-history-daily-ten-yrs.csv"
    ORANY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\ORANY-history-daily-ten-yrs.csv"
    PSO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\PSO-history-daily-ten-yrs.csv"
    ITVPY = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\ITVPY-history-daily-ten-yrs.csv"
    VZ = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\VZ-history-daily-ten-yrs.csv"
    T = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Communication Services\T-history-daily-ten-yrs.csv"

    #AEROSPACE & DEFENCE
    BAESF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\BAESF-history-daily-ten-yrs.csv"
    RTX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\RTX-history-daily-ten-yrs.csv"
    AXON = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\AXON-history-daily-ten-yrs.csv"
    BA = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\BA-history-daily-ten-yrs.csv"
    LHX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\LHX-history-daily-ten-yrs.csv"
    NOC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\NOC-history-daily-ten-yrs.csv"
    TXT = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Aerospace & Defence\TXT-history-daily-ten-yrs.csv"

    #ESG
    BEP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\BEP-history-daily-ten-yrs.csv"
    ENPH = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\ENPH-history-daily-ten-yrs.csv"
    FSLR = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\FSLR-history-daily-ten-yrs.csv"
    IBDSF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\IBDSF-history-daily-ten-yrs.csv"
    PLUG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\PLUG-history-daily-ten-yrs.csv"
    VWSYF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\VWSYF-history-daily-ten-yrs.csv"
    NEE = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\NEE-history-daily-ten-yrs.csv"
    SEDG = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\SEDG-history-daily-ten-yrs.csv"
    RUN = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\RUN-history-daily-ten-yrs.csv"
    BLDP = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\ESG\VWSYF-history-daily-ten-yrs.csv"

    #SHIPPING
    DSDVF = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\DSDVF-history-daily-ten-yrs.csv"
    FRO = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\FRO-history-daily-ten-yrs.csv"
    MATX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\MATX-history-daily-ten-yrs.csv"
    DAC = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\DAC-history-daily-ten-yrs.csv"
    GOGL = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\MATX-history-daily-ten-yrs.csv"
    KEX = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\KEX-history-daily-ten-yrs.csv"
    SBLK = r"C:\Users\stokr\OneDrive\Skrivebord\Project\Data\Shipping\MATX-history-daily-ten-yrs.csv"
    

    stock = NVDA
    df = analyze_stock(stock)
    df = compute_all_technical_indicators(df)
    plot_technical_indicators(df)

    df = df.dropna().iloc[33:-1].reset_index(drop=True)
    df_dates = df["Date"].copy()

    target_counts = df["Target_Class"].value_counts().sort_index()
    total = target_counts.sum()
    percentages = (target_counts / total * 100).round(2)


    target_col = "Target_Class"
    feature_cols = [col for col in df.columns if col not in ["Date", "Open", "High", "Low", "Close", "Adj. Close", "Volume", "Change", "future_return"]]
    df_features = df[feature_cols]

    valid_size = 0.15
    test_size = 0.15
    n_total = df_features.shape[0]
    test_split_idx = int(n_total * (1 - test_size))
    valid_split_idx = int(n_total * (1 - (valid_size + test_size)))

    X_raw = df_features.drop(columns=[target_col])
    y = df_features[target_col]

    X_train_raw = X_raw.iloc[:valid_split_idx]
    X_valid_raw = X_raw.iloc[valid_split_idx:test_split_idx]
    X_test_raw  = X_raw.iloc[test_split_idx:]

    y_train = y.iloc[:valid_split_idx]
    y_valid = y.iloc[valid_split_idx:test_split_idx]
    y_test  = y.iloc[test_split_idx:]

    test_dates = df_dates.iloc[test_split_idx:].copy()
    evaluate_class_balance(y_train)

    results_df = compare_imbalance_methods(X_train_raw, y_train, methods=["SMOTE", "undersample", None])
    best_method = results_df.iloc[0]["Method"]
    best_method = None if best_method == "None" else best_method

    X_train_scaled, scaler = rolling_scale(X_train_raw)
    X_valid_scaled = rolling_scale(X_valid_raw, scaler)
    X_test_scaled  = rolling_scale(X_test_raw, scaler)
    X_train_res, y_train_res = apply_resampling(X_train_scaled, y_train, method=best_method)

    final_model, top_features, y_pred_proba, test_dates, best_params = evaluate_model(
        X_train_res, y_train_res,
        X_valid_scaled, y_valid,
        X_test_scaled, y_test,
        test_dates, df_raw=df
    )

    close_test = df.loc[X_test_scaled.index, "Close"]
    y_test_clean = y_test.reset_index(drop=True)


    analyze_overfitting_cv(X_train_res, y_train_res, best_params)

    last_proba = y_pred_proba[-1]
    last_date = test_dates.iloc[-1]
    print(f"Predicted P(>2% return in 20 days) on {last_date.date()}: {last_proba:.4f}")

    dummy_result = evaluate_dummy_classifier(X_train_scaled, y_train, X_test_scaled, y_test)
    logistic_result = evaluate_logistic_baseline(X_train_scaled, y_train, X_test_scaled, y_test)
    compare_models_table(dummy_result, logistic_result)


if __name__ == "__main__":
    main()
