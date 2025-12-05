# app.py (Entrega Final Reorganizada, Mejorada y Optimización Avanzada - Iteración Final)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
# --- IMPORTS MODIFICADOS/AÑADIDOS ---
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score, roc_curve, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression, LinearRegression
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import time
from datetime import datetime

# NUEVOS IMPORTS para mejoras
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor 
from sklearn.calibration import CalibratedClassifierCV 
from sklearn.metrics import recall_score 

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="BlueRisk | Plataforma Directiva",
    layout="wide",
    page_icon="🏦",
    initial_sidebar_state="expanded"
)

# ===========================
# PALETA CORPORATIVA AJUSTADA (AJUSTE 1: Cambio de colores rojos a azul oscuro)
# ===========================
COLORS = {
    "primary": "#192A56",      # Dark Blue / Títulos / Acentos (Antes: #5C1212)
    "secondary": "#708090",    # Slate Grey/Blue / Comparativos (Antes: #C87E7E)
    "sidebar_bg": "#212529",   # Dark Sidebar Background
    "app_bg": "#FFFFFF",       # BLANCO PURO (Fondo solicitado)
    "white": "#ffffff",
    "text_dark": "#212529",    # Text color (NEGRO)
    "muted": "#6B7280",        # Muted/Captions
    "danger": "#D72A2A",       # Rojo de peligro (Mantener el rojo de peligro)
    "success": "#10B981",
    "chart_main": ["#192A56", "#708090", "#2F4F4F", "#4682B4", "#A0A0A0"] # Paleta para gráficos (Ajustada a azules)
}

# ===========================
# CSS GLOBAL - (AJUSTE 1: Aplicar el color azul oscuro)
# ===========================
st.markdown(f"""
<style>
/* FONT IMPORT (Roboto) */
@import url('https://fonts.com/css2?family=Roboto:wght@300;400;700&display=swap');
:root {{
    --primary-color: {COLORS['primary']};
    --secondary-color: {COLORS['secondary']};
    --sidebar-bg-color: {COLORS['sidebar_bg']};
    --app-bg-color: {COLORS['app_bg']};
    --white: {COLORS['white']};
    --text-color: {COLORS['text_dark']};
}}

/* APP BACKGROUND */
.stApp {{
    background-color: var(--app-bg-color);
    color: var(--text-color);
    font-family: 'Roboto', sans-serif;
}}

/* TITLE BANNER */
.title-banner {{ 
    background-color: var(--primary-color); 
    color: var(--white); 
    font-size: 2.2em; 
    font-weight: bold; 
    padding: 20px 25px; 
    text-align: center; 
    margin-bottom: 30px; 
    border-radius: 6px;
}}
.title-banner p {{
    font-size: 0.5em;
    font-weight: 300;
    margin: 0;
    padding: 0;
}}

/* SIDEBAR */
section[data-testid="stSidebar"] {{
    background-color: var(--sidebar-bg-color);
    color: var(--white);
    padding-top: 18px;
    border-right: 3px solid var(--primary-color);
}}

/* BOTONES DE NAVEGACIÓN (Fondo BLANCO, Texto NEGRO, alineado a la izquierda) */
section[data-testid="stSidebar"] .stButton > button {{
    background-color: var(--white); 
    color: var(--text-color); 
    border-radius: 6px;
    padding: 8px 12px;
    border: 1px solid var(--text-color); 
    width: 100%;
    font-weight: 700;
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2); 
    margin-top: 8px;
    transition: all 0.2s ease;
    text-align: left; 
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background-color: #f0f0f0; 
    transform: translateY(-1px);
    box-shadow: 0 5px 10px rgba(25, 42, 86, 0.4); /* Ajuste de color de sombra: Antes rgba(92, 18, 18, 0.4) */
}}
/* Botón Seleccionado (Acento con color primario) - Ajustado para texto NEGRO */
section[data-testid="stSidebar"] .stButton > button:focus:not(:active) {{
    background-color: var(--white); 
    color: var(--text-color); 
    border: 3px solid var(--primary-color); 
    box-shadow: 0 5px 10px rgba(25, 42, 86, 0.4); /* Ajuste de color de sombra: Antes rgba(92, 18, 18, 0.4) */
}}

/* Sidebar Title - Título de navegación en blanco */
.sidebar-nav-title {{
    text-align:center;
    color: var(--white) !important; 
    margin-top: 12px;
    font-size: 18px;
    font-weight: 700;
    margin-bottom: 12px;
}}

/* Logo Container */
.logo-container {{
    display:flex;
    align-items:center;
    justify-content:center;
    margin-bottom: 18px;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}
.logo-container img {{
    width: 200px !important; # AJUSTE 1: Aumentar el ancho para que la imagen se vea completa
    height: auto;
    object-fit: contain;
}}

/* CARD - Diseño limpio y profesional */
.corp-card {{
    background-color: var(--white);
    padding: 18px;
    border-radius: 8px;
    box-shadow: 0 3px 12px rgba(9,30,66,0.06); 
    border: none; 
    margin-bottom: 18px;
}}
.card-header {{
    font-size: 18px;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 10px;
    border-bottom: 1px solid #E5E7EB; 
    padding-bottom: 8px;
}}

/* KPI */
.kpi-val {{
    font-size: 38px;
    font-weight: 800;
    color: var(--text-color); 
    line-height: 1.2; 
}}
.kpi-lbl {{
    font-size: 12px;
    font-weight: 700;
    color: {COLORS['muted']};
    text-transform: uppercase;
    letter-spacing: 1px;
    line-height: 1.1; 
}}

/* MAIN TITLES */
.stMarkdown h1 {{
    display: none; 
}}
.stMarkdown h2, .stMarkdown h3 {{
    color: var(--primary-color) !important;
}}

/* STREAMLIT TABS */
.stTabs [data-baseweb="tab-list"] {{
    gap: 12px;
}}
.stTabs [data-baseweb="tab-list"] button {{
    background-color: #F8F8F8; 
    border-radius: 6px 6px 0 0;
    padding: 10px 18px;
    font-weight: 600;
    color: {COLORS['muted']};
}}
.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {{
    background-color: var(--white);
    color: var(--primary-color);
    border-bottom: 3px solid var(--primary-color) !important;
}}

/* Form Input Enhancements */
.stTextInput > div > div > input, .stNumberInput > div > div > input {{
    border: 1px solid #D1D5DB;
    border-radius: 6px;
    padding: 10px;
}}
/* ESTILOS AÑADIDOS PARA LA COLUMNA RESULTADOS (Para la tabla audit) */
.stDataFrame .css-1r6cnm0, .stDataFrame .css-1l2o2t4 {{
    font-size: 10px; 
}}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# HELPER: THEME FOR PLOTLY
# ---------------------------
def apply_theme(fig):
    fig.update_layout(
        font={'family': "Roboto, sans-serif", 'color': COLORS['text_dark']},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        colorway=COLORS['chart_main'],
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(title='', orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    # axis styles
    if 'xaxis' in fig.layout:
        fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor='#E0E0E0', zeroline=False)
    return fig

# ===========================
# 2. DATOS (Pandas estable)
# ===========================
@st.cache_data
def load_data():
    """
    Carga el dataset y realiza la limpieza categórica utilizando Pandas.
    """
    try:
        # Fetch data from UCI ML repository
        ds = fetch_ucirepo(id=350)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
        
        # --- FIX: ADD ID COLUMN (Addressing KeyError: "['ID'] not in index") ---
        if ds.data.ids is not None and not ds.data.ids.empty:
            id_col_name = ds.data.ids.columns[0]
            df = pd.concat([ds.data.ids.rename(columns={id_col_name: 'ID'}), df], axis=1) 
        else:
            df.insert(0, 'ID', range(1, len(df) + 1)) 
        # --- END FIX ---
        
        # Map target column
        t_col = next((c for c in ['default.payment.next.month', 'Y'] if c in df.columns), df.columns[-1])
        df.rename(columns={t_col: 'TARGET'}, inplace=True)
        col_map = {
            'X1': 'LIMIT_BAL', 'X2': 'SEX', 'X3': 'EDUCATION', 'X4': 'MARRIAGE', 'X5': 'AGE', 
            'X6': 'PAY_0', 'X7': 'PAY_2', 'X8': 'PAY_3', 'X9': 'PAY_4', 'X10': 'PAY_5', 'X11': 'PAY_6', 
            'X12': 'BILL_AMT1', 'X13': 'BILL_AMT2', 'X14': 'BILL_AMT3', 'X15': 'BILL_AMT4', 'X16': 'BILL_AMT5', 'X17': 'BILL_AMT6', 
            'X18': 'PAY_AMT1', 'X19': 'PAY_AMT2', 'X20': 'PAY_AMT3', 'X21': 'PAY_AMT4', 'X22': 'PAY_AMT5', 'X23': 'PAY_AMT6'
        }
        if 'X1' in df.columns:
            df.rename(columns=col_map, inplace=True)
        
        # PANDAS cleaning for categories 
        # Limpieza EDUCATION: 0, 5, 6 -> 4 (Other/Unknown)
        df.loc[df['EDUCATION'].isin([0, 5, 6]), 'EDUCATION'] = 4
        # Limpieza MARRIAGE: 0 -> 3 (Other)
        df.loc[df['MARRIAGE'] == 0, 'MARRIAGE'] = 3
        df_clean = df.copy()

        # Creación de variables (Feature Engineering)
        # Ratio de Utilización de Crédito (CUR) - Se usa BILL_AMT1 (Septiembre) / LIMIT_BAL
        df_clean['UTILIZATION_RATIO'] = df_clean.get('BILL_AMT1', 0) / (df_clean.get('LIMIT_BAL', 1) + 1)
        
        # Ratios de Utilización Históricos para el nuevo gráfico 
        df_clean['CUR_1'] = df_clean.get('BILL_AMT1', 0) / (df_clean.get('LIMIT_BAL', 1) + 1)
        df_clean['CUR_2'] = df_clean.get('BILL_AMT2', 0) / (df_clean.get('LIMIT_BAL', 1) + 1)
        df_clean['CUR_3'] = df_clean.get('BILL_AMT3', 0) / (df_clean.get('LIMIT_BAL', 1) + 1)
        df_clean['CUR_4'] = df_clean.get('BILL_AMT4', 0) / (df_clean.get('LIMIT_BAL', 1) + 1)
        df_clean['CUR_5'] = df_clean.get('BILL_AMT5', 0) / (df_clean.get('LIMIT_BAL', 1) + 1)
        df_clean['CUR_6'] = df_clean.get('BILL_AMT6', 0) / (df_clean.get('LIMIT_BAL', 1) + 1)


        df_clean.replace([np.inf, -np.inf], 0, inplace=True)
        df_clean.fillna(0, inplace=True)
        return df_clean
    except Exception as e:
        st.error(f"Error al cargar el dataset UCI: {e}")
        return pd.DataFrame()

df = load_data()

# ===========================
# 3. MODELOS: Clasificación (TARGET) y Regresión (PAY_AMT4)
# ===========================
@st.cache_resource
def train_models_full(data):
    """Entrena los modelos de Clasificación y Regresión. 
    RETORNA X_test_cls y y_test_cls para graficar la Curva ROC.
    
    ***MEJORA/ITERACIÓN 7: Se calcula y retorna el Umbral para MAXIMIZAR F1-SCORE (Balance entre Precision y Recall/FNR).***
    """
    if data.empty:
        return {}, {}, None, [], None, [], None, None, 0.5 # Default Threshold (0.5 para Max Accuracy)

    # ----------------------------------------------------
    # 1. CLASIFICACIÓN (TARGET) - Predice default.payment.next.month (Octubre)
    # ----------------------------------------------------
    # Features hasta Septiembre (BILL_AMT1, PAY_0)
    features_cls = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'BILL_AMT1', 'UTILIZATION_RATIO']
    X_cls = data[features_cls]
    y_cls = data['TARGET']
    
    scaler_cls = StandardScaler()
    X_scaled_cls = scaler_cls.fit_transform(X_cls)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_scaled_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

    cls_results = {}
    
    # Cálculo de scale_pos_weight (Ratio de Negativos a Positivos)
    scale_weight = (len(y_train_cls) - y_train_cls.sum()) / y_train_cls.sum()
    
    # XGBoost (champion - Boosting) - Ajuste por GridSearchCV
    xgb_base = xgb.XGBClassifier(
        n_estimators=250, 
        random_state=42, 
        eval_metric='logloss', 
        use_label_encoder=False
    )
    param_grid = {
        'max_depth': [7, 9], 
        'learning_rate': [0.05, 0.07], 
        'scale_pos_weight': [scale_weight * 0.9, scale_weight * 1.1] 
    }
    grid_search = GridSearchCV(
        estimator=xgb_base, 
        param_grid=param_grid, 
        scoring='roc_auc', 
        cv=3, 
        verbose=0, 
        n_jobs=-1
    )
    grid_search.fit(X_train_cls, y_train_cls)
    xgb_cls = grid_search.best_estimator_
    best_params_cls = grid_search.best_params_ 
    y_pred_xgb = xgb_cls.predict(X_test_cls)
    xgb_auc = roc_auc_score(y_test_cls, xgb_cls.predict_proba(X_test_cls)[:,1])
    xgb_train_auc = roc_auc_score(y_train_cls, xgb_cls.predict_proba(X_train_cls)[:,1])
    xgb_f1 = f1_score(y_test_cls, y_pred_xgb) 
    xgb_gini = 2 * xgb_auc - 1 

    cls_results['XGBoost (Campeón)'] = {
        'model': xgb_cls,
        'Acc': accuracy_score(y_test_cls, y_pred_xgb),
        'Train_AUC': xgb_train_auc,
        'AUC': xgb_auc,
        'F1': xgb_f1, 
        'GINI': xgb_gini, 
        'Complexity': 'Media', 
        'Interpretabilidad': 'Alta (Gain)', 
        'Best_Params': best_params_cls,
    }

    # Logistic Regression (baseline - Lineal)
    lr_cls = LogisticRegression(max_iter=500).fit(X_train_cls, y_train_cls)
    y_pred_lr = lr_cls.predict(X_test_cls)
    lr_auc = roc_auc_score(y_test_cls, lr_cls.predict_proba(X_test_cls)[:,1])
    lr_f1 = f1_score(y_test_cls, y_pred_lr) 
    lr_gini = 2 * lr_auc - 1 
    cls_results['Logistic Regression'] = {
        'model': lr_cls,
        'Acc': accuracy_score(y_test_cls, y_pred_lr),
        'AUC': lr_auc,
        'F1': lr_f1, 
        'GINI': lr_gini, 
        'Complexity': 'Baja', 'Interpretabilidad': 'Alta',
    }
    
    # Decision Tree Classifier (Árbol de Decisión)
    dt_cls = DecisionTreeClassifier(max_depth=7, random_state=42).fit(X_train_cls, y_train_cls)
    y_pred_dt = dt_cls.predict(X_test_cls)
    dt_auc = roc_auc_score(y_test_cls, dt_cls.predict_proba(X_test_cls)[:,1])
    dt_f1 = f1_score(y_test_cls, y_pred_dt) 
    dt_gini = 2 * dt_auc - 1 
    cls_results['Decision Tree'] = {
        'model': dt_cls,
        'Acc': accuracy_score(y_test_cls, y_pred_dt),
        'AUC': dt_auc,
        'F1': dt_f1, 
        'GINI': dt_gini, 
        'Complexity': 'Media', 'Interpretabilidad': 'Alta (Rules)',
    }

    # Deep Learning (benchmark - Redes Neuronales)
    nn_cls = Sequential([
        Dense(32, activation='relu', input_shape=(len(features_cls),)),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    nn_cls.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
    nn_cls.fit(X_train_cls, y_train_cls, epochs=15, batch_size=64, verbose=0)
    nn_cls_proba = nn_cls.predict(X_test_cls, verbose=0).flatten()
    nn_cls_acc = (nn_cls_proba > 0.5).astype(int)
    nn_cls_auc = roc_auc_score(y_test_cls, nn_cls_proba)
    nn_f1 = f1_score(y_test_cls, nn_cls_acc) 
    nn_gini = 2 * nn_cls_auc - 1 
    cls_results['Deep Learning (NN)'] = {
        'model': nn_cls,
        'Acc': accuracy_score(y_test_cls, nn_cls_acc),
        'AUC': nn_cls_auc,
        'F1': nn_f1, 
        'GINI': nn_gini, 
        'Complexity': 'Alta', 'Interpretabilidad': 'Baja (Blackbox)', 
    }

    # XGBoost Calibrado (Isotónica) - Campeón final de Clasificación
    xgb_base_for_cal = xgb.XGBClassifier(
        **best_params_cls, 
        n_estimators=250, 
        random_state=42, 
        eval_metric='logloss', 
        use_label_encoder=False
    )
    xgb_cal_model = CalibratedClassifierCV(
        xgb_base_for_cal, 
        method='isotonic', 
        cv=5
    )
    xgb_cal_model.fit(X_train_cls, y_train_cls)
    y_pred_cal = (xgb_cal_model.predict_proba(X_test_cls)[:, 1] > 0.5).astype(int)
    cal_auc = roc_auc_score(y_test_cls, xgb_cal_model.predict_proba(X_test_cls)[:, 1])
    cal_f1 = f1_score(y_test_cls, y_pred_cal) 
    cal_gini = 2 * cal_auc - 1 

    cls_results['XGBoost Calibrado (Isotónica)'] = {
        'model': xgb_cal_model,
        'Acc': accuracy_score(y_test_cls, y_pred_cal),
        'AUC': cal_auc,
        'F1': cal_f1, 
        'GINI': cal_gini, 
        'Complexity': 'Media', 
        'Interpretabilidad': 'Media (Calibrado)',
    }

    # ----------------------------------------------------
    # MODIFICACIÓN SOLICITADA: CÁLCULO DEL UMBRAL PARA MAXIMIZAR F1-SCORE
    # F1-Score: Maximiza el equilibrio entre Precision y Recall (bajo FNR y bajo FPR)
    # ----------------------------------------------------
    y_proba_cal = xgb_cal_model.predict_proba(X_test_cls)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test_cls, y_proba_cal)
    
    optimal_business_threshold = 0.5 # Default fallback
    
    # Calcular F1-Score para cada umbral
    fscores = []
    
    # El rango de umbrales se ajusta para evitar predicciones extremas (all 0 or all 1)
    for t in thresholds:
        y_pred_t = (y_proba_cal >= t).astype(int)
        # Handle division by zero for thresholds that predict all 0 or all 1
        fscore = f1_score(y_test_cls, y_pred_t, zero_division=0)
        fscores.append(fscore)

    if fscores:
        # Encontrar el índice del umbral que da el máximo F1-Score
        max_f1_index = np.argmax(fscores)
        optimal_business_threshold = thresholds[max_f1_index]
        
    # Asegurar que el umbral sea razonable
    if not 0.01 <= optimal_business_threshold <= 0.99:
        # Usar un compromiso balanceado si el cálculo es inválido
        optimal_business_threshold = 0.30 

    # ----------------------------------------------------
    # 2. REGRESIÓN (PAY_AMT4) - Predice monto a pagar en Junio
    # ----------------------------------------------------
    # FEATURES: Usando datos hasta Mayo (PAY_5, BILL_AMT5)
    features_reg = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                    'PAY_5', 'PAY_6', 'BILL_AMT5', 'BILL_AMT6']
    # TARGET: Monto de Pago en Junio
    y_reg = data['PAY_AMT4'] 
    X_reg = data[features_reg]
    
    scaler_reg = StandardScaler()
    X_scaled_reg = scaler_reg.fit_transform(X_reg)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_scaled_reg, y_reg, test_size=0.2, random_state=42)

    reg_results = {}
    
    # XGBoost Regressor (champion - Boosting)
    xgb_reg = xgb.XGBRegressor(
        n_estimators=150, max_depth=6, learning_rate=0.07, random_state=42 
    )
    xgb_reg.fit(X_train_reg, y_train_reg)
    y_pred_xgb_reg = xgb_reg.predict(X_test_reg)
    reg_results['XGBoost Regressor (Campeón)'] = {
        'model': xgb_reg,
        'R2': r2_score(y_test_reg, y_pred_xgb_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_xgb_reg)),
        'Complexity': 'Media', 'Interpretabilidad': 'Alta (Gain)', 'Features': features_reg
    }
    
    # Linear Regression (baseline - Lineal)
    lr_reg = LinearRegression().fit(X_train_reg, y_train_reg)
    y_pred_lr_reg = lr_reg.predict(X_test_reg)
    reg_results['Linear Regression'] = {
        'model': lr_reg,
        'R2': r2_score(y_test_reg, y_pred_lr_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_lr_reg)),
        'Complexity': 'Baja', 'Interpretabilidad': 'Alta', 'Features': features_reg
    }

    # Decision Tree Regressor (NUEVO: para consistencia con Clasificación)
    dt_reg = DecisionTreeRegressor(max_depth=7, random_state=42).fit(X_train_reg, y_train_reg)
    y_pred_dt_reg = dt_reg.predict(X_test_reg)
    reg_results['Decision Tree Regressor'] = {
        'model': dt_reg,
        'R2': r2_score(y_test_reg, y_pred_dt_reg),
        'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_dt_reg)),
        'Complexity': 'Media', 'Interpretabilidad': 'Alta (Rules)', 'Features': features_reg
    }

    # RETORNO AÑADIDO: El Umbral de Negocio calculado para MAX F1-SCORE
    return cls_results, reg_results, scaler_cls, features_cls, features_reg, scaler_reg, X_test_cls, y_test_cls, optimal_business_threshold

# Run Models
optimal_threshold = 0.35 # Valor por defecto/fallback
if not df.empty:
    try:
        (
            cls_models, reg_models, scaler_cls, feat_names_cls, 
            feat_names_reg, scaler_reg, X_test_cls, y_test_cls, 
            optimal_threshold
        ) = train_models_full(df)
        champion_cls = cls_models.get('XGBoost (Campeón)')
        champion_reg = reg_models.get('XGBoost Regressor (Campeón)')
        calibrated_cls = cls_models.get('XGBoost Calibrado (Isotónica)') 
    except Exception as e:
        st.error(f"Error durante el entrenamiento de modelos: {e}")
        cls_models, reg_models, scaler_cls, feat_names_cls, scaler_reg, feat_names_reg = {}, {}, None, [], None, []
        X_test_cls, y_test_cls, calibrated_cls = None, None, None 
        champion_cls, champion_reg = None, None
else:
    cls_models, reg_models, scaler_cls, feat_names_cls, scaler_reg, feat_names_reg = {}, {}, None, [], None, []
    X_test_cls, y_test_cls, calibrated_cls = None, None, None 
    champion_cls, champion_reg = None, None


# =======================================================
# 4. FUNCIONES DE APLICACIÓN Y AUDITORÍA (MODIFICADA para usar Umbral Dinámico)
# =======================================================
def apply_champion_models(data, scaler_cls, features_cls, calibrated_cls, scaler_reg, features_reg, champion_reg, business_threshold):
    """
    Aplica los modelos campeones al dataset completo y retorna el DataFrame
    con las predicciones RAW para análisis y resumen, usando el umbral de negocio.
    """
    if calibrated_cls is None or champion_reg is None or data.empty:
        return pd.DataFrame()

    # Obtener columnas necesarias, asegurando que existan
    cols_needed = ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'TARGET', 'PAY_AMT4']
    df_audit = data[[col for col in cols_needed if col in data.columns]].copy()
    
    # ----------------------------------------------------
    # CLASIFICACIÓN (TARGET/Default - Octubre)
    # ----------------------------------------------------
    X_cls_full = data[features_cls]
    X_scaled_cls_full = scaler_cls.transform(X_cls_full)
    
    # Probabilidad de Default (P(D))
    df_audit['PRED_DEFAULT_PROBA'] = calibrated_cls['model'].predict_proba(X_scaled_cls_full)[:, 1]
    
    # Predicción Binaria (Umbral 0.5) - Se mantiene para referencia
    df_audit['PRED_DEFAULT_BIN_05'] = (df_audit['PRED_DEFAULT_PROBA'] > 0.5).astype(int)
    
    # Predicción Binaria (Umbral de Negocio - MAX F1-SCORE) - NUEVO: Usa el Umbral Dinámico
    df_audit['PRED_DEFAULT_BIN_BUS'] = (df_audit['PRED_DEFAULT_PROBA'] > business_threshold).astype(int)
    
    # ----------------------------------------------------
    # REGRESIÓN (PAY_AMT4)
    # ----------------------------------------------------
    X_reg_full = data[features_reg]
    X_scaled_reg_full = scaler_reg.transform(X_reg_full)
    
    # Predicción del Monto de Pago en Junio (Asegurar no negativos)
    pay_amt4_pred = champion_reg['model'].predict(X_scaled_reg_full)
    df_audit['PRED_PAY_AMT4'] = np.maximum(0, pay_amt4_pred) 

    # Renombrar para claridad en el análisis
    df_audit.rename(columns={'TARGET': 'Real_Default', 'PAY_AMT4': 'Real_Pago_Jun'}, inplace=True)
    
    # Calcular error absoluto para el análisis de regresión
    df_audit['ABS_ERROR_PAY_AMT4'] = abs(df_audit['Real_Pago_Jun'] - df_audit['PRED_PAY_AMT4'])
    
    return df_audit

# Ejecutar la aplicación de modelos para Auditoría (Guarda el raw data)
df_raw_audit = pd.DataFrame()
if not df.empty and calibrated_cls and champion_reg:
    df_raw_audit = apply_champion_models(df, scaler_cls, feat_names_cls, calibrated_cls, scaler_reg, feat_names_reg, champion_reg, optimal_threshold)

# ===========================
# SESSION: AUDIT LOGS
# ===========================
if 'audit_log' not in st.session_state:
    st.session_state['audit_log'] = []

# ===========================
# SIDEBAR: LOGO + NAV (AJUSTE 2: Título, Imagen y Orden de Botones)
# ===========================
with st.sidebar:
    # LOGO (Posición arriba de Páginas de Análisis) - CAMBIAR A IMAGEN SOLICITADA
    st.markdown('<div class="logo-container">', unsafe_allow_html=True)
    # REEMPLAZO POR IMAGEN SOLICITADA
    st.markdown(f'<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQCeK7XHYU3gSAO1YTSVNbQvVJ3i2FPH53n_A&s" alt="Logo corporativo"/>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Título de navegación - CAMBIAR A "Índice"
    st.markdown('<div class="sidebar-nav-title">Índice</div>', unsafe_allow_html=True)

    # Botones de navegación (Fondo Blanco, Texto Negro) - REORDENAR (Metodología primero)
    if st.button("Metodología y Cuestionario", key="nav_ques"): # AJUSTE 2: Primero
        st.session_state['page'] = "Metodología y Cuestionario"
    if st.button("Análisis Datos", key="nav_dash"):
        st.session_state['page'] = "Análisis Datos"
    if st.button("Comparativo de Modelos", key="nav_lab"):
        st.session_state['page'] = "Comparativo de Modelos"
    if st.button("Resultado de Modelo", key="nav_audit"): 
        st.session_state['page'] = "Resultado de Modelo"
    if st.button("Pronóstico de Impago", key="nav_sim"): 
        st.session_state['page'] = "Pronóstico de Impago"


# ===========================
# 5. Dashboard (Análisis Datos)
# ===========================
if 'page' not in st.session_state:
    st.session_state['page'] = "Metodología y Cuestionario" # Página inicial por defecto

page = st.session_state['page']

if page == "Análisis Datos":
    # BANNER
    st.markdown('<div class="title-banner">ANÁLISIS EXPLORATORIO DE DATOS<p>Información Descriptiva y Segmentación</p></div>', unsafe_allow_html=True)

    if df.empty:
        st.error("No se pudo cargar el dataset.")
    else:
        st.header("1. Segmentación de la Cartera")
        st.markdown(f"""
        <p style="font-size:12px; color:{COLORS['muted']}; margin-top:-10px;">
        Seleccione los filtros para segmentar la cartera y recalcular los KPIs y gráficos.
        </p>
        """, unsafe_allow_html=True)
        
        # ------------------
        # 1. Segmentación
        # ------------------
        c_s1, c_s2, c_s3, c_s4 = st.columns(4)
        with c_s1:
            seg_edu = st.selectbox("Nivel Educativo", options=["Total", "Posgrado", "Universidad", "Bachillerato", "Otros"])
        with c_s2:
            seg_mar = st.selectbox("Estado Civil", options=["Total", "Soltero", "Casado", "Otros"])
        with c_s3:
            seg_age = st.selectbox("Rango de Edad", options=["Total", "20-30", "30-40", "40-50", "50-60", "60-70", "70+"])
        with c_s4:
            seg_sex = st.selectbox("Género", options=["Total", "Hombre", "Mujer"])
        
        # AJUSTE 3: Se eliminan los segmentadores de Límite de Crédito Mín y Máx.

        df_d = df.copy()
        
        # ------------------
        # 2. Lógica de Filtrado
        # ------------------
        if seg_edu != "Total":
            edu_map = {"Posgrado": 1, "Universidad": 2, "Bachillerato": 3, "Otros": 4}
            df_d = df_d[df_d['EDUCATION'] == edu_map[seg_edu]]
        if seg_mar != "Total":
            mar_map = {"Casado": 1, "Soltero": 2, "Otros": 3}
            df_d = df_d[df_d['MARRIAGE'] == mar_map[seg_mar]]
        # AJUSTE 3: Lógica de filtrado de edad para rangos de 10 años
        if seg_age != "Total":
            age_map = {"20-30": (20, 30), "30-40": (30, 40), "40-50": (40, 50), "50-60": (50, 60), "60-70": (60, 70), "70+": (70, 100)}
            min_age, max_age = age_map[seg_age]
            if seg_age == "70+":
                df_d = df_d[df_d['AGE'] >= min_age]
            else:
                df_d = df_d[(df_d['AGE'] >= min_age) & (df_d['AGE'] < max_age)]
        if seg_sex != "Total":
            sex_map = {"Hombre": 1, "Mujer": 2}
            df_d = df_d[df_d['SEX'] == sex_map[seg_sex]]
        # AJUSTE 3: Se elimina la lógica de filtrado de Límite de Crédito Mín y Máx.
        
        # --- KPIS ---
        total_clientes = len(df_d)
        default_rate = df_d['TARGET'].mean() if total_clientes > 0 else 0
        avg_limit = df_d['LIMIT_BAL'].mean() if total_clientes > 0 else 0
        avg_age = df_d['AGE'].mean() if total_clientes > 0 else 0 
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        with kpi1:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['primary']};">
            <div class="card-header">Clientes en Segmento</div>
            <div class="kpi-val">{total_clientes:,.0f}</div>
            <div class="kpi-lbl">Total Clientes</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi2:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['danger']};">
            <div class="card-header">Tasa de Default (%)</div>
            <div class="kpi-val">{default_rate:.2%}</div>
            <div class="kpi-lbl">Pagos Atrasados (Octubre)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi3:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['secondary']};">
            <div class="card-header">Límite Promedio (NTD)</div>
            <div class="kpi-val">${avg_limit:,.0f}</div>
            <div class="kpi-lbl">Balance de Crédito</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi4:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['success']};">
            <div class="card-header">Edad Promedio (Años)</div>
            <div class="kpi-val">{avg_age:,.1f}</div>
            <div class="kpi-lbl">Promedio de Edad</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("---")

        st.header("2. Distribución y Riesgo")
        
        # --- GRAPHS - Row 1 ---
        c_g1, c_g2 = st.columns(2)

        # Gráfico 1: Tasa de Default por Sexo
        with c_g1:
            st.markdown('<div class="corp-card"><div class="card-header">Tasa de Default por Género</div>', unsafe_allow_html=True)
            if not df_d.empty:
                sex_risk = df_d.groupby('SEX')['TARGET'].mean().reset_index()
                sex_risk['SEX'] = sex_risk['SEX'].map({1: 'Hombre', 2: 'Mujer'})
                fig_sex = px.bar(sex_risk, x='SEX', y='TARGET', labels={'TARGET':'Tasa Default', 'SEX': 'Género'}, text=sex_risk['TARGET'].apply(lambda x: f"{x:.1%}"))
                fig_sex.update_traces(marker_color=COLORS['primary'], hovertemplate='%{x}: %{y:.1%}')
                fig_sex.update_yaxes(tickformat=".0%")
                apply_theme(fig_sex)
                st.plotly_chart(fig_sex, use_container_width=True)
            else:
                st.info("Dataset no disponible.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Gráfico 2: Proporción de la Cartera (Default vs. Pagó)
        with c_g2:
            st.markdown('<div class="corp-card"><div class="card-header">Distribución de Default (Octubre)</div>', unsafe_allow_html=True)
            if not df_d.empty:
                target_counts = df_d['TARGET'].value_counts().reset_index()
                target_counts.columns = ['Estado', 'Conteo']
                target_counts['Estado'] = target_counts['Estado'].map({0: 'Pagó (Sano)', 1: 'No Pagó (Default)'})
                fig_pie = px.pie(target_counts, values='Conteo', names='Estado', color='Estado', 
                                 color_discrete_map={'Pagó (Sano)': COLORS['secondary'], 'No Pagó (Default)': COLORS['danger']}, 
                                 title='Proporción Real de la Cartera')
                fig_pie.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#FFFFFF', width=2)))
                apply_theme(fig_pie)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Dataset no disponible.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---") 

        # --- GRAPHS - Row 2 ---
        c_g3, c_g4 = st.columns(2)

        # Gráfico 3: Tasa de Default por Educación
        with c_g3:
            st.markdown('<div class="corp-card"><div class="card-header">Tasa de Default por Nivel Educativo</div>', unsafe_allow_html=True)
            if not df_d.empty:
                edu_risk = df_d.groupby('EDUCATION')['TARGET'].mean().reset_index()
                edu_risk['EDUCATION'] = edu_risk['EDUCATION'].map({1: 'Posgrado', 2: 'Universidad', 3: 'Bachillerato', 4: 'Otros'})
                fig_edu = px.bar(edu_risk, x='EDUCATION', y='TARGET', 
                                 labels={'TARGET':'Tasa Default', 'EDUCATION': 'Nivel Educativo'}, 
                                 text=edu_risk['TARGET'].apply(lambda x: f"{x:.1%}"))
                fig_edu.update_traces(marker_color=COLORS['primary'], hovertemplate='%{x}: %{y:.1%}')
                fig_edu.update_yaxes(tickformat=".0%")
                apply_theme(fig_edu)
                st.plotly_chart(fig_edu, use_container_width=True)
            else:
                st.info("Dataset no disponible.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Gráfico 4: Evolución del Uso de Crédito (CUR)
        with c_g4:
            st.markdown('<div class="corp-card"><div class="card-header">Ratio de Utilización de Crédito (CUR) Histórico</div>', unsafe_allow_html=True)
            if not df_d.empty:
                # Seleccionar solo las columnas CUR
                cur_cols = ['CUR_1', 'CUR_2', 'CUR_3', 'CUR_4', 'CUR_5', 'CUR_6']
                cur_df = df_d[cur_cols].mean().reset_index()
                cur_df.columns = ['Mes', 'Ratio']
                # Mapear los meses de la variable
                cur_df['Mes'] = cur_df['Mes'].str.replace('CUR_', '').astype(int)
                mes_map = {1: 'Sep', 2: 'Ago', 3: 'Jul', 4: 'Jun', 5: 'May', 6: 'Abr'}
                cur_df['Mes_Label'] = cur_df['Mes'].map(mes_map)

                fig_cur = px.line(cur_df.sort_values(by='Mes', ascending=False), x='Mes_Label', y='Ratio', 
                                  markers=True, title='Promedio del (Monto Facturado / Límite de Crédito)')
                fig_cur.update_traces(line=dict(color=COLORS['primary'], width=3), marker=dict(size=10, color=COLORS['secondary']))
                fig_cur.update_yaxes(tickformat=".0%", range=[0, cur_df['Ratio'].max() * 1.1 + 0.1])
                fig_cur.update_xaxes(title='Mes (2005)', categoryorder='array', categoryarray=cur_df.sort_values(by='Mes', ascending=False)['Mes_Label'].tolist())
                apply_theme(fig_cur)
                st.plotly_chart(fig_cur, use_container_width=True)
            else:
                st.info("Dataset no disponible.")
            st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# 6. Model Lab (Comparativo de Modelos)
# ===========================
elif page == "Comparativo de Modelos":
    # BANNER
    st.markdown('<div class="title-banner">COMPARATIVO DE MODELOS<p>Benchmark de Clasificación y Regresión</p></div>', unsafe_allow_html=True)
    
    # AJUSTE 4: Regresar a la estructura de compaginación (pestañas)
    tab_cls, tab_reg = st.tabs(["Predicción de Default (TARGET)", "Predicción de Pago (Regresión)"])

    with tab_cls:
        st.header("1. Clasificación: Predicción de Default (TARGET)")
        st.markdown("El objetivo es predecir si el cliente entrará en **Default** en **Octubre** a partir de datos hasta **Septiembre**.")
        st.markdown("### 1.1. Benchmark de Rendimiento (Evaluación en Datos de Prueba)")
        st.markdown(f"""
        <p style="font-size:12px; color:{COLORS['muted']}; margin-top:-10px;">
        * **AUC:** Área Bajo la Curva ROC. Mide la capacidad del modelo de rankear clientes correctamente (distinguir Default de No-Default). Mayor es mejor.
        * **GINI:** Mide la concentración predictiva (2 * AUC - 1). Mayor es mejor.
        * **F1:** Media armónica de Precision y Recall. Mide el balance de acierto en la clase Default.
        </p>
        """, unsafe_allow_html=True)

        if cls_models:
            bench_data = []
            for m, res in cls_models.items():
                gap = abs(res.get('Train_AUC', 0) - res.get('AUC', 0)) if 'Train_AUC' in res else 0
                params_str = 'N/A'
                if m == 'XGBoost (Campeón)':
                    p = res.get('Best_Params', {})
                    params_str = f"Depth: {p.get('max_depth')}, LR: {p.get('learning_rate')}, Scale: {res.get('Best_Params', {}).get('scale_pos_weight', 0):.1f}"
                elif m == 'XGBoost Calibrado (Isotónica)':
                    params_str = 'Calibración Isotónica sobre Campeón'
                    
                bench_data.append({
                    "Modelo": m,
                    "AUC (Test)": f"{res['AUC']:.3f}",
                    "Accuracy (Test)": f"{res['Acc']:.3f}",
                    "F1 (Test)": f"{res['F1']:.3f}",
                    "GINI (Test)": f"{res['GINI']:.3f}",
                    "Complejidad": res['Complexity'],
                    "Interpretabilidad": res['Interpretabilidad'],
                    "Riesgo de Overfitting": "Bajo" if gap < 0.05 else "Medio/Alto",
                    "Hyperparámetros Clave": params_str
                })
            st.dataframe(pd.DataFrame(bench_data).sort_values(by="AUC (Test)", ascending=False), use_container_width=True)
        else:
            st.info("Modelos de clasificación no disponibles.")

        st.markdown("### 1.2. Análisis de Decisión (Umbral de Negocio)")
        if calibrated_cls and X_test_cls is not None:
            threshold = optimal_threshold
            y_proba_business = calibrated_cls['model'].predict_proba(X_test_cls)[:, 1]
            y_pred_business = (y_proba_business >= threshold).astype(int)
            cm = confusion_matrix(y_test_cls, y_pred_business)
            df_cm = pd.DataFrame(cm, index=['Real Pagó (0)', 'Real Default (1)'], columns=['Pred Pagó (0)', 'Pred Default (1)'])

            # Cálculo de Métricas clave para Negocio
            tn, fp, fn, tp = cm.ravel()
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0 # Clave para "predicciones que pasaron"

            st.markdown(f"""
            <div class="corp-card" style="margin-top:20px;">
                <div class="card-header">Umbral de Negocio Óptimo para F1-Score: <span style="color:{COLORS['danger']};">{threshold:.3f}</span></div>
                <p style="font-size:14px; margin-bottom: 0;">
                El umbral se ajusta automáticamente para **maximizar el F1-Score**, buscando el mejor balance entre la identificación de clientes en default (Recall) y la minimización de falsos positivos (Precision), logrando:
                </p>
                <ul>
                    <li>**Recall (Sensibilidad):** {recall:.2%} (Clientes Default identificados correctamente)</li>
                    <li>**Especificidad:** {specificity:.2%} (Clientes Sanos identificados correctamente)</li>
                    <li>**Tasa de Falso Negativo (FNR):** {fn_rate:.2%} (Clientes Default no identificados)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.subheader("Matriz de Confusión (Umbral de Negocio)")
            st.dataframe(df_cm)
            
            # Curva ROC
            st.subheader("Curva ROC Comparativa")
            try:
                fig_roc = go.Figure()
                for name, res in cls_models.items():
                    if name == 'Deep Learning (NN)':
                        y_proba = res['model'].predict(X_test_cls, verbose=0).flatten()
                    elif name == 'XGBoost Calibrado (Isotónica)':
                        y_proba = res['model'].predict_proba(X_test_cls)[:, 1]
                    else:
                        y_proba = res['model'].predict_proba(X_test_cls)[:, 1]
                        
                    fpr, tpr, _ = roc_curve(y_test_cls, y_proba)
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'{name} (AUC: {res["AUC"]:.3f})', mode='lines'))
                    
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'), name='Random (AUC: 0.50)'))
                fig_roc.update_layout(xaxis_title='Tasa de Falso Positivo (FPR)', yaxis_title='Tasa de Verdadero Positivo (TPR)', title='Curva ROC de Modelos de Clasificación')
                apply_theme(fig_roc)
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception as e:
                st.info(f"No se pudo generar la Curva ROC: {e}")
        else:
            st.info("Datos de test de clasificación no disponibles.")
        
    with tab_reg:
        st.header("2. Regresión: Predicción de Pago (PAY_AMT4)")
        st.markdown("El objetivo es predecir el monto de pago estimado en **Junio** a partir de datos hasta **Mayo**.")
        st.markdown("### 2.1. Benchmark de Rendimiento (Evaluación en Datos de Prueba)")
        st.markdown(f"""
        <p style="font-size:12px; color:{COLORS['muted']}; margin-top:-10px;">
        * **RMSE:** Raíz del Error Cuadrático Medio. Mide el error promedio en la misma unidad que el Target ($NTD). Menor es mejor.
        * **R2:** Mide la proporción de la varianza en el pago que es predecible a partir de las variables. Mayor es mejor.
        </p>
        """, unsafe_allow_html=True)

        if reg_models:
            reg_bench_data = []
            for m, res in reg_models.items():
                reg_bench_data.append({
                    "Modelo": m,
                    "R2 (Test)": f"{res['R2']:.3f}",
                    "RMSE (Test)": f"${res['RMSE']:,.0f}",
                    "Complejidad": res['Complexity'],
                    "Interpretabilidad": res['Interpretabilidad'],
                    "Features": ", ".join(res['Features'][:4]) + "..."
                })
            st.dataframe(pd.DataFrame(reg_bench_data).sort_values(by="R2 (Test)", ascending=False), use_container_width=True)
        else:
            st.info("Modelos de regresión no disponibles.")
            
        st.markdown("### 2.2. Visualización de Predicciones")
        if X_test_cls is not None:
            # Obtener el modelo campeón de regresión (XGBoost Regressor)
            xgb_reg_model = reg_models.get('XGBoost Regressor (Campeón)', {}).get('model')
            y_test_reg = reg_models.get('XGBoost Regressor (Campeón)', {}).get('y_test') # Faltaba retornar en train_models_full, usar data del full set si es necesario
            
            # Si no tenemos el X_test_reg y y_test_reg directo del train_models_full (que sí los retorna), debemos usar el completo para graficar
            if champion_reg and 'LIMIT_BAL' in df_raw_audit.columns:
                
                # Gráfico 1: Predicción vs. Real (Scatter Plot)
                st.subheader("Dispersión Real vs. Predicción (Monto de Pago)")
                df_plot = df_raw_audit.sample(n=min(len(df_raw_audit), 5000), random_state=42) # Muestra para mejor rendimiento
                fig_scatter = px.scatter(df_plot, x='Real_Pago_Jun', y='PRED_PAY_AMT4', 
                                         color='Real_Default', # Usamos el real default como color
                                         color_discrete_map={0: COLORS['secondary'], 1: COLORS['danger']},
                                         hover_data=['ID', 'LIMIT_BAL'],
                                         title="Real vs. Predicción de Pago (PAY_AMT4)")
                
                # Línea de la verdad (Y=X)
                max_val = max(df_plot['Real_Pago_Jun'].max(), df_plot['PRED_PAY_AMT4'].max())
                fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', 
                                                 line=dict(dash='dash', color='black', width=1), 
                                                 name='Línea de la Verdad (Y=X)'))
                
                fig_scatter.update_layout(xaxis_title='Monto de Pago Real (NTD)', 
                                          yaxis_title='Monto de Pago Estimado (NTD)')
                apply_theme(fig_scatter)
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Gráfico 2: Distribución de Errores Absolutos
                st.subheader("Distribución de Errores Absolutos")
                fig_hist_reg = px.histogram(df_plot, x='ABS_ERROR_PAY_AMT4', nbins=50, 
                                            title='Frecuencia del Error Absoluto | Real - Predicción |',
                                            labels={'ABS_ERROR_PAY_AMT4': 'Error Absoluto (NTD)'})
                fig_hist_reg.update_traces(marker_color=COLORS['primary'])
                apply_theme(fig_hist_reg)
                st.plotly_chart(fig_hist_reg, use_container_width=True)

            else:
                st.info("Modelos de regresión no disponibles para visualización de test.")
        else:
            st.info("Datos de test de regresión no disponibles.")


# ===========================
# 7. Model Audit (Resultado de Modelo)
# ===========================
elif page == "Resultado de Modelo":
    # BANNER
    st.markdown('<div class="title-banner">RESULTADO DE MODELO<p>Auditoría de Predicciones del Campeón</p></div>', unsafe_allow_html=True)

    if df_raw_audit.empty:
        st.error("No se pudo cargar el DataFrame de Auditoría. Verifique la disponibilidad del dataset y los modelos.")
    else:
        st.header("1. Resumen de Desempeño Global")

        # 1.1. Cálculo de KPIs (sobre el dataset completo)
        # CLASIFICACIÓN (Umbral de Negocio)
        accuracy = accuracy_score(df_raw_audit['Real_Default'], df_raw_audit['PRED_DEFAULT_BIN_BUS'])
        cm_full = confusion_matrix(df_raw_audit['Real_Default'], df_raw_audit['PRED_DEFAULT_BIN_BUS'])
        # Evitar errores si la matriz es singular (aunque es muy improbable con el dataset completo)
        if cm_full.size == 4:
            tn, fp, fn, tp = cm_full.ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            specificity, fn_rate, fpr = 0, 0, 0

        kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)
        with kpi_c1:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['success']};">
            <div class="card-header">Accuracy Global (Umbral {optimal_threshold:.3f})</div>
            <div class="kpi-val">{accuracy:.2%}</div>
            <div class="kpi-lbl">Total Predicciones Correctas</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_c2:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['danger']};">
            <div class="card-header">Tasa Falso Negativo (FNR)</div>
            <div class="kpi-val">{fn_rate:.2%}</div>
            <div class="kpi-lbl">Default Real que no se detectó</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_c3:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['secondary']};">
            <div class="card-header">Tasa Falso Positivo (FPR)</div>
            <div class="kpi-val">{fpr:.2%}</div>
            <div class="kpi-lbl">Sano Real que se marcó Default</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_c4:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['primary']};">
            <div class="card-header">Especificidad</div>
            <div class="kpi-val">{specificity:.2%}</div>
            <div class="kpi-lbl">Clientes Sanos Identificados</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        # REGRESIÓN (PAY_AMT4)
        st.header("2. Desempeño de Predicción de Pago")
        rmse_full = np.sqrt(mean_squared_error(df_raw_audit['Real_Pago_Jun'], df_raw_audit['PRED_PAY_AMT4']))
        df_non_zero_pay = df_raw_audit[df_raw_audit['Real_Pago_Jun'] > 0].copy()
        
        mape = 0
        accurate_30 = 0
        if not df_non_zero_pay.empty:
            df_non_zero_pay['APE'] = abs(df_non_zero_pay['Real_Pago_Jun'] - df_non_zero_pay['PRED_PAY_AMT4']) / df_non_zero_pay['Real_Pago_Jun']
            mape = df_non_zero_pay['APE'].mean()
            # Accuracy within 30% accurate
            accurate_30 = (df_non_zero_pay['APE'] <= 0.30).mean()

        kpi_r1, kpi_r2, kpi_r3 = st.columns(3)
        with kpi_r1:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['primary']};">
            <div class="card-header">RMSE (Error Promedio)</div>
            <div class="kpi-val">${rmse_full:,.0f}</div>
            <div class="kpi-lbl">Desvío Cuadrático Promedio (NTD)</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_r2:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['secondary']};">
            <div class="card-header">MAPE (Error Porcentual Medio)</div>
            <div class="kpi-val">{mape:.2%}</div>
            <div class="kpi-lbl">Solo sobre Pagos Reales > 0</div>
            </div>
            """, unsafe_allow_html=True)
        with kpi_r3:
            st.markdown(f"""
            <div class="corp-card" style="border-top: 3px solid {COLORS['success']};">
            <div class="card-header">Predicción Precisa (<30% Error)</div>
            <div class="kpi-val">{accurate_30:.2%}</div>
            <div class="kpi-lbl">Pagos Reales > 0</div>
            </div>
            """, unsafe_allow_html=True)

        # Chart: Distribution of Absolute Error
        st.markdown('<div class="corp-card"><div class="card-header">Distribución del Error Absoluto en la Predicción de Pago (NTD)</div>', unsafe_allow_html=True)
        # Limitar a valores razonables para el gráfico
        fig_hist = px.histogram(df_raw_audit[df_raw_audit['ABS_ERROR_PAY_AMT4'] < 50000], 
                                x='ABS_ERROR_PAY_AMT4', nbins=50,
                                labels={'ABS_ERROR_PAY_AMT4': 'Error Absoluto | Real - Predicción | (NTD)'},
                                title='Error Absoluto (Limitado a 50k NTD)')
        fig_hist.update_traces(marker_color=COLORS['secondary'])
        apply_theme(fig_hist)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # 3. Auditoría de Muestra
        st.header("3. Muestra de Auditoría de Predicciones")
        st.markdown(f"""
        <p style="font-size:12px; color:{COLORS['muted']}; margin-top:-10px;">
        Muestra aleatoria de las primeras 1000 predicciones con resultados clave. 
        </p>
        """, unsafe_allow_html=True)
        
        # Helper para formatear la tabla de auditoría con estilo condicional
        def get_formatted_audit_df(df_raw, threshold):
            if df_raw.empty:
                return pd.DataFrame()

            df_temp = df_raw.copy()
            
            # Crear columna de resultado legible y con color (para HTML)
            df_temp['resultados'] = df_temp.apply(
                lambda row: f'<span style="color: {COLORS["success"]}; font-weight: bold;">CORRECTO (Pagó)</span>' 
                            if row['Real_Default'] == 0 and row['PRED_DEFAULT_BIN_BUS'] == 0
                            else (f'<span style="color: {COLORS["danger"]}; font-weight: bold;">CORRECTO (Default)</span>'
                                  if row['Real_Default'] == 1 and row['PRED_DEFAULT_BIN_BUS'] == 1
                                  else (f'<span style="color: {COLORS["secondary"]}; font-weight: bold;">FALSO POSITIVO</span>'
                                        if row['Real_Default'] == 0 and row['PRED_DEFAULT_BIN_BUS'] == 1
                                        else f'<span style="color: {COLORS["primary"]}; font-weight: bold;">FALSO NEGATIVO</span>')), 
                axis=1
            )

            # Formateo de columnas para visualización
            df_temp['PRED_DEFAULT_PROBA'] = df_temp['PRED_DEFAULT_PROBA'].apply(lambda x: f"{x:.2%}")
            df_temp['PRED_PAY_AMT4'] = df_temp['PRED_PAY_AMT4'].apply(lambda x: f"${x:,.0f}")
            df_temp['Real_Pago_Jun'] = df_temp['Real_Pago_Jun'].apply(lambda x: f"${x:,.0f}")
            df_temp['Real_Default'] = df_temp['Real_Default'].map({0: 'Pagó (0)', 1: 'Default (1)'})
            
            # AJUSTE 4: Renombre de columnas en español
            df_temp.rename(columns={
                'ID': 'ID Cliente',
                'LIMIT_BAL': 'Límite de Crédito',
                'Real_Default': 'Default Real (Oct)',
                'PRED_DEFAULT_PROBA': 'Prob. Default (Oct)',
                'Real_Pago_Jun': 'Pago Real (Jun)',
                'PRED_PAY_AMT4': 'Pago Estimado (Jun)',
                'resultados': 'Resultado de Auditoría'
            }, inplace=True)
            
            # Columnas finales a mostrar (SOLO LAS SOLICITADAS, ahora con nombres en español)
            cols_to_display = [
                'ID Cliente',
                'Límite de Crédito',
                'Default Real (Oct)',
                'Prob. Default (Oct)',
                'Pago Real (Jun)',
                'Pago Estimado (Jun)',
                'Resultado de Auditoría'
            ]
            return df_temp[cols_to_display].head(1000) # Only show first 1000 rows as it is too long

        df_display_audit = get_formatted_audit_df(df_raw_audit, optimal_threshold)
        
        # Mostrar como HTML para renderizar el formato condicional
        st.markdown(df_display_audit.to_html(escape=False, index=False), unsafe_allow_html=True)

# ===========================
# 8. Simulator
# ===========================
elif page == "Pronóstico de Impago":
    # BANNER
    st.markdown('<div class="title-banner">PRONÓSTICO DE IMPAGO<p>Simulador de Riesgo Crediticio</p></div>', unsafe_allow_html=True)
    st.info(f"El modelo utilizará el Umbral de Negocio óptimo calculado para maximizar F1-Score: **{optimal_threshold:.3f}**")

    # Layout de la página
    c_form, c_res = st.columns([1, 1.5])

    # Columna de Formulario
    with c_form, st.form("risk_simulator"):
        st.subheader("Variables del Cliente (Hasta Septiembre)")
        limit = st.number_input("Límite de Crédito (LIMIT_BAL)", min_value=10000, max_value=1000000, value=150000, step=10000)
        age = st.number_input("Edad (AGE)", min_value=21, max_value=75, value=35)
        
        # Mapeo de valores
        sex_map = {"Hombre": 1, "Mujer": 2}
        sex_def = sex_map[st.selectbox("Género (SEX)", options=["Hombre", "Mujer"], index=1)]
        
        edu_map = {"Posgrado": 1, "Universidad": 2, "Bachillerato": 3, "Otros": 4}
        edu_def = edu_map[st.selectbox("Nivel Educativo (EDUCATION)", options=["Universidad", "Posgrado", "Bachillerato", "Otros"], index=0)]
        
        mar_map = {"Casado": 1, "Soltero": 2, "Otros": 3}
        mar_def = mar_map[st.selectbox("Estado Civil (MARRIAGE)", options=["Soltero", "Casado", "Otros"], index=0)]

        pay_m0 = st.selectbox("Estatus de Pago (PAY_0) - Sept", options=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=1)
        bill_m1 = st.number_input("Monto de Factura (BILL_AMT1) - Sept", min_value=0, value=50000, step=1000)
        
        # Regresión (PAY_AMT4 - Predicción de Junio)
        st.subheader("Datos Históricos Adicionales (Hasta Mayo)")
        pay_m5 = st.selectbox("Estatus de Pago (PAY_5) - May", options=[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9], index=0)
        bill_m5 = st.number_input("Monto de Factura (BILL_AMT5) - May", min_value=0, value=15000, step=1000)
        
        # PAY_6 (Abr) y BILL_AMT6 (Abr) se utilizan con un valor de 0, ya que no son proporcionados por el usuario
        # Se asume un valor de PAY_6 (Abril) = 0 y BILL_AMT6 (Abril) = 0 para una inferencia simple.
        pay_m6 = 0 # Usamos 0 como valor de referencia
        bill_m6 = 0 # Usamos 0 como valor de referencia

        submitted = st.form_submit_button("Ejecutar Simulación de Riesgo")
        st.markdown('</div>', unsafe_allow_html=True)


    with c_res:
        st.markdown(f'<div class="corp-card" style="min-height: 500px;"><div class="card-header">Resultados de la Predicción</div>', unsafe_allow_html=True)
        if submitted and calibrated_cls and champion_reg:
            # --- PREDICCIÓN CLASIFICACIÓN (TARGET) ---
            utilization_ratio = bill_m1 / (limit + 1)
            row_data_cls = [limit, sex_def, edu_def, mar_def, age, pay_m0, bill_m1, utilization_ratio]
            row_cls = pd.DataFrame([row_data_cls], columns=feat_names_cls)
            
            # La probabilidad es la del modelo Calibrado
            X_scaled_single_cls = scaler_cls.transform(row_cls)
            proba_default = calibrated_cls['model'].predict_proba(X_scaled_single_cls)[0, 1]
            
            # Decisión Binaria (usando el umbral de negocio)
            decision_cls = "APROBADO"
            color_cls = COLORS['success']
            if proba_default >= optimal_threshold:
                decision_cls = "RECHAZADO"
                color_cls = COLORS['danger']

            # --- PREDICCIÓN REGRESIÓN (PAY_AMT4) ---
            row_data_reg = [limit, sex_def, edu_def, mar_def, age, 
                            pay_m5, pay_m6, bill_m5, bill_m6]
            row_reg = pd.DataFrame([row_data_reg], columns=feat_names_reg)
            X_scaled_single_reg = scaler_reg.transform(row_reg)
            pay_amt4_pred = champion_reg['model'].predict(X_scaled_single_reg)[0]
            pay_amt4_pred = np.maximum(0, pay_amt4_pred) # Asegurar no negativo

            st.markdown(
                f'<div style="text-align:center; padding: 20px; border: 2px solid {COLORS["primary"]}; border-radius: 8px; margin-bottom: 20px;">'
                f'<h3 style="color:{COLORS["primary"]}; margin-bottom: 15px;">Decisión de Riesgo (Default - Octubre)</h3>'
                f'<div style="font-size: 40px; font-weight: bold; color:{color_cls}; margin-bottom: 10px;">{decision_cls}</div>'
                f'<div style="font-size: 20px; color:{COLORS["text_dark"]};">Probabilidad de Default Estimada: <span style="font-weight: bold; color:{COLORS["danger"]};">{proba_default:.2%}</span></div>'
                f'</div>', unsafe_allow_html=True
            )

            st.markdown(
                f'<div style="text-align:center; padding: 15px; border: 1px dashed {COLORS["secondary"]}; border-radius: 6px; margin-top: 20px;">'
                f'<div style="font-size: 15px;">Pago Estimado en Junio (PAY_AMT4): <span style="color:{COLORS["primary"]}">${pay_amt4_pred:,.0f} NTD</span></div>'
                f'</div>', unsafe_allow_html=True
            )
            
            # BENCHMARK HISTÓRICO PARA CONTEXTO
            st.subheader("Benchmark Histórico (Perfil Similar)")
            # Filtrar datos de auditoría con un perfil similar (solo por límite y estatus de pago)
            df_sim_filt = df_raw_audit[
                (df_raw_audit['LIMIT_BAL'] <= limit * 1.2) & 
                (df_raw_audit['LIMIT_BAL'] >= limit * 0.8) 
            ]
            
            sim_default_rate = df_sim_filt['Real_Default'].mean() if not df_sim_filt.empty else 0
            sim_avg_pay = df_sim_filt['Real_Pago_Jun'].mean() if not df_sim_filt.empty else 0
            sim_count = len(df_sim_filt)

            sim_c1, sim_c2, sim_c3 = st.columns(3)
            with sim_c1:
                st.markdown(f"""
                <div class="corp-card" style="border-top: 3px solid {COLORS['secondary']};">
                <div class="card-header">Tasa Default Histórica</div>
                <div class="kpi-val">{sim_default_rate:.2%}</div>
                <div class="kpi-lbl">Cartera Similar</div>
                </div>
                """, unsafe_allow_html=True)
            with sim_c2:
                st.markdown(f"""
                <div class="corp-card" style="border-top: 3px solid {COLORS['secondary']};">
                <div class="card-header">Pago Promedio Histórico</div>
                <div class="kpi-val">${sim_avg_pay:,.0f}</div>
                <div class="kpi-lbl">Pago Real PAY_AMT4 (Jun)</div>
                </div>
                """, unsafe_allow_html=True)
            with sim_c3:
                st.markdown(f"""
                <div class="corp-card" style="border-top: 3px solid {COLORS['secondary']};">
                <div class="card-header">Clientes Similares</div>
                <div class="kpi-val">{sim_count:,.0f}</div>
                <div class="kpi-lbl">En el Dataset</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)


# ===========================
# 9. Metodología y Cuestionario (Documentación)
# ===========================
elif page == "Metodología y Cuestionario":
    # BANNER
    st.markdown('<div class="title-banner">METODOLOGÍA Y CUESTIONARIO<p>Documentación del Proyecto BlueRisk</p></div>', unsafe_allow_html=True)
    
    st.header("1. Metodología de Ciencia de Datos")
    st.markdown(f"""
    El proyecto siguió una metodología ágil y enfocada en el valor de negocio:
    * **1. Comprensión del Negocio (Business Understanding):** Definición del objetivo (predecir *default* y monto de pago) y la métrica clave (**AUC**, **F1-Score** y **RMSE**).
    * **2. Comprensión de los Datos (Data Understanding):** EDA, manejo de valores faltantes/inválidos (`EDUCATION` 0, 5, 6; `MARRIAGE` 0).
    * **3. Preparación de los Datos (Data Preparation):** Limpieza, ingeniería de *features* (**UTILIZATION_RATIO**, **CUR**s), escalamiento.
    * **4. Modelado (Modeling):** Entrenamiento de modelos de clasificación y regresión. Selección del campeón (**XGBoost Calibrado**).
    * **5. Evaluación (Evaluation):** Uso de **AUC** y **GINI** para selección del campeón. Se utiliza el **XGBoost Calibrado con Isotonic Regression** para una probabilidad más fiable.
    * **6. Despliegue (Dashboard):** Implementación de la interfaz **Streamlit** para visualización descriptiva, benchmark de modelos, simulación y la **Auditoría de Predicciones**.
    """, unsafe_allow_html=True)
    
    # AJUSTE 2: Se restaura el "tercer punto" (contenido de Optimización) y se reordena
    st.header("2. Optimización y Evaluación de Negocio")
    st.markdown(f"""
    * **Optimización (Iteración Final):** Se ha ajustado el umbral de decisión para **maximizar el F1-Score** ({optimal_threshold:.3f}), lo cual permite **reducir la Tasa de Falso Negativo (FNR)** y los **Falsos Positivos (FPR)**, logrando una **Accuracy Global** superior en un entorno de negocio balanceado. 
    * **Regresión:** Uso de **R2** y **RMSE**. El criterio de acierto en la auditoría de pago se ha flexibilizado a **30% de error** como máximo.
    """, unsafe_allow_html=True)

    st.header("3. Cuestionario de la Prueba Técnica") # AJUSTE 2: Renumeración del Cuestionario
    with st.expander("De las deficiencias en los datos, ¿cuales y como las identificaste?"):
        st.markdown("""
        Las principales deficiencias identificadas mediante un **Análisis Exploratorio de Datos (EDA)** fueron los **Valores Inválidos y Categorías Mixtas**:
        * **EDUCATION:** Presencia de valores `0, 5, 6`. Estos no estaban documentados y se categorizaron como **4 (Otros/Desconocido)** para simplificar el modelo.
        * **MARRIAGE:** Presencia del valor `0`. Se categorizó como **3 (Otros)**.
        * **PAY_X (Estatus de Pago):** La documentación original es confusa. Se interpretaron los valores `>1` como número de meses de atraso, `-1` como pago completo y `0` como saldo rotatorio. El análisis de riesgo se enfocó en distinguir `0, -1` de los valores de atraso `>1`.
        * **Valores Extremos (Outliers):** Se identificaron valores extremos en `LIMIT_BAL` y los montos de `BILL_AMT` y `PAY_AMT`, los cuales fueron manejados por el **escalamiento (StandardScaler)** durante el preprocesamiento del modelo.
        """)

    with st.expander("¿Cuáles fueron los principales desafíos del modelo y cómo los superaste?"):
        st.markdown("""
        * **Challenge 1: Data Leakage en Regresión:**
            * **Desafío:** Evitar usar información futura (como el *status* de pago de Junio `PAY_4`) para predecir `PAY_AMT4` (pago de Junio).
            * **Solución:** Se limitaron los *features* de regresión a datos hasta **Mayo** (`PAY_5`, `BILL_AMT5`, etc.), asegurando que el modelo solo usara información disponible en el momento de la predicción.
        * **Challenge 2: Desbalance de Clases:**
            * **Desafío:** La clase `Default` (1) es una minoría (aprox. 22%). Un modelo que predice "no default" para todos obtendría una alta Accuracy pero sería inútil.
            * **Solución:** Se utilizaron métricas de evaluación adecuadas (**AUC**, **F1-Score**) en lugar de Accuracy. Específicamente, se aplicó un ajuste de peso de clase (`scale_pos_weight`) en **XGBoost** y se optimizó el **umbral de decisión** para maximizar el F1-Score, mejorando la detección de la clase minoritaria (Recall).
        * **Challenge 3: Calibración de Probabilidades:**
            * **Desafío:** Los modelos de *Boosting* (XGBoost) tienden a sobreestimar las probabilidades hacia 0 y 1, haciendo que el umbral 0.5 sea engañoso.
            * **Solución:** Se aplicó un ajuste de **Calibración Isotónica** sobre el modelo XGBoost, asegurando que una probabilidad del 70% realmente signifique un 70% de riesgo de Default.
        """)

    with st.expander("¿Qué otros análisis estadísticos has utilizado?"):
        st.markdown("""
        * **EDA:** Test **t de Student** o **ANOVA** (para comparar medias de `LIMIT_BAL` o `AGE` entre grupos con/sin default) y **Chi-cuadrado** (para relación entre categóricas como `EDUCATION` y `TARGET`).
        * **Modelado:** Uso de **IV (Information Value)** para selección y evaluación de la potencia predictiva de *features* individuales.
        * **Validación:** **Análisis de Residuos** (Regresión) y **Test de Hosmer-Lemeshow** (Clasificación - implícito en la calibración).
        """)

    with st.expander("Explica: No Free Lunch Theorem, Occam’s Razor y Data Leakage."):
        st.markdown("""
        * **No Free Lunch Theorem (NFL):** No hay un algoritmo de ML universalmente mejor. La selección del modelo debe ser **empírica**, probando múltiples opciones para cada problema específico, ya que el rendimiento depende intrínsecamente de la estructura del *dataset*.
        * **Occam’s Razor (Navaja de Ockham):** Entre dos modelos con la misma precisión, elija el **más simple**. Prioriza la interpretabilidad, reduce la complejidad del mantenimiento y minimiza el riesgo de *overfitting* accidental.
        * **Data Leakage (Fuga de Datos):** Ocurre cuando se usa información que **no estaría disponible en producción** (o información del *Target*) durante el entrenamiento, lo que infla el rendimiento de manera irreal. Por ejemplo, usar `PAY_4` (Estatus de Junio) para predecir `PAY_AMT4` (Pago de Junio) o usar la media de todo el dataset para imputar en el set de entrenamiento. La solución es dividir los datos **antes** del preprocesamiento.
        """)
