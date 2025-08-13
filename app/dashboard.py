import os
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Rutas y archivos de trabajo
DATA_PATH   = os.environ.get('PREDICTIONS_PATH', 'data/predictions.csv')
CLEAN_PATH  = 'data/telco_churn_cleaned.csv'
SAMPLE_PATH = 'data/sample_telco_churn.csv'

# ---------- Utilidades ----------
def fig_vacia(titulo='Sin datos'):
    fig = go.Figure()
    fig.update_layout(title=titulo, xaxis_title=None, yaxis_title=None)
    return fig

def elegir_col_proba(df: pd.DataFrame) -> str | None:
    # Preferimos la más “humana” si existe
    if 'prob_abandono' in df.columns:
        return 'prob_abandono'
    if 'churn_proba' in df.columns:
        return 'churn_proba'
    return None

def _modelo_rapido(df: pd.DataFrame) -> pd.DataFrame | None:
    """Si no hay predictions.csv, calculo probabilidad de abandono al vuelo."""
    try:
        if 'Churn' not in df.columns:
            return None
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        X = df.drop(columns=['Churn']); y = df['Churn']
        cat = [c for c in X.columns if X[c].dtype == 'object']
        num = [c for c in X.columns if X[c].dtype != 'object']

        pre  = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat),
            ('num', StandardScaler(with_mean=False),      num),
        ])
        pipe = Pipeline([('pre', pre), ('clf', LogisticRegression(max_iter=1000))]).fit(X, y)

        out = df.copy()
        out['prob_abandono'] = pipe.predict_proba(X)[:, 1]
        return out
    except Exception as e:
        print('Error en modelo rápido:', e)
        return None

def cargar_datos():
    """Devuelve (df, fuente_str). Intenta predictions -> limpio -> sample."""
    # 1) predictions.csv
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH)
            # Si viene con nombre técnico, no pasa nada: elegimos columna más abajo
            return df, 'predictions.csv'
        except Exception as e:
            print('Error leyendo predictions.csv:', e)

    # 2) limpio
    if os.path.exists(CLEAN_PATH):
        try:
            df = pd.read_csv(CLEAN_PATH)
            df2 = _modelo_rapido(df)
            if df2 is not None:
                return df2, 'telco_churn_cleaned.csv (modelo rápido)'
        except Exception as e:
            print('Error usando limpio:', e)

    # 3) sample (incluido)
    if os.path.exists(SAMPLE_PATH):
        try:
            df = pd.read_csv(SAMPLE_PATH)
            df2 = _modelo_rapido(df)
            if df2 is not None:
                return df2, 'sample_telco_churn.csv (modelo rápido)'
        except Exception as e:
            print('Error usando sample:', e)

    return None, 'sin datos'

# ---------- App ----------
app = Dash(__name__)
server = app.server

badge_style = {
    'display':'inline-block','padding':'4px 8px','borderRadius':'10px',
    'background':'#eef','color':'#223','fontSize':'12px','marginLeft':'8px'
}

app.layout = html.Div([
    html.H2('Abandono de clientes — Panel interactivo'),
    html.Div(id='badge-fuente', style={'marginBottom':'6px'}),

    html.Div([
        html.Label('Umbral de riesgo (probabilidad de abandono):'),
        dcc.Slider(id='thr', min=0.0, max=1.0, step=0.01, value=0.5,
                   marks={0.0:'0.0',0.25:'0.25',0.5:'0.5',0.75:'0.75',1.0:'1.0'}),
    ], style={'maxWidth':'600px', 'marginBottom':'10px'}),

    html.Div([
        html.Label('Tipo de contrato:'),
        dcc.Dropdown(id='contract', options=[], value=None, multi=True,
                     placeholder='Selecciona tipo(s) de contrato...'),
    ], style={'maxWidth':'500px'}),

    html.Div(id='kpi', style={'marginTop':'8px','marginBottom':'8px'}),

    dcc.Graph(id='hist_proba'),
    dcc.Graph(id='bar_contract'),

    # Auto-refresh para detectar archivos generados mientras la app corre
    dcc.Interval(id='tick', interval=3000, n_intervals=0)
])

@app.callback(
    Output('contract','options'),
    Output('kpi','children'),
    Output('hist_proba','figure'),
    Output('bar_contract','figure'),
    Output('badge-fuente','children'),
    Input('thr','value'),
    Input('contract','value'),
    Input('tick','n_intervals')
)
def actualizar(umbral, contratos, _n):
    df, fuente = cargar_datos()

    # Badge de fuente de datos
    badge = html.Span(f'Datos: {fuente}', style=badge_style)

    # Opciones del dropdown
    opciones = []
    if isinstance(df, pd.DataFrame) and 'Contract' in df.columns and not df.empty:
        opciones = [{'label': v, 'value': v} for v in sorted(df['Contract'].dropna().unique())]

    if not isinstance(df, pd.DataFrame) or df.empty:
        msg = html.Div('Sin datos disponibles en data/. Ejecuta los notebooks o deja el sample.')
        return opciones, msg, fig_vacia('Sin datos'), fig_vacia('Sin datos'), badge

    col_proba = elegir_col_proba(df)
    if col_proba is None:
        msg = html.Div("No encontré la columna de probabilidad ('prob_abandono' o 'churn_proba').")
        return opciones, msg, fig_vacia('Falta probabilidad'), fig_vacia('Falta probabilidad'), badge

    dff = df.copy()
    if contratos and 'Contract' in dff.columns:
        dff = dff[dff['Contract'].isin(contratos)]

    if len(dff) == 0:
        msg = html.Div('No hay filas tras aplicar los filtros.')
        return opciones, msg, fig_vacia('Sin filas'), fig_vacia('Sin filas'), badge

    umbral = float(umbral or 0.5)
    alto_riesgo = int((dff[col_proba] >= umbral).sum())
    kpi = html.Div([
        html.H4(f'Registros: {len(dff)} | Alto riesgo (≥ {umbral:.2f}): '
                f'{alto_riesgo} ({(alto_riesgo/max(len(dff),1))*100:.1f}%)')
    ])

    # Histograma de probabilidad de abandono
    try:
        fig_hist = px.histogram(dff, x=col_proba, nbins=30,
                                title='Distribución de probabilidad de abandono')
        fig_hist.update_layout(xaxis_title='Probabilidad de abandono', yaxis_title='Conteo')
    except Exception:
        fig_hist = fig_vacia('No se pudo generar histograma')

    # Barras por contrato (tasa de alto riesgo)
    if 'Contract' in dff.columns:
        try:
            grp = dff.groupby('Contract').apply(
                lambda x: (x[col_proba] >= umbral).mean()
            ).reset_index(name='tasa_alto_riesgo')
            grp = grp.sort_values('tasa_alto_riesgo', ascending=False)
            fig_bar = px.bar(grp, x='Contract', y='tasa_alto_riesgo',
                             title='Tasa de alto riesgo por contrato')
            fig_bar.update_layout(xaxis_title='Contrato', yaxis_title='Tasa')
        except Exception:
            fig_bar = fig_vacia('No se pudo generar barra por contrato')
    else:
        fig_bar = fig_vacia('No existe la columna Contrato en los datos')

    return opciones, kpi, fig_hist, fig_bar, badge

if __name__ == '__main__':
    app.run(debug=True)