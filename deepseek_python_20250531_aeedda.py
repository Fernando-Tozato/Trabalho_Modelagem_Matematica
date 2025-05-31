import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Classe CubicSpline (mantida igual à anterior)
class CubicSpline:
    # ... (Cole aqui a implementação completa da classe CubicSpline do início)

# Inicialização do app Dash
app = dash.Dash(__name__)
server = app.server

# Simulação de dados - você deve substituir por seu dataset real
def generate_sample_data():
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 2, 11)
    hours = int((end_date - start_date).total_seconds() / 3600)
    
    timestamps = [start_date + timedelta(hours=i) for i in range(hours)]
    usage = [50 + 30 * np.sin(i/50) + 5 * np.random.normal() for i in range(hours)]
    generation = [40 + 25 * np.cos(i/45) + 5 * np.random.normal() for i in range(hours)]
    
    # Criar status de estabilidade baseado em regras simples
    stability = []
    for i in range(hours):
        # Regras simples para simular estabilidade
        if abs(usage[i] - generation[i]) > 25:
            stability.append(1)  # Instável
        elif abs(usage[i] - usage[i-1] if i>0 else 0) > 10:
            stability.append(1)  # Instável
        else:
            stability.append(0)  # Estável
    
    return pd.DataFrame({
        'Timestamp': timestamps,
        'Real-time Power Usage (kW)': usage,
        'Real-time Power Generation (kW)': generation,
        'Power_Stability_Status': stability
    })

# Gerar dados de exemplo
df = generate_sample_data()

# Criar índices numéricos para o tempo
df['TimeIndex'] = range(len(df))

# Criar objetos spline para cada coluna de dados
spline_usage = CubicSpline(df['TimeIndex'].tolist(), df['Real-time Power Usage (kW)'].tolist())
spline_generation = CubicSpline(df['TimeIndex'].tolist(), df['Real-time Power Generation (kW)'].tolist())

# Layout do aplicativo
app.layout = html.Div([
    html.H1("Análise de Estabilidade da Rede Elétrica", style={'textAlign': 'center'}),
    
    html.Div([
        # Painel de seleção de dados
        html.Div([
            html.H3("Seleção de Dados"),
            dcc.DatePickerRange(
                id='date-range',
                min_date_allowed=df['Timestamp'].min(),
                max_date_allowed=df['Timestamp'].max(),
                start_date=df['Timestamp'].min(),
                end_date=df['Timestamp'].min() + timedelta(days=3)
            ),
            
            html.Label("Selecione as métricas:"),
            dcc.Dropdown(
                id='metric-selector',
                options=[
                    {'label': 'Uso de Energia (kW)', 'value': 'usage'},
                    {'label': 'Geração de Energia (kW)', 'value': 'generation'}
                ],
                value=['usage', 'generation'],
                multi=True
            ),
            
            html.Button('Atualizar Gráficos', id='update-button', n_clicks=0),
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
        
        # Painel de previsão de estabilidade
        html.Div([
            html.H3("Previsão de Estabilidade"),
            html.Label("Uso de Energia (kW):"),
            dcc.Input(id='input-usage', type='number', value=50),
            
            html.Label("Geração de Energia (kW):"),
            dcc.Input(id='input-generation', type='number', value=45),
            
            html.Label("Taxa de Variação (kW/h):"),
            dcc.Input(id='input-derivative', type='number', value=0),
            
            html.Button('Verificar Estabilidade', id='predict-button', n_clicks=0),
            
            html.Div(id='stability-output', style={'marginTop': '20px', 'fontSize': '20px', 'fontWeight': 'bold'})
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '20px'}),
    ]),
    
    # Gráficos
    dcc.Graph(id='main-graph'),
    dcc.Graph(id='derivative-graph'),
    
    # Armazenar dados do intervalo selecionado
    dcc.Store(id='selected-data')
])

# Callbacks para atualizar os gráficos
@app.callback(
    [Output('main-graph', 'figure'),
     Output('derivative-graph', 'figure'),
     Output('selected-data', 'data')],
    [Input('update-button', 'n_clicks')],
    [State('date-range', 'start_date'),
     State('date-range', 'end_date'),
     State('metric-selector', 'value')]
)
def update_graphs(n_clicks, start_date, end_date, selected_metrics):
    # Converter datas para datetime
    start_date = datetime.fromisoformat(start_date)
    end_date = datetime.fromisoformat(end_date)
    
    # Filtrar dados pelo intervalo selecionado
    mask = (df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)
    filtered_df = df[mask]
    
    # Criar dados para os gráficos
    main_fig = go.Figure()
    derivative_fig = go.Figure()
    
    # Configurações comuns
    time_index = filtered_df['TimeIndex'].tolist()
    timestamps = filtered_df['Timestamp'].tolist()
    
    # Adicionar dados originais e interpolados para cada métrica selecionada
    for metric in selected_metrics:
        if metric == 'usage':
            col_name = 'Real-time Power Usage (kW)'
            color = 'blue'
            spline = spline_usage
        else:
            col_name = 'Real-time Power Generation (kW)'
            color = 'green'
            spline = spline_generation
        
        # Dados originais
        original_y = filtered_df[col_name].tolist()
        main_fig.add_trace(go.Scatter(
            x=timestamps,
            y=original_y,
            mode='markers',
            name=f'Original - {col_name}',
            marker=dict(color=color, size=8)
        )
        
        # Curva interpolada (pontos densos)
        x_dense = np.linspace(time_index[0], time_index[-1], 500).tolist()
        y_dense = [spline.interpolated_function(x) for x in x_dense]
        
        # Converter índices para timestamps
        ts_dense = [df['Timestamp'].iloc[0] + timedelta(hours=x) for x in x_dense]
        
        main_fig.add_trace(go.Scatter(
            x=ts_dense,
            y=y_dense,
            mode='lines',
            name=f'Interpolado - {col_name}',
            line=dict(color=color, width=2)
        )
        
        # Derivada
        derivative = [spline.derivative(x) for x in x_dense]
        derivative_fig.add_trace(go.Scatter(
            x=ts_dense,
            y=derivative,
            mode='lines',
            name=f'Derivada - {col_name}',
            line=dict(color=color, width=2)
        )
    
    # Encontrar pontos críticos de instabilidade
    critical_points = []
    for idx, row in filtered_df.iterrows():
        if row['Power_Stability_Status'] == 1:
            critical_points.append(go.Scatter(
                x=[row['Timestamp']],
                y=[row['Real-time Power Usage (kW)']],
                mode='markers',
                name='Instabilidade',
                marker=dict(color='red', size=12, symbol='x')
            ))
    
    # Adicionar pontos de instabilidade ao gráfico principal
    for trace in critical_points:
        main_fig.add_trace(trace)
    
    # Formatar gráficos
    main_fig.update_layout(
        title='Dados de Energia com Interpolação',
        xaxis_title='Timestamp',
        yaxis_title='kW',
        hovermode='x unified'
    )
    
    derivative_fig.update_layout(
        title='Taxa de Variação (Derivada)',
        xaxis_title='Timestamp',
        yaxis_title='kW/h',
        hovermode='x unified'
    )
    
    # Armazenar dados do intervalo selecionado para correlação
    selected_data = {
        'usage': filtered_df['Real-time Power Usage (kW)'].tolist(),
        'generation': filtered_df['Real-time Power Generation (kW)'].tolist(),
        'stability': filtered_df['Power_Stability_Status'].tolist()
    }
    
    return main_fig, derivative_fig, selected_data

# Callback para previsão de estabilidade
@app.callback(
    Output('stability-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State('input-usage', 'value'),
     State('input-generation', 'value'),
     State('input-derivative', 'value'),
     State('selected-data', 'data')]
)
def predict_stability(n_clicks, usage, generation, derivative, selected_data):
    if n_clicks == 0:
        return ""
    
    # Validar entradas
    if None in [usage, generation, derivative]:
        return "Por favor, preencha todos os campos"
    
    # Calcular limiares baseados nos dados selecionados
    def calculate_threshold(data, factor=1.5):
        if not data:
            return 0
        mean_val = sum(data) / len(data)
        std_val = (sum((x - mean_val)**2 for x in data) / len(data))**0.5
        return mean_val + factor * std_val
    
    # Limiares baseados nos dados do intervalo selecionado
    usage_threshold = calculate_threshold(selected_data['usage'])
    generation_threshold = calculate_threshold(selected_data['generation'], factor=-1.5)
    derivative_threshold = calculate_threshold(
        [abs(d) for d in selected_data.get('derivative', [])] if selected_data.get('derivative') 
        else [abs(x - y) for x, y in zip(selected_data['usage'][1:], selected_data['usage'][:-1])]
    )
    
    # Regras de decisão baseadas em correlação
    instability_score = 0
    
    # 1. Diferença entre uso e geração
    power_diff = abs(usage - generation)
    if power_diff > usage_threshold * 0.7:
        instability_score += 1
    
    # 2. Geração abaixo do limiar
    if generation < generation_threshold:
        instability_score += 1
    
    # 3. Taxa de variação alta
    if abs(derivative) > derivative_threshold:
        instability_score += 1
    
    # Determinar estabilidade
    if instability_score >= 2:
        return html.Span("Status: INSTÁVEL", style={'color': 'red'})
    else:
        return html.Span("Status: ESTÁVEL", style={'color': 'green'})

if __name__ == '__main__':
    app.run_server(debug=True)