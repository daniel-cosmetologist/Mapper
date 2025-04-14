import base64
import io
import math
import numpy as np
import pandas as pd
import kmapper as km
from sklearn import ensemble, cluster
import dash
from dash import dcc, html, Input, Output, dash_table, callback_context as ctx
import dash_cytoscape as cyto
import plotly.graph_objects as go
import plotly.express as px

############################
# 1. Функция условного форматирования для DataTable (окраска по 5-му и 95-му процентилю)
############################
def generate_style_data_conditional_percentiles(df):
    styles = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        p5 = np.percentile(df[col].dropna(), 5)
        p95 = np.percentile(df[col].dropna(), 95)
        styles.append({
            'if': {'filter_query': f'{{{col}}} < {p5}', 'column_id': col},
            'backgroundColor': 'darkblue',
            'color': 'white'
        })
        styles.append({
            'if': {'filter_query': f'{{{col}}} > {p95}', 'column_id': col},
            'backgroundColor': 'red',
            'color': 'white'
        })
    return styles

############################
# 2. Функция построения единого Mapper‑графа (строится один раз)
############################
def build_single_mapper(df, features, cover_params, n_clusters=3):
    X = df[features].to_numpy()
    projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
    projector.fit(X)
    lens1 = projector.decision_function(X)
    
    mapper = km.KeplerMapper(verbose=2)
    lens2 = mapper.fit_transform(X, projection="knn_distance_5")
    lens = np.c_[lens1, lens2]
    
    cover = km.Cover(
        n_cubes=cover_params.get("n_cubes", 10),
        perc_overlap=cover_params.get("perc_overlap", 0.5)
    )
    G = mapper.map(
        lens, 
        X,
        cover=cover,
        clusterer=cluster.AgglomerativeClustering(n_clusters=n_clusters)
    )
    
    # Вычисляем размеры узлов (логарифмическая нормализация)
    node_sizes = {nid: len(members) for nid, members in G['nodes'].items()}
    min_size, max_size_val = 2, 9
    max_count = max(node_sizes.values()) if node_sizes else 1
    normalized_sizes = {}
    for nid, cnt in node_sizes.items():
        if cnt > 1:
            log_size = math.log(cnt, max_count)
            s = min_size + (max_size_val - min_size) * (log_size / math.log(max_count, max_count))
        else:
            s = min_size
        normalized_sizes[nid] = s

    # Формируем элементы (узлы и рёбра) для Cytoscape
    elements = []
    for nid in G['nodes']:
        elements.append({
            'data': {'id': nid, 'label': nid},
            'style': {'width': normalized_sizes[nid], 'height': normalized_sizes[nid]}
        })
    for edge in G['simplices']:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, tgt = edge[0], edge[1]
            elements.append({'data': {'source': src, 'target': tgt}})
    
    # Собираем дополнительную информацию по узлам: считаем количество пациентов по target
    node_data = {}
    for nid, members in G['nodes'].items():
        subset = df.iloc[members]
        node_data[nid] = {
            'members': members,
            'size': normalized_sizes[nid],
            'target1_count': (subset['target'] == 1).sum(),
            'target0_count': (subset['target'] == 0).sum()
        }
    
    return G, elements, node_data

############################
# 3. Загрузка данных по умолчанию и построение Mapper‑графа
############################
df_path = "../datasets/db_nl_preprocessed-edit.csv"  # Укажите корректный путь к файлу
df = pd.read_csv(df_path).fillna(0)
df['target'] = df['target'].astype(int)
features = df.columns.tolist()
cover_params = {"n_cubes": 20, "perc_overlap": 0.7}
G, base_elements, node_data = build_single_mapper(df, features, cover_params, n_clusters=3)
# Сохраняем глобальный mapping для таблицы
global_node_mapping = node_data.copy()

############################
# 4. Фиксированные цвета для вершин
############################
# Если у узла количество пациентов с target=1 >= target=0, цвет = pink, иначе cyan.
def get_node_color(info):
    if info['target1_count'] >= info['target0_count']:
        return "pink"
    else:
        return "cyan"

# Обновляем базовые элементы с корректными цветами
for el in base_elements:
    if 'source' not in el['data']:  # только для узлов
        nid = el['data']['id']
        el['style']['background-color'] = get_node_color(node_data[nid])

############################
# 5. Опции для фильтрации (фильтр для узлов по группам)
############################
group_options = [
    {'label': 'Мужчины (все)', 'value': 'men'},
    {'label': 'Женщины (все)', 'value': 'women'},
    {'label': 'Мужчины, target=1', 'value': 'm1'},
    {'label': 'Мужчины, target=0', 'value': 'm0'},
    {'label': 'Женщины, target=1', 'value': 'f1'},
    {'label': 'Женщины, target=0', 'value': 'f0'},
    {'label': 'Все', 'value': 'all'}
]

############################
# 6. Столбцы для DataTable
############################
basic_cols = ["возраст", "рост", "масса_тела", "bmi", "пол", "target"]
basic_cols = [c for c in basic_cols if c in df.columns]
full_cols = df.columns.tolist()

table_columns_basic = [{"name": c, "id": c} for c in basic_cols]
table_columns_full = [{"name": c, "id": c} for c in full_cols]
style_data_conditional = generate_style_data_conditional_percentiles(df)

############################
# 7. Дополнительные графики
############################
# Гистограмма узлов: количество записей (пациентов) в каждом узле
node_ids_list = list(G['nodes'].keys())
node_counts_list = [len(G['nodes'][nid]) for nid in node_ids_list]
bar_fig = go.Figure(data=go.Bar(x=node_ids_list, y=node_counts_list, marker_color='blue'))
bar_fig.update_layout(
    title='Количество строк на ноду в графе Mapper',
    xaxis_title='Node ID',
    yaxis_title='Число пациентов'
)

# Список числовых признаков для гистограмм (исключаем target и пол)
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in ['target', 'пол']]

############################
# 8. Интерфейс Dash с вкладками
############################
app = dash.Dash(__name__)
server = app.server

# Компонент загрузки файла (пока оставляем, но данные уже загружены по умолчанию)
upload_component = dcc.Upload(
    id='upload-data',
    children=html.Div(['Перетащите или ', html.A('выберите CSV-файл')]),
    style={
        'width': '50%', 'height': '60px', 'lineHeight': '60px',
        'borderWidth': '2px', 'borderStyle': 'dashed',
        'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
    },
    multiple=False
)

# Dropdown для выбора фильтра
filter_dropdown = dcc.Dropdown(
    id='filter-dropdown',
    options=group_options,
    multi=True,
    value=['all']  # по умолчанию выбираем "Все"
)

# Кнопка для переключения набора столбцов (basic/full)
toggle_button = html.Button("Показать всю таблицу", id="toggle-columns", n_clicks=0)

# Cytoscape-компонент (граф строится один раз; рамка добавлена в стиле)
cyto_graph = cyto.Cytoscape(
    id='cytoscape-graph',
    elements=base_elements,
    stylesheet=[
        {'selector': 'node', 'style': {'label': 'data(label)', 'font-size': '1px'}},
        {'selector': 'node:selected', 'style': {'background-color': 'red'}},
        {'selector': 'edge', 'style': {'line-color': 'light-gray', 'width': 0.1}}
    ],
    layout={
        'name': 'cose',
        'randomize': False,
        'nodeRepulsion': 8000,
        'idealEdgeLength': 80,
        'numIter': 1000,
        'gravity': 0.2
    },
    style={'width': '100%', 'height': '800px', 'border': '2px solid black'},
    boxSelectionEnabled=True
)

# DataTable для упрощённой таблицы
data_table_basic = dash_table.DataTable(
    id='table-basic',
    data=[],
    columns=table_columns_basic,
    page_size=100,
    filter_action="native",
    sort_action="native",
    export_format="csv",
    style_table={'height': '500px', 'overflowY': 'auto'},
    fixed_rows={'headers': True},
    style_data_conditional=style_data_conditional
)

# DataTable для полной таблицы
data_table_full = dash_table.DataTable(
    id='table-full',
    data=[],
    columns=table_columns_full,
    page_size=50,
    filter_action="native",
    sort_action="native",
    export_format="csv",
    style_table={'height': '500px', 'overflowY': 'auto'},
    fixed_rows={'headers': True},
    style_data_conditional=style_data_conditional
)

# График для гистограмм признаков – обновляется по выбору признака
feature_hist = dcc.Graph(id='feature-hist')

# Интерфейс с вкладками: граф, упрощённая таблица, полная таблица, гистограмма узлов, гистограммы признаков
app.layout = html.Div([
    html.H1("Интерактивный дашборд для анализа Mapper‑графа"),
    upload_component,
    dcc.Tabs(id='tabs', value='tab-graph', children=[
        dcc.Tab(label='Граф', value='tab-graph', children=[
            html.Div([
                html.Label("Выберите фильтр:"),
                filter_dropdown
            ], style={'width': '50%', 'padding': '10px'}),
            cyto_graph
        ]),
        dcc.Tab(label='Упрощённая таблица', value='tab-table-basic', children=[
            data_table_basic,
            html.Div(id='table-basic-summary', style={'padding': '10px', 'font-weight': 'bold'})
        ]),
        dcc.Tab(label='Полная таблица', value='tab-table-full', children=[
            data_table_full,
            html.Div(id='table-full-summary', style={'padding': '10px', 'font-weight': 'bold'})
        ]),
        dcc.Tab(label='Гистограмма узлов', value='tab-bar', children=[
            dcc.Graph(id='node-bar', figure=bar_fig, style={'width': '80%', 'margin': 'auto'})
        ]),
        dcc.Tab(label='Гистограммы признаков', value='tab-hist', children=[
            html.Div([
                html.Label("Выберите признак:"),
                dcc.Dropdown(
                    id='feature-dropdown',
                    options=[{'label': f, 'value': f} for f in numeric_features],
                    value=numeric_features[0] if numeric_features else None,
                    clearable=False
                )
            ], style={'width': '40%', 'padding': '10px'}),
            feature_hist
        ])
    ]),
    html.Div([toggle_button], style={'padding': '10px'}),
])
    
############################
# 9. Callback для обновления окраски узлов (граф не перестраивается, меняется только цвет)
############################
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('filter-dropdown', 'value')
)
def update_cytoscape(selected_filters):
    # Если выбран "Все" или фильтр пуст, используем все группы – раскраска по target:
    if not selected_filters or 'all' in selected_filters:
        active_filters = {'m1', 'm0', 'f1', 'f0'}
    else:
        active_filters = set(selected_filters)
    
    new_elems = []
    for nid, info in node_data.items():
        # Для каждого узла определяем его характеристики
        m1 = info['male_t1_count']
        m0 = info['male_t0_count']
        f1 = info['female_t1_count']
        f0 = info['female_t0_count']
        # Для простоты в данном варианте раскрашиваем узлы по target:
        # Если сумма (муж target=1 + жен target=1) больше или равна сумме (муж target=0 + жен target=0) → pink,
        # иначе → cyan.
        chosen_color = "pink" if (m1 + f1) >= (m0 + f0) else "cyan"
        new_elems.append({
            'data': {'id': nid, 'label': nid},
            'style': {
                'width': info['size'],
                'height': info['size'],
                'background-color': chosen_color
            }
        })
    # Добавляем рёбра – только если оба конца присутствуют
    shown_ids = {el['data']['id'] for el in new_elems}
    for edge in base_elements:
        d = edge.get('data', {})
        if d.get('source') in shown_ids and d.get('target') in shown_ids:
            new_elems.append(edge)
    return new_elems

############################
# 10. Callback для обновления таблиц при выборе узлов
############################
@app.callback(
    [Output('table-basic', 'data'),
     Output('table-basic-summary', 'children'),
     Output('table-full', 'data'),
     Output('table-full-summary', 'children')],
    Input('cytoscape-graph', 'selectedNodeData')
)
def update_tables(selected_nodes):
    if not selected_nodes:
        return [], "Узлы не выбраны", [], "Узлы не выбраны"
    all_indices = []
    for nd in selected_nodes:
        nid = nd.get('id')
        if nid in global_node_mapping:
            all_indices.extend(global_node_mapping[nid].get('members', []))
    unique_indices = list(set(all_indices))
    if not unique_indices:
        return [], "Нет строк", [], "Нет строк"
    subset_df = df.iloc[unique_indices].copy()
    total = len(subset_df)
    t1_count = (subset_df['target'] == 1).sum()
    t0_count = (subset_df['target'] == 0).sum()
    summary_text = f"Выбрано узлов: {len(selected_nodes)}; записей: {total}; target=1: {t1_count}, target=0: {t0_count}"
    data_basic = subset_df[basic_cols].to_dict('records')
    data_full  = subset_df[full_cols].to_dict('records')
    return data_basic, summary_text, data_full, summary_text

############################
# 11. Callback для построения гистограммы выбранного признака
############################
@app.callback(
    Output('feature-hist', 'figure'),
    Input('feature-dropdown', 'value')
)
def update_feature_hist(feature):
    if feature is None or feature not in df.columns:
        return go.Figure()
    fig = px.histogram(df, x=feature, nbins=30, title=f"Гистограмма: {feature}")
    fig.update_layout(bargap=0.05)
    return fig

############################
# 12. Callback для построения гистограммы узлов (по числу пациентов в узле)
############################
@app.callback(
    Output('node-bar', 'figure'),
    Input('cytoscape-graph', 'elements')
)
def update_node_bar(elements):
    node_ids = list(node_data.keys())
    counts = [len(node_data[nid]['members']) for nid in node_ids]
    fig = go.Figure(data=go.Bar(x=node_ids, y=counts, marker_color='blue'))
    fig.update_layout(
        title='Количество строк на ноду в графе Mapper',
        xaxis_title='Node ID',
        yaxis_title='Число пациентов'
    )
    return fig

############################
# 13. Запуск приложения
############################
if __name__ == '__main__':
    app.run_server(debug=True, port=8061)
