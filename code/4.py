import base64
import io
import math
import json
import numpy as np
import pandas as pd
import kmapper as km
from sklearn import ensemble, cluster

import dash
from dash import dcc, html, Input, Output, dash_table
import dash_cytoscape as cyto
import plotly.graph_objects as go
import plotly.express as px

##################################
# 1. Функция для условного форматирования DataTable
##################################
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

##################################
# 2. Построение единого Mapper‑графа
##################################
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

    # Вычислим размеры узлов (логарифмическая нормализация)
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

    # Базовые элементы для Cytoscape (узлы + рёбра)
    elements = []
    for nid in G['nodes']:
        elements.append({
            'data': {'id': nid, 'label': nid},
            'style': {
                'width': normalized_sizes[nid],
                'height': normalized_sizes[nid]
            }
        })
    for edge in G['simplices']:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            src, tgt = edge[0], edge[1]
            elements.append({
                'data': {'source': src, 'target': tgt},
                'classes': 'edge'
            })

    # Считаем количество пациентов target=1 и target=0
    node_data = {}
    for nid, members in G['nodes'].items():
        subset = df.iloc[members]
        node_data[nid] = {
            'members': members,
            'size': normalized_sizes[nid],
            'target1_count': int((subset['target'] == 1).sum()),
            'target0_count': int((subset['target'] == 0).sum())
        }
    return G, elements, node_data

##################################
# 3. Загрузка данных по умолчанию
##################################
df_path = "../datasets/db_nl_preprocessed-edit.csv"  # Поменяйте путь на корректный
df = pd.read_csv(df_path).fillna(0)
df['target'] = df['target'].astype(int)
features = df.columns.tolist()

cover_params = {"n_cubes": 20, "perc_overlap": 0.5}
G, base_elements, node_data = build_single_mapper(df, features, cover_params, n_clusters=3)

##################################
# 4. Раскрашиваем узлы по target (1 → pink, 0 → cyan)
##################################
def get_node_color(info):
    return "pink" if info['target1_count'] >= info['target0_count'] else "cyan"

for el in base_elements:
    if 'source' not in el['data']:  # узел
        nid = el['data']['id']
        el['style']['background-color'] = get_node_color(node_data[nid])

##################################
# 5. Столбцы таблиц
##################################
basic_cols = ["возраст", "рост", "масса_тела", "bmi", "пол", "target"]
basic_cols = [c for c in basic_cols if c in df.columns]
full_cols = df.columns.tolist()

table_columns_basic = [{"name": c, "id": c} for c in basic_cols]
table_columns_full = [{"name": c, "id": c} for c in full_cols]
style_data_conditional = generate_style_data_conditional_percentiles(df)

##################################
# 6. Дополнительные графики (гистограмма узлов, гистограммы признаков)
##################################
node_ids_list = list(G['nodes'].keys())
node_counts_list = [len(G['nodes'][nid]) for nid in node_ids_list]
node_bar_fig = go.Figure(data=go.Bar(x=node_ids_list, y=node_counts_list, marker_color='blue'))
node_bar_fig.update_layout(
    title='Количество строк на ноду в графе Mapper',
    xaxis_title='Node ID',
    yaxis_title='Число пациентов'
)

numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f not in ['target','пол']]

##################################
# 7. Подготовка к кэшированию (dcc.Store)
##################################
mapper_store_data = {
    "base_elements": base_elements,
    "node_data": node_data,
    "df_json": df.to_json(date_format='iso', orient='split')
}

##################################
# 8. Создаём приложение Dash
##################################
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Компонент Upload (не используется сейчас, но можно интегрировать при желании)
upload_component = dcc.Upload(
    id='upload-data',
    children=html.Div(['Перетащите или ', html.A('выберите CSV-файл')]),
    style={
        'width': '50%', 'height': '60px', 'lineHeight': '60px',
        'borderWidth': '2px', 'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    multiple=False
)

# Выпадающий список для (пока) неактивного фильтра
group_options = [
    {'label': 'Мужчины (все)', 'value': 'men'},
    {'label': 'Женщины (все)', 'value': 'women'},
    {'label': 'Мужчины, target=1', 'value': 'm1'},
    {'label': 'Мужчины, target=0', 'value': 'm0'},
    {'label': 'Женщины, target=1', 'value': 'f1'},
    {'label': 'Женщины, target=0', 'value': 'f0'},
    {'label': 'Все', 'value': 'all'}
]

filter_dropdown = dcc.Dropdown(
    id='filter-dropdown',
    options=group_options,
    multi=True,
    value=['all']
)

toggle_button = html.Button("Показать всю таблицу", id="toggle-columns", n_clicks=0)

store_component = dcc.Store(id="mapper-store", data=mapper_store_data)

# Создаём компоненты для Cytoscape, таблиц и гистограмм
cyto_graph = cyto.Cytoscape(
    id='cytoscape-graph',
    elements=base_elements,
    stylesheet=[
        {'selector': 'node', 'style': {
            'label': 'data(label)',
            'font-size': '1px'
        }},
        {'selector': 'node:selected', 'style': {
            'background-color': 'blue',
            'border-color': 'blue',
            'border-width': '2px'
        }},
        {'selector': 'edge', 'style': {
            'line-color': 'lightgray',
            'width': 0.5,
            'line-style': 'dotted'
        }}
    ],
    layout={
        'name': 'cose',
        'animate': False,
        'randomize': False,
        'nodeRepulsion': 3000,
        'idealEdgeLength': 40,
        'numIter': 500,
        'gravity': 0.3
    },
    style={'width': '100%', 'height': '600px', 'border': '2px solid black'},
    boxSelectionEnabled=True
)

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

feature_hist = dcc.Graph(id='feature-hist')

# ------------------- Layout приложения -------------------
# 1) Верхняя часть: Mapper (граф) + фильтр
# 2) Tabs – снизу, для таблиц и гистограмм
# 3) Контейнеры с id, стили управляются callback’ом (показываем/прячем при переключении вкладки)
app.layout = html.Div([
    html.H1("Интерактивный дашборд для анализа Mapper‑графа"),
    store_component,
    upload_component,

    # Граф всегда наверху
    html.Div([
        html.Div([
            html.Label("Выберите фильтр (пока не активен):"),
            filter_dropdown
        ], style={'width': '50%', 'padding': '10px'}),
        cyto_graph
    ], style={'margin-bottom': '30px'}),

    # Вкладки внизу
    dcc.Tabs(id='tabs', value='basic_table', children=[
        dcc.Tab(label='Упрощённая таблица', value='basic_table'),
        dcc.Tab(label='Полная таблица', value='full_table'),
        dcc.Tab(label='Гистограмма узлов', value='node_bar'),
        dcc.Tab(label='Гистограммы признаков', value='feature_hist')
    ]),

    # Контейнер для содержимого вкладок (скрываем/показываем через callback)
    html.Div([
        html.Div(id='basic-table-div', children=[
            data_table_basic,
            html.Div(id='table-basic-summary', style={'padding': '10px', 'font-weight': 'bold'})
        ], style={'display': 'none'}),

        html.Div(id='full-table-div', children=[
            data_table_full,
            html.Div(id='table-full-summary', style={'padding': '10px', 'font-weight': 'bold'})
        ], style={'display': 'none'}),

        html.Div(id='node-bar-div', children=[
            dcc.Graph(id='node-bar', figure=node_bar_fig, style={'width': '80%', 'margin': 'auto'})
        ], style={'display': 'none'}),

        html.Div(id='feature-hist-div', children=[
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
        ], style={'display': 'none'})
    ], style={'margin-top': '20px'}),

    # Кнопка "Показать всю таблицу" (переключает столбцы в "table-full")
    html.Div([toggle_button], style={'padding': '10px'})
])

# ------------------- Callbacks -------------------

# 1) Callback: переключение вкладок -> показывать/прятать соответствующие div
@app.callback(
    Output('basic-table-div', 'style'),
    Output('full-table-div', 'style'),
    Output('node-bar-div', 'style'),
    Output('feature-hist-div', 'style'),
    Input('tabs', 'value')
)
def switch_tabs(tab_value):
    style_show = {'display': 'block'}
    style_hide = {'display': 'none'}

    return (
        style_show if tab_value == 'basic_table' else style_hide,
        style_show if tab_value == 'full_table' else style_hide,
        style_show if tab_value == 'node_bar' else style_hide,
        style_show if tab_value == 'feature_hist' else style_hide
    )

# 2) Callback: окраска узлов (пока логика фильтров не активна)
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('filter-dropdown', 'value'),
    Input('mapper-store', 'data')
)
def update_cytoscape(selected_filters, mapper_store):
    if mapper_store is None:
        raise dash.exceptions.PreventUpdate
    base_elems = mapper_store.get("base_elements", [])
    cached_node_data = mapper_store.get("node_data", {})

    if not selected_filters or 'all' in selected_filters:
        # пока просто раскрашиваем по target
        pass

    new_elems = []
    for nid, info in cached_node_data.items():
        chosen_color = "pink" if info['target1_count'] >= info['target0_count'] else "cyan"
        new_elems.append({
            'data': {'id': nid, 'label': nid},
            'style': {
                'width': info['size'],
                'height': info['size'],
                'background-color': chosen_color
            }
        })
    shown_ids = {el['data']['id'] for el in new_elems}
    for edge in base_elems:
        d = edge.get('data', {})
        if d.get('source') in shown_ids and d.get('target') in shown_ids:
            new_elems.append(edge)
    return new_elems

# 3) Callback: обновление таблиц по выбранным узлам
@app.callback(
    Output('table-basic', 'data'),
    Output('table-basic-summary', 'children'),
    Output('table-full', 'data'),
    Output('table-full-summary', 'children'),
    Input('cytoscape-graph', 'selectedNodeData'),
    Input('mapper-store', 'data')
)
def update_tables(selected_nodes, mapper_store):
    if mapper_store is None:
        raise dash.exceptions.PreventUpdate
    df_json = mapper_store.get("df_json", None)
    if not df_json:
        raise dash.exceptions.PreventUpdate

    df_cached = pd.read_json(io.StringIO(df_json), orient='split')
    if not selected_nodes:
        return [], "Узлы не выбраны", [], "Узлы не выбраны"

    node_dict = mapper_store.get("node_data", {})
    all_indices = []
    for nd in selected_nodes:
        nid = nd.get('id')
        if nid in node_dict:
            all_indices.extend(node_dict[nid].get('members', []))
    unique_indices = list(set(all_indices))
    if not unique_indices:
        return [], "Нет строк", [], "Нет строк"

    subset_df = df_cached.iloc[unique_indices].copy()
    total = len(subset_df)
    t1_count = (subset_df['target'] == 1).sum()
    t0_count = (subset_df['target'] == 0).sum()
    summary_text = f"Выбрано узлов: {len(selected_nodes)}; записей: {total}; target=1: {t1_count}, target=0: {t0_count}"

    data_basic = subset_df[basic_cols].to_dict('records')
    data_full  = subset_df[full_cols].to_dict('records')
    return data_basic, summary_text, data_full, summary_text

# 4) Callback: гистограмма выбранного признака
@app.callback(
    Output('feature-hist', 'figure'),
    Input('feature-dropdown', 'value'),
    Input('mapper-store', 'data')
)
def update_feature_hist(feature, mapper_store):
    if mapper_store is None:
        raise dash.exceptions.PreventUpdate
    df_json = mapper_store.get("df_json", None)
    if not df_json or not feature:
        return go.Figure()

    df_cached = pd.read_json(io.StringIO(df_json), orient='split')
    if feature not in df_cached.columns:
        return go.Figure()

    fig = px.histogram(df_cached, x=feature, nbins=30, title=f"Гистограмма: {feature}")
    fig.update_layout(bargap=0.05)
    return fig

# 5) Callback: гистограмма узлов
@app.callback(
    Output('node-bar', 'figure'),
    Input('cytoscape-graph', 'elements'),
    Input('mapper-store', 'data')
)
def update_node_bar(elements, mapper_store):
    if mapper_store is None:
        raise dash.exceptions.PreventUpdate
    cached_node_data = mapper_store.get("node_data", {})
    node_ids = list(cached_node_data.keys())
    counts = [len(cached_node_data[nid]['members']) for nid in node_ids]
    fig = go.Figure(data=go.Bar(x=node_ids, y=counts, marker_color='blue'))
    fig.update_layout(
        title='Количество строк на ноду в графе Mapper',
        xaxis_title='Node ID',
        yaxis_title='Число пациентов'
    )
    return fig

# 6) Callback: переключение столбцов в "Полной таблице" (basic / full)
@app.callback(
    Output('table-full', 'columns'),
    Input('toggle-columns', 'n_clicks')
)
def update_columns(n_clicks):
    if n_clicks % 2 == 1:
        return table_columns_full
    else:
        return table_columns_basic

# ------------------- Запуск -------------------
if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8061)