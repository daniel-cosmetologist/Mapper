import math
import numpy as np
import pandas as pd
import kmapper as km
from sklearn import ensemble, cluster
import dash
from dash import dcc, html, Dash, Input, Output, dash_table
import plotly.graph_objects as go
import dash_cytoscape as cyto
from thresholds import thresholds

# Функция для интерполяции цвета по градиенту
def gradient_color(value, low, high, color_low, color_high):
    """
    Интерполирует цвет между цветами color_low и color_high, если value лежит между low и high.
    Цвета задаются кортежами (R, G, B).
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return f"rgb({color_low[0]},{color_low[1]},{color_low[2]})"
    if value <= low:
        return f"rgb({color_low[0]},{color_low[1]},{color_low[2]})"
    if value >= high:
        return f"rgb({color_high[0]},{color_high[1]},{color_high[2]})"
    ratio = (value - low) / (high - low)
    r = int(color_low[0] + (color_high[0] - color_low[0]) * ratio)
    g = int(color_low[1] + (color_high[1] - color_low[1]) * ratio)
    b = int(color_low[2] + (color_high[2] - color_low[2]) * ratio)
    return f"rgb({r},{g},{b})"


# Функция для построения Mapper-графа и вычисления списка элементов для Cytoscape.
def build_mapper_elements(data, features, cover_params={"n_cubes": 20, "perc_overlap": 0.7}, n_clusters=3):
    X = np.array(data[features])
    projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
    projector.fit(X)
    lens1 = projector.decision_function(X)
    mapper = km.KeplerMapper(verbose=3)
    lens2 = mapper.fit_transform(X, projection="knn_distance_5")
    lens = np.c_[lens1, lens2]
    cover = km.Cover(n_cubes=cover_params["n_cubes"], perc_overlap=cover_params["perc_overlap"])
    G = mapper.map(lens, X, cover=cover, clusterer=cluster.AgglomerativeClustering(n_clusters=n_clusters))
    
    # Вычисляем размеры узлов (логарифмическая нормализация)
    node_sizes = {}
    for node_id, members in G['nodes'].items():
        node_sizes[node_id] = len(members)
    min_size = 2
    max_size = 9
    if node_sizes:
        max_count = max(node_sizes.values())
    else:
        max_count = 1
    normalized_sizes = {}
    for node, count in node_sizes.items():
        if count > 1:
            log_size = math.log(count, max_count)
            normalized_size = min_size + (max_size - min_size) * (log_size / math.log(max_count, max_count))
        else:
            normalized_size = min_size
        normalized_sizes[node] = normalized_size

    elements = [
        {'data': {'id': node, 'label': node}, 'style': {'width': normalized_sizes[node], 'height': normalized_sizes[node]}}
        for node in G['nodes']
    ]
    elements += [
        {'data': {'source': edge[0], 'target': edge[1]}}
        for edge in G['simplices'] if isinstance(edge, (list, tuple)) and len(edge) >= 2
    ]
    
    # Помимо стандартных размеров можно вычислить дополнительные метрики для цвета
    node_data_dict = {}
    target_col = 'target'
    gender_col = 'пол'
    for node_id, members in G['nodes'].items():
        local_df = data.iloc[members]
        count = len(members)
        frac_ms = local_df[target_col].mean() if count > 0 else 0
        frac_male = local_df[gender_col].mean() if count > 0 else 0
        node_data_dict[node_id] = {
            'frac_ms': frac_ms,
            'frac_male': frac_male,
            'size': normalized_sizes[node_id]
        }
    # Перекодируем узлы с дополнительными данными (цвет по frac_ms, например)
    new_elements = []
    for node_id in G['nodes']:
        data_vals = node_data_dict[node_id]
        # Цвет узла будем задавать по доле наличия метаболического синдрома (frac_ms)
        # Для frac_ms интерполируем от зеленого (низкое значение) до красного (высокое)
        color = gradient_color(data_vals['frac_ms'], 0, 1, (0, 255, 0), (255, 0, 0))
        new_elements.append({
            'data': {
                'id': node_id,
                'label': node_id,
                'frac_ms': data_vals['frac_ms'],
                'frac_male': data_vals['frac_male']
            },
            'style': {
                'width': data_vals['size'],
                'height': data_vals['size'],
                'background-color': color
            }
        })
    # Добавляем рёбра
    for edge in G['simplices']:
        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
            source, target = edge[0], edge[1]
            new_elements.append({'data': {'source': source, 'target': target}})
    return G, new_elements, node_data_dict


# Загрузка и подготовка датасета
dataset_path = '../datasets/db_nl_preprocessed-edit.csv'
df = pd.read_csv(dataset_path)
df = df.fillna(0)
# Предполагаем, что все столбцы используются для Mapper
features = [c for c in df.columns]

# Предполагаем, что:
# - столбец 'target' принимает значения 1 (диагноз метаболического синдрома) и 0
# - столбец 'пол' имеет строковые значения: "male" или "female"
df['target'] = df['target'].astype(int)
# df['пол'] = df['пол'].astype(str).str.lower()  # гарантируем нижний регистр

# Разбиваем датасет на 4 группы
df_male_target1 = df[(df['пол'] == 1) & (df['target'] == 1)]
df_male_target0 = df[(df['пол'] == 1) & (df['target'] == 0)]
df_female_target1 = df[(df['пол'] == 0) & (df['target'] == 1)]
df_female_target0 = df[(df['пол'] == 0) & (df['target'] == 0)]

# Построим Mapper-графы для каждой группы
cover_params = {"n_cubes": 10, "perc_overlap": 0.5}  # можно экспериментировать с этими параметрами
G_m1, elements_m1, node_data_m1 = build_mapper_elements(df_male_target1, features, cover_params)
G_m0, elements_m0, node_data_m0 = build_mapper_elements(df_male_target0, features, cover_params)
G_f1, elements_f1, node_data_f1 = build_mapper_elements(df_female_target1, features, cover_params)
G_f0, elements_f0, node_data_f0 = build_mapper_elements(df_female_target0, features, cover_params)





# Словарь, где ключ – группа, значение – элементы для Cytoscape
group_elements = {
    "male_target1": elements_m1,
    "male_target0": elements_m0,
    "female_target1": elements_f1,
    "female_target0": elements_f0
}

# Для простоты создадим выпадающий список с множественным выбором
group_options = [
    {'label': 'Мужчины с target=1', 'value': 'male_target1'},
    {'label': 'Мужчины с target=0', 'value': 'male_target0'},
    {'label': 'Женщины с target=1', 'value': 'female_target1'},
    {'label': 'Женщины с target=0', 'value': 'female_target0'},
    {'label': 'Все вместе', 'value': 'all'},
]

# Базовый stylesheet Cytoscape (остальные стили задаются индивидуально в элементах)
cytoscape_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': 'data(label)',
            'font-size': '1px'
        }
    },
    {
        'selector': 'node:selected',
        'style': {'background-color': 'red'}
    },
    {
        'selector': 'edge',
        'style': {
            'line-color': 'light-gray',
            'width': 0.1
        }
    }
]

# Построим барчарт с количеством узлов (для общей картины по всем группам)
overall_node_ids = list(df.index)  # это пример, можно оставить предыдущий граф с узлами Mapper
bar_fig = go.Figure(data=go.Bar(x=["Всего"], y=[len(df)]))
bar_fig.update_layout(title='Общее количество строк в датасете')

# Генерация стилей для DataTable с использованием градиентного окрашивания
# Здесь используем функцию, которая перебирает все строки датафрейма
def build_gradient_styles(data):
    styles = []
    color_low = (173, 216, 230)   # светло-голубой
    color_high = (255, 182, 193)  # светло-розовый
    for i, row in enumerate(data):
        for col, bounds in thresholds.items():
            if col in row:
                try:
                    val = float(row[col])
                except (TypeError, ValueError):
                    continue
                color = gradient_color(val, bounds['low'], bounds['high'], color_low, color_high)
                styles.append({
                    'if': {'row_index': i, 'column_id': col},
                    'backgroundColor': color,
                    'color': 'black'
                })
    return styles
table_data = df.to_dict('records')
table_columns = [{"name": c, "id": c} for c in df.columns]
table_styles = build_gradient_styles(table_data)

# Определяем интерфейс Dash-приложения
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.H1('Интерактивный дашборд для анализа Mapper-графов'),
    html.Div([
        html.Label("Выберите группы для отображения:"),
        dcc.Dropdown(
            id='group-filter',
            options=group_options,
            multi=True,
            value=['all']
        )
    ], style={'width': '50%', 'padding': '10px'}),
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=[],  # элементы будут обновляться через callback
        stylesheet=cytoscape_stylesheet,
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '800px'},
        boxSelectionEnabled=True
    ),
    html.H2('Данные выбранных узлов'),
    dash_table.DataTable(
        id='data-table',
        data=table_data,
        columns=table_columns,
        page_size=100,
        style_table={'height': '500px', 'overflowY': 'auto'},
        fixed_rows={'headers': True},
        style_data_conditional=table_styles
    ),
    dcc.Graph(figure=bar_fig)
])

# Callback для обновления Cytoscape-элементов согласно выбранным группам
@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('group-filter', 'value')
)
def update_cytoscape(selected_groups):
    if not selected_groups or 'all' in selected_groups:
        # Если выбран вариант "Все", объединяем элементы всех групп
        all_el = []
        for grp in group_elements.values():
            all_el.extend(grp)
        return all_el
    else:
        all_el = []
        for grp in selected_groups:
            all_el.extend(group_elements.get(grp, []))
        return all_el

# Callback для обновления DataTable при выборе узлов в Cytoscape
@app.callback(
    Output('data-table', 'data'),
    Input('cytoscape-graph', 'selectedNodeData')
)
def update_table(selected_nodes):
    if not selected_nodes:
        return table_data
    else:
        # Собираем индексы из выбранных узлов для всех групп, объединяем их
        all_rows = []
        # Здесь для простоты ищем по каждому узлу в общем графе.
        # Если у вас нужны отдельные обработки для каждой группы – можно усложнить логику.
        for node in selected_nodes:
            # Пробуем найти node в общем наборе G['nodes'] (если его нет, пропускаем)
            # Заметим, что при раздельном построении графов G может отсутствовать информация об узлах.
            # В этом случае можно заранее сохранить привязки индексов для каждой группы.
            # Здесь для примера оставляем метод, аналогичный предыдущему варианту.
            node_id = node['id']
            # Предположим, что для всех групп имеются индексы в столбце nodes из одного общего G.
            # Если нет, то можно сохранить отдельно m1, m0, f1, f0 и искать там.
            rows_list = []
            # Пробуем искать сначала в группе "male_target1"
            for grp in group_elements:
                # Если наш узел найден в общей выборке, можно добавить его индексы.
                # Здесь приведена упрощённая логика – на практике нужно сохранить маппинг node_id -> индексы.
                rows_list.extend([])  # Поставьте здесь логику получения индексов для узла
            all_rows.extend(rows_list)
        unique_rows = list(set(all_rows))
        # Если уникальных индексов получено, возвращаем соответствующие строки из df
        if unique_rows:
            return df.iloc[unique_rows].to_dict('records')
        else:
            return table_data
        

app.run_server(debug=True, port=8061)