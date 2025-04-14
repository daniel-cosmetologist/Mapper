import math
import numpy as np
import pandas as pd
import kmapper as km
from sklearn import ensemble, cluster
import dash
from dash import dcc, html, Dash, Input, Output, dash_table
import plotly.graph_objects as go
import dash_cytoscape as cyto

############################
# 1. Условные стили DataTable – форматирование по 5-му и 95-му процентилю
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
# 2. Функция построения единого Mapper-графа (строится один раз)
############################

def build_single_mapper(df, features, cover_params, n_clusters=3):
    X = np.array(df[features])
    projector = ensemble.IsolationForest(random_state=0, n_jobs=-1)
    projector.fit(X)
    lens1 = projector.decision_function(X)
    
    mapper = km.KeplerMapper(verbose=2)
    lens2 = mapper.fit_transform(X, projection="knn_distance_5")
    lens = np.c_[lens1, lens2]
    
    cover = km.Cover(n_cubes=cover_params.get("n_cubes", 10),
                     perc_overlap=cover_params.get("perc_overlap", 0.5))
    G = mapper.map(lens, X, cover=cover,
                   clusterer=cluster.AgglomerativeClustering(n_clusters=n_clusters))
    
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

    # Элементы узлов и рёбер для Cytoscape
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
    
    # Словарь с информацией по узлам: список индексов и количества по группам
    node_data = {}
    for nid, members in G['nodes'].items():
        node_data[nid] = {
            'members': members,
            'size': normalized_sizes[nid],
            'male_t1_count': 0,
            'male_t0_count': 0,
            'female_t1_count': 0,
            'female_t0_count': 0
        }
    for nid, info in node_data.items():
        subset = df.iloc[info['members']]
        info['male_t1_count']   = subset[(subset['пол'] == 1) & (subset['target'] == 1)].shape[0]
        info['male_t0_count']   = subset[(subset['пол'] == 1) & (subset['target'] == 0)].shape[0]
        info['female_t1_count'] = subset[(subset['пол'] == 0) & (subset['target'] == 1)].shape[0]
        info['female_t0_count'] = subset[(subset['пол'] == 0) & (subset['target'] == 0)].shape[0]
    
    return G, elements, node_data

############################
# 3. Загрузка данных и построение Mapper-графа
############################

dataset_path = '../datasets/db_nl_preprocessed-edit.csv'
df = pd.read_csv(dataset_path).fillna(0)
df['target'] = df['target'].astype(int)
# Используем все столбцы – можно задать и иной набор
features = df.columns.tolist()

cover_params = {"n_cubes": 20, "perc_overlap": 0.7}
G, base_elements, node_data = build_single_mapper(df, features, cover_params, n_clusters=3)
global_node_mapping = node_data.copy()

############################
# 4. Фиксированные цвета для 4 групп
############################

# Фиксированные цвета:
FIXED_COLORS = {
    'm1': "#00008B",   # мужчины, target=1 – тёмно-синий
    'm0': "#00FFFF",   # мужчины, target=0 – cyan
    'f1': "#8B0000",   # женщины, target=1 – тёмно-красный
    'f0': "#FFB6C1",   # женщины, target=0 – розовый (lightpink)
}

# Дополнительные варианты для фильтрации по полу целиком:
# для объединённых групп используем другой набор (если потребуется)
FIXED_COLORS_COMBINED = {
    'men': "#87CEFA",    # светло-синий для всех мужчин
    'women': "#FFC0CB",  # светло-розовый для всех женщин
}

# Опции для фильтрации – здесь реализуем 6 вариантов:
group_options = [
    {'label': 'Мужчины (все)', 'value': 'men'},
    {'label': 'Женщины (все)', 'value': 'women'},
    {'label': 'Мужчины, target=1', 'value': 'm1'},
    {'label': 'Мужчины, target=0', 'value': 'm0'},
    {'label': 'Женщины, target=1', 'value': 'f1'},
    {'label': 'Женщины, target=0', 'value': 'f0'},
]

############################
# 5. Столбцы для DataTable: базовые и полные
############################

basic_cols = [c for c in ["возраст", "рост", "масса_тела", "bmi", "пол", "target"] if c in df.columns]
table_columns_basic = [{"name": c, "id": c} for c in basic_cols]
table_columns_full = [{"name": c, "id": c} for c in df.columns]

style_data_conditional = generate_style_data_conditional_percentiles(df)

############################
# 6. Интерфейс Dash
############################

app = dash.Dash(__name__)
server = app.server

cytoscape_stylesheet = [
    {'selector': 'node',         'style': {'label': 'data(label)', 'font-size': '1px'}},
    {'selector': 'node:selected','style': {'background-color': 'red'}},
    {'selector': 'edge',         'style': {'line-color': 'light-gray', 'width': 0.1}}
]

bar_fig = go.Figure(data=go.Bar(x=["Всего"], y=[len(df)]))
bar_fig.update_layout(title='Общее количество строк в датасете')

app.layout = html.Div([
    html.H1("Интерактивный дашборд для анализа Mapper-графа"),
    
    html.Div([
        html.Label("Выберите фильтр:"),
        dcc.Dropdown(
            id='filter-dropdown',
            options=group_options,
            multi=True,
            value=[]  # по умолчанию пусто – узлы серые
        ),
    ], style={'width': '50%', 'padding': '10px'}),
    
    # Кнопка для переключения столбцов в таблице
    html.Div([
        html.Button("Показать всю таблицу", id="toggle-columns", n_clicks=0)
    ], style={'padding': '10px'}),
    
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=base_elements,   # граф строится один раз
        stylesheet=cytoscape_stylesheet,
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '800px'}
    ),
    
    html.H2("Данные выбранных узлов"),
    dash_table.DataTable(
        id='data-table',
        data=[],  # данные обновляются callback’ом
        columns=table_columns_basic,
        page_size=100,
        style_table={'height': '500px', 'overflowY': 'auto'},
        fixed_rows={'headers': True},
        style_data_conditional=style_data_conditional
    ),
    html.Div(id='table-summary', style={'padding': '10px', 'font-weight': 'bold'}),
    dcc.Graph(figure=bar_fig)
])

############################
# 7. Callback для изменения цвета узлов
############################
# При выборе фильтра мы меняем только цвета узлов – при отсутствии фильтра узлы серые.
# Если выбран фильтр, то для каждого узла определяется, к какой группе (или группам) он принадлежит.
# Если выбран фильтр "men" или "women", мы используем соответствующий комбинированный вариант,
# иначе – для конкретных групп m1, m0, f1, f0 берутся фиксированные цвета.
# Если выбрано несколько значений, то узел окрашивается по группе, имеющей наибольшее число пациентов,
# учитывая только те группы, которые удовлетворяют фильтру.

@app.callback(
    Output('cytoscape-graph', 'elements'),
    Input('filter-dropdown', 'value')
)
def update_cytoscape(selected_filters):
    new_elems = []
    # Если фильтр не выбран, все узлы серые:
    if not selected_filters:
        for nid, info in node_data.items():
            new_elems.append({
                'data': {'id': nid, 'label': nid},
                'style': {
                    'width': info['size'],
                    'height': info['size'],
                    'background-color': 'gray'
                }
            })
    else:
        # Для каждого узла вычисляем, к каким подгруппам он принадлежит:
        # Для каждой строки узла:
        #   если пол==1 и target==1 -> группа "m1"
        #   если пол==1 и target==0 -> "m0"
        #   если пол==0 и target==1 -> "f1"
        #   если пол==0 и target==0 -> "f0"
        # Также для объединённых фильтров "men" и "women":
        #   если пол==1 -> группа "men"
        #   если пол==0 -> группа "women"
        for nid, info in node_data.items():
            m1c = info['male_t1_count']
            m0c = info['male_t0_count']
            f1c = info['female_t1_count']
            f0c = info['female_t0_count']
            
            # Собираем группы, которым узел принадлежит
            node_groups = set()
            if m1c > 0:
                node_groups.add('m1')
                node_groups.add('men')
            if m0c > 0:
                node_groups.add('m0')
                node_groups.add('men')
            if f1c > 0:
                node_groups.add('f1')
                node_groups.add('women')
            if f0c > 0:
                node_groups.add('f0')
                node_groups.add('women')
            # Выбираем только те группы, которые входят в фильтр
            valid_groups = node_groups.intersection(selected_filters)
            # Если фильтр выбран как объединённый (men или women) и узел содержит пациентов данного пола,
            # то используем их; иначе, если выбран конкретный (m1, m0, f1, f0), то работаем с ним.
            if not valid_groups:
                # Если узел не удовлетворяет выбранному фильтру, его всё равно окрашиваем,
                # потому что граф не пересоздаётся – здесь можно выбрать нейтральный цвет, но по требованию узлы не должны быть серыми.
                # Поэтому для узла, который не проходит фильтр, оставляем исходный цвет (например, серый)
                chosen_color = 'gray'
            else:
                # Если в фильтре присутствует объединённый вариант, приоритет отдадим ему:
                if 'men' in valid_groups:
                    # Используем суммарное значение по мужчинам:
                    count_val = m1c + m0c
                    # Определяем, какой target преобладает:
                    chosen_color = FIXED_COLORS['m1'] if m1c >= m0c else FIXED_COLORS['m0']
                elif 'women' in valid_groups:
                    count_val = f1c + f0c
                    chosen_color = FIXED_COLORS['f1'] if f1c >= f0c else FIXED_COLORS['f0']
                else:
                    # Если выбраны конкретные группы (m1, m0, f1, f0) – выбираем ту, у которой максимум
                    counts = {grp: 0 for grp in ['m1','m0','f1','f0']}
                    counts['m1'] = m1c
                    counts['m0'] = m0c
                    counts['f1'] = f1c
                    counts['f0'] = f0c
                    # Ограничим словарь только выбранными группами:
                    filtered_counts = {grp: cnt for grp, cnt in counts.items() if grp in valid_groups}
                    # Если filtered_counts пуст (что не должно быть), оставляем gray
                    if filtered_counts:
                        dominant = max(filtered_counts, key=filtered_counts.get)
                        chosen_color = FIXED_COLORS[dominant]
                    else:
                        chosen_color = 'gray'
            new_elems.append({
                'data': {'id': nid, 'label': nid},
                'style': {
                    'width': info['size'],
                    'height': info['size'],
                    'background-color': chosen_color
                }
            })
    # Добавляем все рёбра (иначе граф будет неполным)
    shown_ids = {el['data']['id'] for el in new_elems}
    for edge in base_elements:
        d = edge.get('data', {})
        if 'source' in d and 'target' in d:
            if d['source'] in shown_ids and d['target'] in shown_ids:
                new_elems.append(edge)
    return new_elems

############################
# 8. Callback для обновления DataTable и сводки по выбранным узлам,
#     с фильтрацией по выбранному фильтру.
############################

def row_groups(row):
    """Возвращает множество групп, которым принадлежит пациент"""
    groups = set()
    if row['пол'] == 1:
        groups.add('men')
        if row['target'] == 1:
            groups.add('m1')
        else:
            groups.add('m0')
    elif row['пол'] == 0:
        groups.add('women')
        if row['target'] == 1:
            groups.add('f1')
        else:
            groups.add('f0')
    return groups

@app.callback(
    [Output('data-table', 'data'),
     Output('table-summary', 'children')],
    [Input('cytoscape-graph', 'selectedNodeData'),
     Input('filter-dropdown', 'value')]
)
def update_table(selected_nodes, selected_filters):
    if not selected_nodes or len(selected_nodes) == 0:
        return [], "Узлы не выбраны"
    
    # Объединяем индексы из выбранных узлов
    all_indices = []
    for nd in selected_nodes:
        nid = nd['id']
        info = global_node_mapping.get(nid, {})
        members = info.get('members', [])
        all_indices.extend(members)
    unique_indices = list(set(all_indices))
    if not unique_indices:
        return [], "Нет строк"
    
    # Извлекаем соответствующие строки и применяем фильтрацию по выбранным группам
    subset_df = df.iloc[unique_indices].copy()
    if selected_filters:
        # Для каждой строки определяем группы
        subset_df['row_groups'] = subset_df.apply(row_groups, axis=1)
        # Оставляем строку, если хотя бы одно выбранное значение входит в row_groups
        mask = subset_df['row_groups'].apply(lambda gr: bool(gr.intersection(set(selected_filters))))
        subset_df = subset_df[mask]
        # Убираем временный столбец
        subset_df.drop(columns=['row_groups'], inplace=True)
    
    total = len(subset_df)
    males = subset_df[subset_df['пол'] == 1].shape[0]
    females = subset_df[subset_df['пол'] == 0].shape[0]
    t1 = subset_df[subset_df['target'] == 1].shape[0]
    summary_text = (
        f"Выбрано узлов: {len(selected_nodes)}, пациентов: {total}. "
        f"Мужчин: {males}, Женщин: {females}, Target=1: {t1}"
    )
    return subset_df.to_dict('records'), summary_text

############################
# 9. Callback для переключения набора столбцов DataTable
############################

@app.callback(
    Output('data-table', 'columns'),
    Input('toggle-columns', 'n_clicks')
)
def toggle_columns(n_clicks):
    if n_clicks % 2 == 1:
        return table_columns_full
    else:
        return table_columns_basic

############################
# 10. Запуск приложения
############################

if __name__ == '__main__':
    app.run_server(debug=True, port=8061)
