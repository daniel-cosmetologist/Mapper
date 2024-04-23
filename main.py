import pandas as pd
import networkx as nx
import csv
from dash import html, dcc, Dash, Input, Output, dash_table
import dash_cytoscape as cyto
import math

# Загрузка данных
nodes_data_path = 'nodes_to_rows.csv'
nodes_data = pd.read_csv(nodes_data_path)

dataset_path = 'db_nl_preprocessed.csv'
original_dataset = pd.read_csv(dataset_path)

G_nx = nx.Graph()
with open('mapper_graph_adjacency_list.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader, None)  # Пропустить заголовок
    for row in reader:
        G_nx.add_edge(row[0], row[1])

# Расчет логарифмического размера каждой узлы на основе данных
node_sizes = {}
for index, row in nodes_data.iterrows():
    node_id = row['node_id']
    row_indices = row['row_indices']
    count = len(set(map(int, row_indices.split(','))))
    node_sizes[node_id] = count

# Минимальный и максимальный размеры узлов
min_size = 2  # Минимальный размер узла
max_size = 9  # Максимальный размер узла

# Находим минимальное и максимальное значение для корректного масштабирования
min_count = min(node_sizes.values()) if node_sizes else 1
max_count = max(node_sizes.values()) if node_sizes else 1

# Нормализация размеров с использованием логарифмической шкалы
normalized_sizes = {}
for node, count in node_sizes.items():
    if count > 1:
        # Применяем логарифмическое масштабирование
        log_size = math.log(count, max_count)  # База логарифма — максимальное значение count
        normalized_size = min_size + (max_size - min_size) * (log_size / math.log(max_count, max_count))
    else:
        normalized_size = min_size
    normalized_sizes[node] = normalized_size

# Обновляем стиль узлов
elements = [
    {'data': {'id': node, 'label': node}, 'style': {'width': normalized_sizes[node], 'height': normalized_sizes[node]}}
    for node in G_nx.nodes()
] + [
    {'data': {'source': edge[0], 'target': edge[1]}}
    for edge in G_nx.edges()
]

stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': 'gray',
            'label': 'data(label)',
            'font-size': '1px',
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

# Функция для создания условных стилей для DataTable
def generate_style_data_conditional(df, thresholds):
    styles = []
    for column, bounds in thresholds.items():
        # Ниже нормы (светло-синий)
        styles.append({
            'if': {
                'filter_query': f'{{{column}}} < {bounds["low"]}',
                'column_id': column
            },
            'backgroundColor': '#add8e6',
            'color': 'black'
        })
        # В норме (светло-зелёный)
        styles.append({
            'if': {
                'filter_query': f'{{{column}}} >= {bounds["low"]} && {{{column}}} <= {bounds["high"]}',
                'column_id': column
            },
            'backgroundColor': '#90ee90',
            'color': 'black'
        })
        # Выше нормы (светло-красный)
        styles.append({
            'if': {
                'filter_query': f'{{{column}}} > {bounds["high"]}',
                'column_id': column
            },
            'backgroundColor': '#ffcccb',
            'color': 'black'
        })
        # Пустой
        styles.append({
            'if': {
                'filter_query': f'{{{column}}} is blank',
                'column_id': column
            },
            'backgroundColor': '#ffffff',
            'color': 'black'
        })
    return styles

thresholds = {
    # 'возраст': {'low': 18, 'high': 65},
    # 'пол': {'low': 0, 'high': 1}, # 0 для женщин, 1 для мужчин
    'масса_тела': {'low': 50, 'high': 90},
    'рост': {'low': 150, 'high': 200},
    'от': {'low': 70, 'high': 99},  # Охват талии
    'об': {'low': 80, 'high': 120}, # Охват бедер
    'оп': {'low': 18.5, 'high': 24.9},  # Индекс массы тела
    'жм': {'low': 10, 'high': 200},  # Не уверен, что это за показатель
    'мм': {'low': 10, 'high': 200},  # Тоже неясно, что за показатель
    'ож': {'low': 10, 'high': 200},  # И этот показатель также неизвестен
    'внутриклеточная_жидкость': {'low': 20, 'high': 40}, # В литрах
    'тощая_мт': {'low': 0, 'high': 100},  # Тоже неясно, что за показатель
    'акм': {'low': 0, 'high': 100},  # Активная клеточная масса?
    'холестерин': {'low': 3.6, 'high': 5.2},
    'нас._жир': {'low': 0.2, 'high': 0.5},
    'натрий_na': {'low': 135, 'high': 145},
    'сахара': {'low': 3.3, 'high': 5.5},
    'энергия': {'low': 1000, 'high': 2500},
    'вода': {'low': 1.5, 'high': 2.5},  # В литрах
    'пищ._волокна': {'low': 25, 'high': 38},
    'мононенас._жир': {'low': 0, 'high': 2},  # Процент от общего потребления жира
    'белки': {'low': 0.8, 'high': 1.5},  # Грамм на кг массы тела
    'аргинин': {'low': 2, 'high': 6},  # В граммах, дозировки варьируются
    'валин': {'low': 2, 'high': 6},  # Аминокислоты
    'зола': {'low': 1, 'high': 5},  # В контексте пищевых продуктов может быть различной
    'полиненас._жир': {'low': 0, 'high': 2},  # Процент от общего потребления жира
    'калий_k': {'low': 3.5, 'high': 5.0},
    'крахмал': {'low': 26, 'high': 130},  # В граммах в день
    'гистидин': {'low': 1, 'high': 4},  # Аминокислоты
    'жиры': {'low': 20, 'high': 35},  # Процент от общего потребления калорий
    'кальций_ca': {'low': 800, 'high': 1300},
    'алкоголь': {'low': 0, 'high': 2},  # Дозировки в стандартных напитках
    'кремний_si': {'low': 5, 'high': 20},  # В мг, дозировки варьируются
    'сера_s': {'low': 800, 'high': 900},
    'изолейцин': {'low': 2, 'high': 6},  # Аминокислоты
    'углеводы': {'low': 130, 'high': 300},
    'олеиновая_кислота': {'low': 0, 'high': 2},  # Процент от общего потребления жира
    'магний_mg': {'low': 310, 'high': 420},
    'фруктоза': {'low': 15, 'high': 40},  # В граммах в день
    # Витамины и микроэлементы
    'a': {'low': 700, 'high': 900},  # Мкг
    'b1_тиамин': {'low': 1.1, 'high': 1.5},
    'b2_рибофлав.': {'low': 1.1, 'high': 1.5},
    'b5_пантотен._кис.': {'low': 5, 'high': 10},
    'b6_пиридоксин': {'low': 1.3, 'high': 2.0},
    'b9_фолаты': {'low': 200, 'high': 500},  # Мкг
    'b12_кобаламин': {'low': 2.4, 'high': 3.0},  # Мкг
    'b4_холин': {'low': 425, 'high': 550},
    'β-каротин': {'low': 600, 'high': 700},  # Мкг
    'ретин._эквив.': {'low': 800, 'high': 1000},  # Мкг
    # Аминокислоты
    'лейцин': {'low': 2, 'high': 6},
    'фосфор_p': {'low': 700, 'high': 1000},
    # Омега жирные кислоты
    'омега-3': {'low': 1.1, 'high': 1.6},
    'лактоза': {'low': 0, 'high': 50},  # Некоторые люди имеют непереносимость
    'метионин': {'low': 2, 'high': 6},
    # Минералы
    'железо_fe': {'low': 8, 'high': 18},
    'омега-6': {'low': 5, 'high': 10},
    'галактоза': {'low': 0, 'high': 50},  # Также как и лактоза
    'метионин+цистеин': {'low': 4, 'high': 8},  # Сумма двух аминокислот
    'c': {'low': 75, 'high': 90},
    'треонин': {'low': 2, 'high': 6},
    'd': {'low': 5, 'high': 15},  # В мкг
    'триптофан': {'low': 1, 'high': 1.5},
    'e_α-токоферол': {'low': 15, 'high': 19},
    'гамма-линолен._к-та': {'low': 0.5, 'high': 3},
    'фенилаланин': {'low': 2, 'high': 6},
    'h_биотин': {'low': 30, 'high': 100},  # В мкг
    'бор_b': {'low': 0.5, 'high': 1.5},
    'фенилаланин+тирозин': {'low': 4, 'high': 8},
    'pp_ниацин': {'low': 16, 'high': 20},
    # Микроэлементы
    'ванадий_v': {'low': 0, 'high': 1.8},  # Мкг
    'йод_i': {'low': 150, 'high': 300},  # Мкг
    'кобальт_co': {'low': 2, 'high': 10},  # Мкг
    'марганец_mn': {'low': 2.3, 'high': 4.5},
    'медь_cu': {'low': 900, 'high': 1100},  # Мкг
    'молибден_mo': {'low': 45, 'high': 90},  # Мкг
    'никель_ni': {'low': 0, 'high': 100},  # Мкг
    'селен_se': {'low': 55, 'high': 70},
    'фтор_f': {'low': 3, 'high': 4},
    'хром_cr': {'low': 35, 'high': 50},
    'цинк_zn': {'low': 11, 'high': 13},
    # Специфические для некоторых профессий и спорта
    # 'профессия_работники_преимущественно_умственного_труда': {'low': 0, 'high': 1},  # Бинарные переменные, для примера
    # 'профессия_работники_занятые_легким_физическим_трудом': {'low': 0, 'high': 1},
    # 'спорт_легкий_спорт': {'low': 0, 'high': 1},
    # 'спорт_не_занимаюсь': {'low': 0, 'high': 1},
    'bmi': {'low': 18.5, 'high': 24.9},  # Индекс массы тела
}



conditional_styles = generate_style_data_conditional(original_dataset, thresholds)


app = Dash(__name__)
server = app.server
app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        stylesheet=stylesheet,
        layout={'name': 'cose'},
        # Expected one of ["random","preset","circle","concentric","grid",
        # "breadthfirst","cose","cose-bilkent","fcose","cola","euler","spread","dagre","klay"].
        style={'width': '100%', 'height': '900px'},
        boxSelectionEnabled=True
    ),
    html.Div(id='selected-node-data', style={'padding-top': '20px'}),
    # Место для вывода таблицы после выбора узлов
])

@app.callback(
    Output('selected-node-data', 'children'),
    [Input('cytoscape-graph', 'selectedNodeData')]
)
def update_output(data):
    if not data:
        return "No nodes selected"
    else:
        node_ids = [node['id'] for node in data]
        all_rows_list = []
        for node_id in node_ids:
            rows_str = nodes_data.loc[nodes_data['node_id'] == node_id, 'row_indices'].values[0]
            rows_list = list(map(int, rows_str.split(',')))
            all_rows_list.extend(rows_list)

        unique_rows_list = list(set(all_rows_list))
        selected_rows = original_dataset.iloc[unique_rows_list]

        return dash_table.DataTable(
            data=selected_rows.to_dict('records'),
            columns=[{"name": i, "id": i} for i in selected_rows.columns],
            page_size=100,  # Pagination
            style_table={'height': '1000px', 'overflowY': 'auto'},
            fixed_rows={'headers': True, 'data': 0},
            # fixed_columns={'headers': True, 'data': 1},
            style_data_conditional=conditional_styles,  # Условные стили, сгенерированные функцией
        )

if __name__ == '__main__':
    app.run_server(debug=False, port=8050)