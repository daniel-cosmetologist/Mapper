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

app = Dash(__name__)
app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        stylesheet=stylesheet,
        layout={'name': 'cose'},
        style={'width': '100%', 'height': '900px'},
        boxSelectionEnabled=True
    ),
    html.Div(id='selected-node-data', style={'padding-top': '20px'})
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
            page_size=10,  # Pagination
            style_table={'height': '400px', 'overflowY': 'auto'},
            fixed_rows={'headers': True, 'data': 0}
        )

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
