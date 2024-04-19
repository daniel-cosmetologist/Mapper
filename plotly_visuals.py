import dash
from dash import html
import dash_cytoscape as cyto
import csv
import networkx as nx

# Инициализация Dash приложения
app = dash.Dash(__name__)

# Загрузка данных из CSV и построение графа NetworkX
G_nx = nx.Graph()
with open('mapper_graph_adjacency_list.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader, None)  # Пропускаем заголовок если он есть
    for row in reader:
        G_nx.add_edge(row[0], row[1])

# Генерация данных для Cytoscape
elements = [
    {'data': {'id': node, 'label': node}} for node in G_nx.nodes()
] + [
    {'data': {'source': edge[0], 'target': edge[1]}} for edge in G_nx.edges()
]

# Определение стилей узлов и рёбер
default_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': 'gray',
            'label': 'data(label)',
            'font-size': '2px',
            'width': '2px',
            'height': '2px'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'line-color': 'light-gray',
            'width': 0.1
        }
    }
]

# Добавление Cytoscape компонента на страницу
app.layout = html.Div([
    cyto.Cytoscape(
    id='cytoscape-graph',
    elements=elements,
    stylesheet=default_stylesheet,
    layout={'name': 'cose'},
    style={'width': '100%', 'height': '900px'},
    )
])

# Expected one of ["random","preset","circle","concentric","grid","breadthfirst","cose","cose-bilkent",
#                  "fcose","cola","euler","spread","dagre","klay"].

if __name__ == '__main__':
    app.run_server(debug=True, port=8051)
