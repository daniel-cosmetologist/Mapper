import dash
from dash import html, dcc, Output, Input, callback, dash_table
import dash_cytoscape as cyto
import pandas as pd
import networkx as nx
import csv

# Load and prepare your data
nodes_data_path = 'nodes_to_rows.csv'
nodes_data = pd.read_csv(nodes_data_path)

dataset_path = 'db_nl_preprocessed.csv'
original_dataset = pd.read_csv(dataset_path)

G_nx = nx.Graph()
with open('mapper_graph_adjacency_list.csv', 'r') as file:
    reader = csv.reader(file)
    next(reader, None)  # Skip the header if present
    for row in reader:
        G_nx.add_edge(row[0], row[1])

# Prepare elements for the Cytoscape graph
elements = [
    {'data': {'id': node, 'label': node}} for node in G_nx.nodes()
] + [
    {'data': {'source': edge[0], 'target': edge[1]}} for edge in G_nx.edges()
]

stylesheet = [
    {
        'selector': 'node',
        'style': {
            'background-color': 'gray',
            'label': 'data(label)',
            'font-size': '1px',
            'width': '3px',
            'height': '3px'
        }
    },
    {
        'selector': 'node:selected',
        'style': {
            'background-color': 'red'
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

app = dash.Dash(__name__)
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
            page_size=100,  # Add pagination
            style_table={
                'overflowX': 'auto',
                'overflowY': 'auto',
                'height': '400px',
                'minWidth': '100%',
            },
            fixed_rows={'headers': True, 'data': 0},  # This property fixes the header row
            style_cell={
                'height': 'auto',
                'minWidth': '10px', 'width': '30px', 'maxWidth': '80px',
                'whiteSpace': 'normal',
                'textAlign': 'left'
            },
            style_header={
                'backgroundColor': 'rgb(210, 210, 210)',
                'fontWeight': 'bold',
                'position': 'sticky',  
                'top': 0,
                'zIndex': 2  
            },
            style_data_conditional=[
                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgb(248, 248, 248)'},
                {'if': {'row_index': 'even'}, 'backgroundColor': 'rgb(230, 230, 230)'}
            ]
        )


if __name__ == '__main__':
    app.run_server(debug=True, port=8053)
