import pandas as pd
import ipywidgets as widgets
from IPython.display import display, clear_output

# Загружаем данные из nodes_to_rows.csv
nodes_data_path = 'nodes_to_rows.csv'
nodes_data = pd.read_csv(nodes_data_path)

# Загружаем исходный датасет
dataset_path = 'df_median (1).csv'
original_dataset = pd.read_csv(dataset_path)

# Функция для отображения строк по выбранному node_id
def show_rows(node_id):
    # Находим строки, соответствующие выбранному node_id
    rows_str = nodes_data.loc[nodes_data['node_id'] == node_id, 'row_indices'].values[0]
    # Преобразуем строку индексов в список индексов
    rows_list = list(map(int, rows_str.split(',')))
    # Выбираем строки из исходного датасета
    selected_rows = original_dataset.iloc[rows_list]
    # Очищаем предыдущий вывод
    clear_output(wait=True)
    # Отображаем данные
    display(node_id_selector)  # Повторно отображаем выбор node_id
    display(selected_rows)

# Создаем выпадающий список с node_id
node_id_selector = widgets.Dropdown(
    options=nodes_data['node_id'].unique(),
    description='Node ID:',
    disabled=False,
)

# Создаем интерактивный виджет
interactive_widget = widgets.interactive(show_rows, node_id=node_id_selector)

# Отображаем виджет
display(interactive_widget)
