import numpy as np
import kmapper as km
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
# Here, we use t-SNE for dimensionality reduction, but you can use any technique (e.g., PCA, MDS)
projected_data = mapper.fit_transform(X, projection=TSNE())

# Create dictionary called 'graph' with nodes, edges and meta-information
graph = mapper.map(projected_data, X, clusterer=DBSCAN(eps=0.3, min_samples=10))

# Visualize it
mapper.visualize(graph, path_html="mapper_visualization_output.html",
                 title="Mapper on dataset")
