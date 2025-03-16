import sklearn
import kaleido
import plotly.express as px
from utils import read_dataframe
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

CLUSTER_COLUMNS = ['Petal_width', 'Petal_length']


def run_kmeans_and_visualization(df_iris):
    print(f'[K-means] Use {CLUSTER_COLUMNS} with K=3')
    # Part A - Run K-means with K=3 using Petal_width and Petal_length columns

    scaler = sklearn.preprocessing.StandardScaler()
    normalized_iris = scaler.fit_transform(df_iris[CLUSTER_COLUMNS])
    model_cluster = sklearn.cluster.KMeans(
        n_clusters=3, n_init='auto', random_state=5963)
    model_cluster.fit(normalized_iris)

    # Part A - Create a scatter plot to visualize your clusters found

    df_iris_clustered = df_iris.copy()
    # Change label from 0, 1,... to '#1', '#2', ...
    df_iris_clustered['cluster'] = model_cluster.labels_
    df_iris_clustered['cluster'] = df_iris_clustered['cluster'].map(lambda v: f'#{v + 1}')
    fig = px.scatter(df_iris_clustered, title='[KMeans] Petal width and Petal length of Iris Species(K=3)',
                     x='Petal_width', y='Petal_length', color='cluster')
    fig.show()

    # Part B - Create a scatter plot to visualize the actual Species_name

    fig_species = px.scatter(df_iris, title='[Actual] Petal width and length of Iris Species',
                             x='Petal_length', y='Petal_width', color='Species_name')
    fig_species.show()

    pass


if __name__ == '__main__':
    run_kmeans_and_visualization(read_dataframe())