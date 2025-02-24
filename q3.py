import sklearn
import kaleido
import plotly.express as px
from utils import read_dataframe

CLUSTER_COLUMNS = ['Petal_width', 'Petal_length']

def run_kmeans_and_visualization(df_iris):
    print(f'[K-means] Use {CLUSTER_COLUMNS} with K=3')
    # Part A - Run K-means with K=3 using Petal_width and Petal_length columns

    # Part A - Create a scatter plot to visualize your clusters found

    # Part B - Create a scatter plot to visualize the actual Species_name

    pass

if __name__ == '__main__':
    run_kmeans_and_visualization(read_dataframe())