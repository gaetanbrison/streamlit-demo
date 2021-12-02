# Presentation Scikit-Network
## The goal is to display and let user play with some examples in scikit-network


#I) Import packages

import streamlit as st
import numpy as np

from sknetwork.embedding import Spring
from sknetwork.utils import KNNDense, CNNDense
from sknetwork.visualization import svg_graph
from sklearn.metrics import accuracy_score


from sknetwork.data import karate_club, painters, movie_actor
from sknetwork.classification import KNN
from sknetwork.embedding import GSVD
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph


#II) Start coding

st.image('scikit-network.jpeg', width=200)
st.title('Gaëtan est trop fort')

st.write("""
### Explore different clustering models and datasets
""")

dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Painters','Other')
)

st.write(f"Dataset: {dataset_name}")

classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('KNNDense', 'SVM', 'Random Forest')
)


painter_name = st.sidebar.multiselect(
    'Select 3 painters',
    ('Pablo Picasso', 'Claude Monet', 'Michel Angelo', 'Edouard Manet',
       'Peter Paul Rubens', 'Rembrandt', 'Gustav Klimt', 'Edgar Degas',
       'Vincent van Gogh', 'Leonardo da Vinci', 'Henri Matisse',
       'Paul Cezanne', 'Pierre-Auguste Renoir', 'Egon Schiele')
)



def get_dataset(name):
    data = None
    if name == 'Painters':
        graph = painters(metadata=True)
        adjacency = graph.adjacency
        position = graph.position
        names = graph.names
    return adjacency, position, names


adjacency = get_dataset(dataset_name)[0]
position = get_dataset(dataset_name)[1]
names = get_dataset(dataset_name)[2]
#seeds = get_dataset(dataset_name)[3]

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == 'SVM':
        C = st.sidebar.slider('C', 0.01, 10.0)
        params['C'] = C
    elif clf_name == 'KNNDense':
        K = st.sidebar.slider('Select Optimal: K', 1, 15)
        params['K'] = K
    else:
        max_depth = st.sidebar.slider('max_depth', 2, 15)
        params['max_depth'] = max_depth
        n_estimators = st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators'] = n_estimators
    return params

def tune_model(painter_name):
    if len(painter_name)>2:
        graph = painters(metadata=True)
        painter1 = graph["names"].tolist().index(painter_name[0])
        painter2 = graph["names"].tolist().index(painter_name[1])
        painter3 = graph["names"].tolist().index(painter_name[2])
    else:
        painter1 = 5
        painter2 = 6
        painter3 = 11
    return painter1, painter2, painter3




params = add_parameter_ui(classifier_name)
st.write(f'Classifier Selected= {classifier_name}')
st.write(f'Painters Selected= {painter_name}')
st.write(f'Parameter Selected = {params["K"]}')
st.write(f'Names in the Dataset= {list(get_dataset(dataset_name)[2])}')
#st.write(f'Names = {names}')



rembrandt = tune_model(painter_name)[0]
klimt = tune_model(painter_name)[1]
cezanne = tune_model(painter_name)[2]
seeds = {cezanne: 0, rembrandt: 1, klimt: 2}
knn = KNN(GSVD(3), n_neighbors=params['K'])
labels = knn.fit_transform(adjacency, seeds)
membership = knn.membership_
scores = membership[:,0].toarray().ravel()
image = svg_digraph(adjacency, position, names, scores=scores, seeds=seeds)
st.image(image, use_column_width = True)

### END

st.write("For more information: [documentation scikit-network](https://scikit-network.readthedocs.io/en/latest/)")
