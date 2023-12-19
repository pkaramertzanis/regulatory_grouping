# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
import math
from pathlib import Path
import re
import random
import math

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

from typing import List
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat

from cheminfo_toolkit import Molecule

def visualise_network(molecules: List[Molecule], impath: Path, number_closest_neighbours:int=4, minimum_group_size:int=20)-> None:
    '''
    Visualise the group structures using the 2D QSAR molecular structures.

    :param molecules: List of molecules
    :param impath: Path to export the figure as png
    :param number_closest_neighbours: Number of closest neighbours to include as edges
    :param minimum_group_size: Minimum number of substances to include group in the graph
    :return:
    '''

    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')


    groups = [molecule.group for molecule in molecules]

    # select the groups for which we have a sufficient number of processable structures
    counts = Counter(groups)
    selected_groups = [group for group in counts if counts[group]>=minimum_group_size]
    log.info(f'{len(selected_groups)} groups with a minimum of {minimum_group_size} structures will be included in the network visualisation')


    # prepare the data
    mols = [mol.rdkit_mol for mol in molecules if mol.group in selected_groups]
    group_names = [mol.group for mol in molecules if mol.group in selected_groups]
    # compute default fingerprints
    fingerprints = [mol.compute_fingerprint() for mol in molecules if mol.group in selected_groups]
    data = pd.DataFrame({'mol': mols, 'group': group_names, 'fingerprint': fingerprints})
    data = data.sort_values(by=['group'])
    data = data.reset_index(drop=True).reset_index().rename({'index': 'entry ID'}, axis='columns')



    # assign random colours to the group and colour the substances
    def random_color():
        r = random.randint(0, 255)/255.
        g = random.randint(0, 255)/255.
        b = random.randint(0, 255)/255.
        return (r, g, b)
    arn_group_colours = {group_name: random_color() for group_name in selected_groups}
    log.info(f'groups have been assigned random colours')
    data['colour'] = data['group'].map(arn_group_colours)


    # compute the distance matrix
    distance_matrix_lower_triang = GetTanimotoDistMat(data['fingerprint'].to_list())
    distance_matrix = np.zeros((len(data), len(data)))
    ind = np.tril_indices(len(data), -1)
    distance_matrix[ind] = distance_matrix_lower_triang
    distance_matrix = distance_matrix + distance_matrix.T
    # set the diagonal to a large value
    np.fill_diagonal(distance_matrix, 1000)


    # find the closest neighbours
    n_neigh = 4
    closest_neighbours = (pd.DataFrame(np.apply_along_axis(func1d= lambda x: np.argsort(x)[:n_neigh], axis=0, arr=distance_matrix).T, index=data['entry ID'])
                          .stack()
                          .rename('entry ID (close neighbour)').map(lambda idx: data.iloc[idx]['entry ID'])
                          .reset_index()
                          .rename({'level_1': 'close neighbour ID'})
                          )
    closest_neighbours['group'] = closest_neighbours['entry ID'].map(data[['entry ID', 'group']].set_index('entry ID').squeeze())
    closest_neighbours['group (close neighbour)'] = closest_neighbours['entry ID (close neighbour)'].map(data[['entry ID', 'group']].set_index('entry ID').squeeze())
    closest_neighbours['colour'] = closest_neighbours['entry ID'].map(data[['entry ID', 'colour']].set_index('entry ID').squeeze())
    closest_neighbours['colour (close neighbour)'] = closest_neighbours['entry ID (close neighbour)'].map(data[['entry ID', 'colour']].set_index('entry ID').squeeze())
    closest_distances = (pd.DataFrame(np.apply_along_axis(func1d= lambda x: np.sort(x)[:n_neigh], axis=0, arr=distance_matrix).T, index=data['entry ID'])
                         .stack()
                         .rename('distance')
                         .reset_index()
                         .rename({'level_1': 'close neighbour ID'}))
    closest_neighbours['distance'] = closest_distances['distance']


    # create the graph
    import networkx as nx
    graph = nx.Graph()
    # add nodes
    for idx, row in closest_neighbours[['entry ID', 'group', 'colour']].drop_duplicates().iterrows():
        attrs = {'group': row['group'],
                 'colour': row['colour']}
        graph.add_node(row['entry ID'], **attrs)
    # add edges
    for idx, row in closest_neighbours.iterrows():
        attrs = {'distance': (row['distance']+0.5)/10.}
        graph.add_edge(row['entry ID'], row['entry ID (close neighbour)'], **attrs)



    # compute the node positions
    log.info('computing the node positions...')
    pos = nx.layout.spring_layout(graph, iterations=200,
                                  # pos = pos,
                                  threshold=1.e-5,
                                  weight='distance',
                                  k=2./math.sqrt(graph.number_of_nodes()))
    log.info('node positions have been set')
    for n in graph.nodes:
        graph.nodes[n]['pos (spring)'] = pos[n]



    # draw the graph
    figsize=(6, 6)
    dpi=600
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.subplots()
    # .. draw the nodes
    node_sizes = 2
    node_colours = [graph.nodes[n]['colour'] for n in graph.nodes]
    node_labels = [graph.nodes[n]['group'] for n in graph.nodes]
    positions = {n: graph.nodes[n]['pos (spring)'] for n in graph.nodes}
    nodes = nx.draw_networkx_nodes(graph, pos=positions, ax=ax, nodelist=graph.nodes,
                                   node_size=node_sizes, node_color=node_colours, linewidths=None,
                                   alpha=1., label=node_labels)

    # .. draw the edges
    options = {}
    nx.draw_networkx_edges(graph, pos=positions, ax=ax, width=0.1, alpha=0.8,
                           edge_color='#AAAAAA', **options)


    # remove the axes
    ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(impath)
    plt.close(fig)
    log.info(f'figure saved in {str(impath)}')


    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
