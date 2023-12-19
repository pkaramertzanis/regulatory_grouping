# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import math
import textwrap
from typing import List
from collections import Counter

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat

from cheminfo_toolkit import Molecule, Fingerprint_engine

def visualise_fingerprint_heatmap(molecules: List[Molecule], impath: Path, minimum_group_size=10, fingerprint_engine: Fingerprint_engine=None, label_minimum_group_size=1, tick_label_size=2.2)-> None:
    '''
    Visualise the fingerprint heatmap using the molecular structures.

    :param molecules: list with correctly instantiated molecules
    :param impath: Path to export the figure as png
    :param minimum_group_size: Minimum number of substances to include group in the graph
    :param fingerprint_engine: Fingerprint_engine to compute the fingerprints, if None the default is used
    :param label_minimum_group_size: minimum group size to show the label for
    :param tick_label_size: tick label size (to be adjusted according to label_minimum_group_size)
    :return:
    '''

    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

    groups = [molecule.group for molecule in molecules]

    # select the groups for which we have a sufficient number of processable structures
    counts = Counter(groups)
    selected_groups = [group for group in counts if counts[group]>=minimum_group_size]

    # prepare the data
    mols = [mol.rdkit_mol for mol in molecules if mol.group in selected_groups]
    group_names = [mol.group for mol in molecules if mol.group in selected_groups]
    group_numbers = [mol.group_number for mol in molecules if mol.group in selected_groups]
    if fingerprint_engine is not None:
        fingerprints = [mol.compute_fingerprint(fingerprint_engine) for mol in molecules if mol.group in selected_groups]
    else:
        fingerprints = [mol.compute_fingerprint() for mol in molecules if mol.group in selected_groups]
    data = pd.DataFrame({'mol': mols, 'group': group_names, 'group number': group_numbers, 'fingerprint': fingerprints})

    # shorten the group name
    def group_name_format(group_name):
        group_name = group_name.lower()
        group_name = textwrap.shorten(group_name, width=40, placeholder="...")
        return group_name
    data['group'] = '('+data['group number'].astype(str)+') '+data['group'].apply(group_name_format)

    distance_matrix_lower_triang = GetTanimotoDistMat(data['fingerprint'].to_list())

    distance_matrix = np.zeros((len(data),len(data)))
    ind = np.tril_indices(len(data), -1)
    distance_matrix[ind] = distance_matrix_lower_triang
    distance_matrix = distance_matrix + distance_matrix.T


    plt.close('all')
    fig = plt.figure(figsize=(6,6), dpi=600)
    ax = fig.subplots()
    h = sns.heatmap(distance_matrix, ax=ax, cmap='rocket', square=True,
                    cbar_kws={"shrink": 0.3})
    plt.setp(plt.getp(ax, 'xaxis'), visible=True)
    plt.setp(plt.getp(ax, 'yaxis'), visible=True)
    # .. set the tick parameters of the legend
    ax.figure.axes[-1].tick_params(axis='y', labelsize=4, width=0.1, length=0.7)
    # .. set the major ticks of the heatmap
    ticks = data.reset_index(drop=True).reset_index().groupby('group')['index'].agg(['min', 'max'])
    ticks['tick_pos'] = (ticks['min'] + ticks['max']) / 2.
    major_ticks = ticks['tick_pos'].to_list()
    # filter the tick labels to show according to group size
    major_tick_labels = ticks.index.to_list()
    major_tick_labels = [(major_tcik_label if ticks['max'].iloc[i]-ticks['min'].iloc[i]+1>label_minimum_group_size else '') for i, major_tcik_label in enumerate(major_tick_labels)]

    ax.figure.axes[0].set_xticks(major_ticks)
    ax.figure.axes[0].set_xticklabels(major_tick_labels)
    ax.figure.axes[0].set_yticks(major_ticks)
    ax.figure.axes[0].set_yticklabels(major_tick_labels)
    ax.figure.axes[0].tick_params(axis='both',  # changes apply to the x-axis
                                  which='major',  # major ticks are affected
                                  bottom=False,  # ticks along the bottom edge are off
                                  top=False,  # ticks along the top edge are off
                                  left=False,
                                  right=False,
                                  labelbottom=True,
                                  labelleft=True,
                                  labeltop=False,
                                  labelright=False,
                                  labelsize=tick_label_size,
                                  length=0,
                                  pad=2.
                                  )
    # .. set the minor ticks of the heatmap
    minor_ticks = ticks['min'].to_list() + [ticks['max'].max()]
    ax.figure.axes[0].set_xticks(minor_ticks, minor=True)
    ax.figure.axes[0].set_yticks(minor_ticks, minor=True)
    ax.figure.axes[0].tick_params(axis='both',  # changes apply to the x-axis
                                  which='minor',  # minor ticks are affected
                                  bottom=True,  # ticks along the bottom edge are off
                                  top=True,  # ticks along the top edge are off
                                  left=True,  # ticks along the bottom edge are off
                                  right=True,  # ticks along the top edge are off
                                  labelbottom=False, # labels along the bottom edge are on
                                  labelleft=False, # labels along the left edge are on
                                  labeltop=False,  # labels along the bottom edge are on
                                  labelright=False,  # labels along the left edge are on
                                  labelsize=tick_label_size,
                                  width=0.1,
                                  length=0.7,
                                  )
    # .. create a custom grid
    for grid_line in minor_ticks:
        if grid_line>0 and grid_line<len(data):
            ax.figure.axes[0].plot([0, len(data)], [grid_line, grid_line], 'k-', lw=0.1)
            ax.figure.axes[0].plot([grid_line, grid_line], [0, len(data)], 'k-', lw=0.1)
    # ax.figure.axes[0].set_aspect('equal')
    fig.tight_layout()
    fig.savefig(impath)
    plt.close(fig)
    log.info(f'figure saved in {str(impath)}')


    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')