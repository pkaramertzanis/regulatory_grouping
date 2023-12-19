# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import textwrap
from typing import List
from collections import Counter

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import matplotlib.patheffects as PathEffects
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.metrics import jaccard_score
from cheminfo_toolkit import Molecule, Fingerprint_engine

def visualise_structure_proximity_2D(molecules: List[Molecule], impath: Path, minimum_group_size: int=10, random_state: int=0, fingerprint_engine: Fingerprint_engine=None)-> None:
    '''
    Visualise the structure proximity in 2D space using t-SNE

    :param molecules: list with correctly instantiated molecules
    :param impath: Path to export the figure as png.
    :param minimum_group_size: Minimum number of substances to include group in the graph
    :param random_state: seed to ensure deterministic behaviour
    :param fingerprint_engine: Fingerprint_engine to compute the fingerprints, if None the default is used
    :return:
    '''

    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

    groups = [molecule.group for molecule in molecules]

    # select the groups for which we have a sufficient number of processable structures
    counts = Counter(groups)
    selected_groups = [group for group in counts if counts[group]>=minimum_group_size] # this array also defines the order
    log.info(f'plot will include {len(selected_groups)} groups')

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
        group_name = textwrap.shorten(group_name, width=30, placeholder="...")
        return group_name

    X = pd.DataFrame(data['fingerprint'].apply(lambda fp: fp.ToList()).to_list())

    tsne = TSNE(n_components=2, learning_rate='auto',
                init = 'pca', perplexity = 30, metric='jaccard',
                random_state=random_state)
    X_embedded = tsne.fit_transform(X)


    # create colours for the groups
    palette = np.array(sns.color_palette("hls", data['group'].nunique()))
    colours = [palette[i] for i in range(data['group'].nunique())]
    colours = {group: colour for group, colour in zip(data['group'].drop_duplicates(), colours)}
    data['group colour'] = data['group'].map(colours)

    # visualise the clusters in 2D space
    fig = plt.figure(figsize=(8,8), dpi=600)
    axs = fig.subplots(1,2, gridspec_kw={'width_ratios': [3, 1], 'wspace': 0.4})
    ax = axs[0]
    distances_from_median_centre_std_threshold = 4
    for i_group, group in enumerate(selected_groups):
        msk = data['group'] == group
        group_number = data.loc[msk, 'group number'].iloc[0]
        ax.scatter(X_embedded[msk, 0], X_embedded[msk, 1], c=data.loc[msk, 'group colour'], label=f"({group_number}) {group_name_format(group)}",  s=2)

        # Position of each label (only show if std deviation of median distances is below a threshold
        xtext, ytext = np.median(X_embedded[msk], axis=0)
        distances_from_median_centre = np.linalg.norm(X_embedded[msk] - np.array([xtext, ytext]), axis=1)
        distances_from_median_centre_std = np.std(distances_from_median_centre)
        if distances_from_median_centre_std <= distances_from_median_centre_std_threshold:
            txt = ax.text(xtext, ytext, str(group_number), fontsize=6)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=2, foreground="w"),
                PathEffects.Normal()])
            log.info(f'label for group {group_number, group} drawn')


    ax.tick_params(axis='both',  # changes apply to the x-axis
                   which='major',  # major ticks are affected
                   bottom=False,  # ticks along the bottom edge are off
                   top=False,  # ticks along the top edge are off
                   left=False,
                   right=False,
                   labelbottom=False,
                   labelleft=False,
                   labeltop=False,
                   labelright=False,
                   labelsize=2.5,
                   length=0,
                   pad=2.
                   )
    plt.setp(ax, xlabel=None)
    plt.setp(ax, ylabel=None)
    plt.setp(ax.spines['top'], visible=False)
    plt.setp(ax.spines['bottom'], visible=False)
    plt.setp(ax.spines['right'], visible=False)
    plt.setp(ax.spines['left'], visible=False)



    # visualise a horizontal barplot with group sizes
    ax = axs[1]
    group_sizes = data['group'].value_counts().reindex(selected_groups)
    group_sizes = group_sizes.rename('number of structures').reset_index().rename({'index': 'group'}, axis='columns')
    group_sizes['group colour'] = group_sizes['group'].map(colours)
    group_sizes['group number'] = group_sizes['group'].map(data[['group number', 'group']].drop_duplicates().set_index('group').squeeze())
    group_sizes['group short'] = '('+group_sizes['group number'].astype(str)+') '+group_sizes['group'].map(group_name_format)
    ax.barh(group_sizes['group short'], group_sizes['number of structures'], color=group_sizes['group colour'])

    loc = plticker.MultipleLocator(base=10.0)
    plt.setp(ax.xaxis, major_locator=loc)
    ax.tick_params(axis='y',  # changes apply to the y-axis
                                  which='major',  # major ticks are affected
                                  bottom=False,  # ticks along the bottom edge are off
                                  top=False,  # ticks along the top edge are off
                                  left=False,
                                  right=False,
                                  labelbottom=False,
                                  labelleft=True,
                                  labeltop=False,
                                  labelright=False,
                                  labelsize=5,
                                  length=0,
                                  pad=2.
                                  )
    plt.setp(ax, xlabel='number of structures')
    plt.setp(ax.get_xaxis().get_label(), fontsize=6)
    plt.setp(ax, ylabel=None)
    plt.setp(ax.spines['top'], visible=False)
    plt.setp(ax.spines['right'], visible=False)
    plt.setp(ax.spines['left'], visible=False)
    plt.setp(ax.spines['bottom'], position=('outward',-8.))
    plt.setp(plt.getp(ax.xaxis, 'ticklabels'), fontsize=6)



    # fig.tight_layout()
    fig.savefig(impath)
    plt.close(fig)
    log.info(f'figure saved in {str(impath)}')


    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')