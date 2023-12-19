# setup logging
import logger
log = logger.get_logger(__name__)

import matplotlib
matplotlib.use('Tkagg')
# %matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from cheminfo_toolkit import Molecule
from build_model import group_predictor_rf, group_predictor_kn
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from model_domain import Domain
from glob import glob
import textwrap
import random

from rdkit import Chem

from visualise_ARN_groups import visualise_ARN_groups


output_folder = Path('output') / 'iteration13'

# load the best rf model
with open(output_folder/'best_model_rf.pickle', 'rb') as handle:
    best_model_rf = pickle.load(handle)

# load the domain
with open(output_folder/'domain_rf.pickle', 'rb') as handle:
    domain_rf = pickle.load(handle)

# load the molecules in the training set
with open(output_folder/'training_set.pickle', 'rb') as handle:
    molecules_train = pickle.load(handle)

# feature names
feature_names_out = best_model_rf['models details']['best estimator'][0].get_feature_names_out()

# obtain the random forest estimator of the best model
forest = best_model_rf['models details']['best estimator']._final_estimator

# arrange feature importances in descending importance order
importances = forest.feature_importances_
ids = np.argsort(importances)[::-1]
forest_importances_mean = pd.Series(importances, index=feature_names_out).iloc[ids]

# keep only the most important bits
n_most_important_bits = 500
important_bits = [b for b in forest_importances_mean.iloc[:n_most_important_bits].index]

# compute the selected fingerprint bits for the molecules in the training set
fingerprint_engine = best_model_rf['models details']['fingerprint engine']
fingerprints = []
for mol in molecules_train:
    fingerprints.append(mol.compute_fingerprint(fingerprint_engine).ToList())
fingerprints = pd.DataFrame(fingerprints, columns=[f'x{i}' for i in range(best_model_rf['models details']['fingerprint engine'].params['nBits'])])

# keep only the selected bits
selected_fingerprints = fingerprints.loc[:, important_bits]
selected_fingerprints.insert(loc=0, column='group number', value=[mol.group_number for mol in molecules_train])
selected_fingerprints.insert(loc=1, column='group name', value=[mol.group for mol in molecules_train])

# keep only the explicitly modelled groups and order the rows
selected_fingerprints = selected_fingerprints.loc[selected_fingerprints['group name']!='miscellaneous chemistry']
selected_fingerprints = selected_fingerprints.sort_values(by='group number', ascending=True)

palette = np.array(sns.color_palette("hls", selected_fingerprints['group number'].nunique()))
colours = [palette[i] for i in range(selected_fingerprints['group number'].nunique())]
colours = {group: colour for group, colour in zip(selected_fingerprints['group number'].drop_duplicates(), colours)}
selected_fingerprints.insert(2, 'group colour', selected_fingerprints['group number'].map(colours))
selected_fingerprints = selected_fingerprints.iloc[::-1,:]

plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

cm = sns.clustermap(1-selected_fingerprints.iloc[:, 3:], row_colors=selected_fingerprints['group colour'], col_cluster=False, row_cluster=False, figsize=(10, 10), cmap='gray')
cm.cax.set_visible(False)

# .. create a custom grid with horizontal lines to separate the substance groups in the main part of the figure
first_mol_in_group_index = np.unique(selected_fingerprints['group number'], return_index=1)[1]
for grid_line in first_mol_in_group_index[:-1]:
    cm.figure.axes[3].plot([0, n_most_important_bits], [grid_line, grid_line], 'k-', lw=0.5, alpha=0.5)

# .. create a custom grid with horizontal lines to separate the substance groups in the color representation of the groups
for grid_line in first_mol_in_group_index[:-1]:
    cm.figure.axes[2].plot([0, 1.], [grid_line, grid_line], 'k-', lw=0.5, alpha=0.5)

# remove the ticks from the horizontal axis in the color representation of the groups
cm.figure.axes[2].set_xticks([])

# remove the ticks from the horizontal axis of the main part of the graph
cm.figure.axes[3].set_xticks([])

# .. set the major ticks of the heatmap
ticks = selected_fingerprints.reset_index(drop=True).reset_index().groupby('group number')['index'].agg(['min', 'max'])
ticks['tick_pos'] = (ticks['min'] + ticks['max']) / 2.
major_ticks = ticks['tick_pos'].to_list()
major_tick_labels = ticks.index.to_list()
cm.figure.axes[3].set_yticks(major_ticks)
cm.figure.axes[3].set_yticklabels(major_tick_labels)
cm.figure.axes[3].tick_params(axis='y',  # changes apply to the x-axis
                              which='major',  # major ticks are affected
                              bottom=False,  # ticks along the bottom edge are off
                              top=False,  # ticks along the top edge are off
                              left=False,
                              right=True,
                              labelbottom=False,
                              labelleft=False,
                              labeltop=False,
                              labelright=True,
                              labelsize=8,
                              length=0,
                              pad=2.
                              )


cm.figure.savefig(output_folder/'rf_fingerprint_bit_heatmap.png', dpi=600)


plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

# sns.heatmap(selected_fingerprints.iloc[:, 3:])