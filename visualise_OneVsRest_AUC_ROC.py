# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, roc_auc_score


def group_name_format(group_name, maximum_group_name_length: int=50):
    group_name = group_name.lower()
    group_name = textwrap.shorten(group_name, width=maximum_group_name_length, placeholder="...")
    return group_name

def visualise_OneVsRest_AUC_ROC(y_scores, y_true, impath: Path, maximum_group_name_length: int=50) -> None:
    '''
    Visualises the OneVsRest ROC area under the curve
    :param y_scores: dataframe with shape (nmol, ngroups) with the probabilities for each group
    :param y_true: series of length nmol with the true group for each molecule
    :param impath: Path to export the figure as png
    :param maximum_group_name_length: shorten group names so that the name does not exceed this character limit
    :return: None
    '''

    # .. create the one hot encoding for the true group
    y_onehot = (pd.Series(1, index=pd.MultiIndex.from_arrays([y_scores.index.to_list(), y_true]), dtype=int)
               .unstack(level=1, fill_value=0)
               .reindex_like(y_scores))
    roc_curves = {}
    fpr_ticks = np.linspace(0, 1, 100)
    for group in y_scores.columns:
        fpr, tpr, thresholds = roc_curve(y_onehot.loc[:, group], y_scores.loc[:, group])
        interp_tpr = np.interp(fpr_ticks, fpr, tpr)
        interp_tpr[0] = 0.
        roc_curves[group] = {'fpr': fpr_ticks, 'tpr (interpolated)': interp_tpr, 'auc': roc_auc_score(y_onehot.loc[:, group], y_scores.loc[:, group])}

    data = pd.DataFrame({'group': roc_curves.keys(), 'auc': [roc['auc'] for roc in roc_curves.values()]})
    data['group'] = data['group'].apply(group_name_format, maximum_group_name_length=maximum_group_name_length)
    data['set'] = 'test set'
    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
    fig = plt.figure(figsize=(4, 3), dpi=600)
    ax = fig.subplots()
    p = sns.barplot(ax=ax, data=data, y='group', x='auc', orient='h', color='gray')
    ax.figure.axes[0].tick_params(axis='y',  # changes apply to the x-axis
                                  which='major',  # major ticks are affected
                                  bottom=False,  # ticks along the bottom edge are off
                                  top=False,  # ticks along the top edge are off
                                  left=False,
                                  right=False,
                                  labelbottom=False,
                                  labelleft=True,
                                  labeltop=False,
                                  labelright=False,
                                  labelsize=2.5,
                                  length=0,
                                  pad=2.
                                  )
    plt.setp(ax, xlabel=None)
    plt.setp(ax, ylabel=None)
    plt.setp(ax.spines['top'], visible=False)
    plt.setp(ax.spines['right'], visible=False)
    plt.setp(ax.spines['left'], visible=False)
    plt.setp(ax, xlim=(0.1*max(0, min(data['auc']-0.1)//0.1), 1.))
    plt.setp(ax, xlabel='One-vs-Rest ROC AUC')
    plt.setp(ax.xaxis.label, fontsize=5)
    plt.setp(ax.spines['bottom'], position=('outward', 1.0))
    plt.setp(plt.getp(ax.xaxis, 'ticklabels'), fontsize=5)
    fig.tight_layout()
    fig.savefig(impath)
    plt.close(fig)
    log.info(f'OneVsRest ROC AUC barplot saved in {impath}')
    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

