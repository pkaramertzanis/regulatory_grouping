# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import seaborn as sns


def visualise_confusion_matrix(y: pd.Series,
                               y_pred: pd.Series,
                               all_groups: pd.Series,
                               impath: Path)-> None:
    '''
    Visualise the confusion matrix.

    :param y: pandas series with true group assignments
    :param y: pandas series with predicted group assignments
    :param all_groups: all_groups to set the tick labels
    :param impath: Path to export the figure as png
    '''

    # function to shorten the group name in the labels
    def group_name_format(group_name):
        group_name = group_name.lower()
        group_name = textwrap.shorten(group_name, width=30, placeholder="...")
        return group_name

    all_groups = all_groups.copy()
    all_groups = all_groups.apply(group_name_format)

    y = pd.Series(y).apply(group_name_format).to_list() # use to_list to avoid index complication with the crosstab
    y_pred = pd.Series(y_pred).apply(group_name_format).to_list() # use to_list to avoid index complication with the crosstab
    compare = pd.crosstab(y, y_pred)
    compare = compare.reindex(all_groups, axis='index').reindex(all_groups,axis='columns').fillna(0.)
    compare = compare.div(compare.sum(axis=1), axis=0)

    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
    fig = plt.figure(figsize=(4, 4), dpi=600)
    ax = fig.subplots()
    sns.heatmap(compare, ax=ax, cmap='RdBu_r', square=True, vmin=0, vmax=1,
                linewidths=0.1, linecolor='white',
                xticklabels=True, yticklabels=True,
                # annot=False, fmt='.3f', annot_kws={'fontsize': 4},
                cbar_kws={"shrink": 0.60, 'location': 'top'})
    ax.figure.axes[-1].tick_params(axis='x', labelsize=4, width=0.1, length=0.7)
    ax.figure.axes[0].set_xlabel('predicted group', fontsize=4)
    ax.figure.axes[0].set_ylabel('true group', fontsize=4)
    ax.figure.axes[0].tick_params(axis='both',  # changes apply to the x-axis
                                  which='major',  # major ticks are affected
                                  bottom=True,  # ticks along the bottom edge are off
                                  top=False,  # ticks along the top edge are off
                                  left=True,
                                  right=False,
                                  labelbottom=True,
                                  labelleft=True,
                                  labeltop=False,
                                  labelright=False,
                                  labelsize=3,
                                  length=1.,
                                  width=0.1,
                                  pad=2.
                                  )
    fig.tight_layout()
    fig.savefig(impath)
    plt.close(fig)
    log.info(f'saved figure {str(impath)} created')
    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

