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


def group_name_format(group_name, maximum_group_name_length: int=50):
    group_name = group_name.lower()
    group_name = textwrap.shorten(group_name, width=maximum_group_name_length, placeholder="...")
    return group_name

def visualise_classification_report(classification_report: pd.DataFrame, impath: Path, maximum_group_name_length: int=50) -> None:
    '''
    Visualises the classification report
    :param classification_report: classification report produced by scikit-learn
    :param impath: Path to export the figure as png
    :param maximum_group_name_length: shorten group names so that the name does not exceed this character limit
    :return: None
    '''

    data = classification_report.reset_index().rename({'index': 'group'}, axis='columns').melt(id_vars=['group'], value_vars=['precision', 'recall', 'f1-score'], var_name='metric', value_name='value')
    data['group'] = data['group'].apply(group_name_format, maximum_group_name_length=maximum_group_name_length)
    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
    fig = plt.figure(figsize=(4, 3), dpi=600)
    ax = fig.subplots()
    p = sns.barplot(ax=ax, data=data, y='group', x='value', hue='metric', orient='h')
    sns.move_legend(
        ax, "lower center",
        bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False, fontsize=5
    )
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
    plt.setp(ax.spines['bottom'], position=('outward', 1.0))
    plt.setp(plt.getp(ax.xaxis, 'ticklabels'), fontsize=5)
    fig.tight_layout()
    fig.savefig(impath)
    plt.close(fig)
    log.info(f'classification report visualisation saved in {impath}')
    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
