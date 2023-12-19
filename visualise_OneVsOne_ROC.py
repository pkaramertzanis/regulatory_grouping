# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import textwrap
from itertools import combinations

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.metrics import roc_curve, auc

def group_name_format(group_name, maximum_group_name_length: int=50):
    group_name = group_name.lower()
    group_name = textwrap.shorten(group_name, width=maximum_group_name_length, placeholder="...")
    return group_name


def visualise_OneVsOne_AUC_ROC(y_scores, y_true, impath: Path, n_most_confused_groups:int=8, maximum_group_name_length: int=50) -> None:
    '''
    Visualises the OneVsOne ROC area under the curve for the n most confused group pairs
    :param y_scores: dataframe with shape (nmol, ngroups) with the probabilities for each group
    :param y_true: series of length nmol with the true group for each molecule
    :param impath: Path to export the figure as png
    :param n_most_confused_groups: visualise the n_most_confused_groups most confused groups
    :param maximum_group_name_length: shorten group names so that the name does not exceed this character limit
    :return: None
    '''

    pair_list = list(combinations(np.unique(y_true), 2))
    print(pair_list)
    pair_scores = []
    mean_tpr = dict()
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    for ix, (label_a, label_b) in enumerate(pair_list):
        log.info(f'evaluating oVo ROC for pair {ix} out of {len(pair_list)}')
        a_mask = y_true == label_a
        b_mask = y_true == label_b
        ab_mask = np.logical_or(a_mask, b_mask)

        a_true = a_mask[ab_mask]
        b_true = b_mask[ab_mask]

        fpr_a, tpr_a, _ = roc_curve(a_true, y_scores.loc[ab_mask, label_a])
        fpr_b, tpr_b, _ = roc_curve(b_true, y_scores.loc[ab_mask, label_b])

        mean_tpr[ix] = np.zeros_like(fpr_grid)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_a, tpr_a)
        mean_tpr[ix] += np.interp(fpr_grid, fpr_b, tpr_b)
        mean_tpr[ix] /= 2
        mean_tpr[ix][0] = 0.
        mean_score = auc(fpr_grid, mean_tpr[ix])
        pair_scores.append(
            {'group A': label_a, 'group B': label_b, 'oVo ROC AUC': mean_score, 'support A': a_mask.sum(),
             'support B': b_mask.sum(), 'fpr': fpr_grid, 'tpr': mean_tpr[ix]})
    pair_scores = pd.DataFrame(pair_scores)
    n_smallest = 8
    pair_scores_nsmallest = pair_scores.loc[pair_scores['oVo ROC AUC'].nsmallest(n_smallest).index].sort_values(
        by='oVo ROC AUC', ascending=False)

    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

    fig = plt.figure(figsize=(4, 4), dpi=600)
    ax = fig.subplots()
    for i_pair, (idx, pair_score) in enumerate(pair_scores_nsmallest.iterrows()):
        # add horizontal rectangles
        y_rect_anchor = i_pair
        x_rect_anchor = 0.5 - pair_score['oVo ROC AUC'] / 2.
        rect_width = pair_score['oVo ROC AUC']
        rect_height = 0.8
        rectangle = Rectangle([x_rect_anchor, y_rect_anchor], rect_width, rect_height, fc='w', ec='k', linewidth=0.5)
        rect = ax.add_patch(rectangle)

        # .. add the group labels
        text_h_offset = 0.02
        group_name_A = group_name_format(pair_score['group A'], maximum_group_name_length=maximum_group_name_length) \
                       + f'\n{pair_score["support A"]} substances'
        t = ax.text(x_rect_anchor - text_h_offset, y_rect_anchor + rect_height / 2, group_name_A,
                    ha="right", va="center", rotation=0, size=3,
                    bbox=dict(boxstyle="square,pad=0.3",
                              fc="lightgray", ec="gray", lw=0.1))
        group_name_B = group_name_format(pair_score['group B'], maximum_group_name_length=maximum_group_name_length) \
                       + f'\n{pair_score["support B"]} substances'
        t = ax.text(x_rect_anchor + rect_width + text_h_offset, y_rect_anchor + rect_height / 2, group_name_B,
                    ha="left", va="center", rotation=0, size=3,
                    bbox=dict(boxstyle="square,pad=0.3",
                              fc="lightgray", ec="gray", lw=0.2))

        # .. add the AUC ROC text
        auc_text = f'mean oVo AUC = {pair_score["oVo ROC AUC"]:.3f}'
        t = ax.text(x_rect_anchor + rect_width / 2, y_rect_anchor + rect_height / 2, auc_text,
                    ha="center", va="center", rotation=0, size=4)

        # add the ROC curve
        roc_offset = 0.1
        roc_plot_origin_y = y_rect_anchor + roc_offset * rect_height
        roc_plot_height = (1 - 2 * roc_offset) * rect_height

        roc_plot_origin_x = x_rect_anchor + roc_offset * rect_height / n_smallest
        roc_plot_width = (1 - 2 * roc_offset) * rect_height / n_smallest

        tpr = pair_score['tpr'] * roc_plot_height + roc_plot_origin_y
        fpr = pair_score['fpr'] * roc_plot_width + roc_plot_origin_x
        ax.plot(fpr, tpr, linestyle='-', color='r', linewidth=0.2)
        ax.plot([roc_plot_origin_x, roc_plot_origin_x + roc_plot_width],
                [roc_plot_origin_y, roc_plot_origin_y + roc_plot_height], linestyle='--', color='k', linewidth=0.1)

    plt.setp(ax, ylim=[0, n_smallest])
    plt.setp(ax.xaxis, visible=False)
    plt.setp(ax.yaxis, visible=False)
    plt.setp(ax.spines['top'], visible=False)
    plt.setp(ax.spines['bottom'], visible=False)
    plt.setp(ax.spines['right'], visible=False)
    plt.setp(ax.spines['left'], visible=False)
    fig.tight_layout()

    fig.savefig(impath)
    plt.close(fig)
    log.info(f'OneVsOne ROC AUC figure saved in {impath}')
    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')