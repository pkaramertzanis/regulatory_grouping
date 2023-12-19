# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
import math
import textwrap

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt


def visualise_structural_coverage(arn_groups: pd.DataFrame, impath: Path, minimum_group_size=1, indicate_quality=True)-> None:
    '''
    Visualise the structural coverage using the 2D QSAR molecular structures.

    :param arn_groups: DataFrame with ARN grouping information.
    :param impath: Path to export the figure as png.
    :param minimum_group_size: groups with fewer than minimum_group_size processable structures will be indicated with red font
    :param indicate_quality: if True we show the high and medium/low structural coverage separately
    :return: None
    '''

    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

    # prepare the data
    cols = ['Group_number', 'Group_name_ARN', 'DSSTox_QC_Level', 'structure processable']
    data = arn_groups[cols].copy()

    # assign quality attribute
    quality_scores = {'high': ['DSSTox_High', 'Public_High_CAS', 'Public_High'],
                      'medium/low': ['DSSTox_Low', 'Public_Low', 'Public_Medium']}
    data['structural information'] = 'no structure'
    for quality_score, vals in quality_scores.items():
        data['structural information'] = np.where(data['DSSTox_QC_Level'].isin(vals), quality_score, data['structural information'])

    # check that the structure is processable, otherwise set the quality to 'no structure'
    msk = data['structure processable']
    data.loc[~msk, 'structural information'] = 'no structure'

    res = data.groupby(['Group_number', 'Group_name_ARN'])['structural information'].value_counts()
    res = res.unstack(level=2).fillna(0.).astype(int)
    res = res.reindex(['high', 'medium/low', 'no structure'], axis='columns')
    # compute percentages
    res_pct = res.div(res.sum(axis='columns'), axis='index')*100
    # sort the groups based on structure availability (regardless of quality)
    # idxs = res_pct.sort_values(by=res_pct['high'] + res_pct['medium/low'], ascending=False).index
    idxs = (res_pct['high'] + res_pct['medium/low']).sort_values(ascending=False).index
    res = res.loc[idxs]


    # create the pie chart graph
    n_cols = 8
    n_rows = math.ceil(len(res)/n_cols)
    plt.close('all')
    fig = plt.figure(figsize=(6, 6), dpi=600)
    fig.subplots_adjust()
    axs = fig.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw = {'wspace':0.01, 'hspace':0.10})
    for i_group, ax in enumerate(axs.flatten()):

        if i_group < len(res):
            group_data = res.iloc[i_group]
            # group_data = group_data[group_data>0]
            names = group_data.index.to_list()
            size = group_data.to_numpy()
            if not indicate_quality:
                size = np.array([size[:2].sum(), size[2]])
                # make the high quality structure count more pronounced
                explode = [0., 0.]
            else:
                explode = [0., 0., 0.]

            # set the colors
            if indicate_quality:
                colors = ['#29CE2F', '#F2D72D', '#000000']
            else:
                colors = ['#29CE2F', '#000000']

            # Create a circle at the center of the plot
            my_circle = plt.Circle((0, 0), 0.5, color='white')

            # add the size of the group
            ax.text(0, 0, group_data.sum(), ha="center", va="center", rotation=0, size=5, bbox=None)
            # add the name of the group
            group_number = group_data.name[0]
            group_name = group_data.name[1].lower()
            group_name = f'({group_number}) {group_name}'
            # group_name = textwrap.shorten(group_name, width=35, placeholder="...")
            group_name = group_name[:30] + ('...' if len(group_name)>30 else '')
            group_name = '\n'.join(textwrap.wrap(group_name, width=25, break_long_words=True))
            n_processable_structures = res.iloc[i_group][['high', 'medium/low']].sum()
            if n_processable_structures>=minimum_group_size:
                ax.text(0, -1.1, group_name, ha="center", va="top", rotation=0, size=3, bbox=None, color='k')
            else:
                ax.text(0, -1.1, group_name, ha="center", va="top", rotation=0, size=3, bbox=None, color='r')


            # Custom wedges
            ax.pie(size, labels=None, explode=explode, colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})
            ax.add_artist(my_circle)

        else:
            plt.setp(ax, visible=False)

    fig.savefig(impath)
    plt.close(fig)
    log.info(f'figure saved in {str(impath)}')

    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
