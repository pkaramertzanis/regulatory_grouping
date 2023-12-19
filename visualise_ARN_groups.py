# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import math
from pathlib import Path
import re

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt

from typing import List
from collections import Counter

from rdkit import Chem
from rdkit.Chem import Draw

from cheminfo_toolkit import Molecule

def visualise_ARN_groups(molecules: List[Molecule], impath: Path, minimum_group_size=10)-> None:
    '''
    Visualise the group structures for the groups, only substances with available structures are included

    :param molecules: List of molecules to visualise
    :param impath: Path to export the figure as png
    :param minimum_group_size: Minimum number of substances to visualise the group
    :return:
    '''

    if not impath.exists():
        e = IOError(f'folder {str(impath)} must exist')
        log.error(e)
        raise

    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

    groups = [molecule.group for molecule in molecules]

    # select the groups for which we have a sufficient number of processable structures
    counts = Counter(groups)
    selected_groups = [group for group in counts if counts[group]>=minimum_group_size]
    log.info(f'{len(selected_groups)} groups with a minimum of {minimum_group_size} structures will be visualised')

    # prepare the data
    mols = [mol.rdkit_mol for mol in molecules if mol.group in selected_groups]
    group_names = [mol.group for mol in molecules if mol.group in selected_groups]
    group_numbers = [mol.group_number for mol in molecules if mol.group in selected_groups]
    CASRNs = [mol.CASRN for mol in molecules if mol.group in selected_groups]
    data = pd.DataFrame({'mol': mols, 'group': group_names, 'group number': group_numbers, 'CASRN': CASRNs})




    for arn_group_name in selected_groups:
        log.info(f'visualising ARN group {arn_group_name}')
        msk = data['group'] == arn_group_name
        arn_group = data.loc[msk].copy()
        arn_group_number = arn_group['group number'].iloc[0]
        n_cols = 8
        n_rows = math.ceil(len(arn_group)/n_cols)
        plt.close('all')
        plt.ioff()
        fig = plt.figure(figsize=(6, 6), dpi=600)
        fig.subplots_adjust()
        axs = fig.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw = {'wspace':0.01, 'hspace':0.01})
        h = fig.text(0.1, 0.95, f'({arn_group_number}) {arn_group_name.lower()}', fontdict={'fontsize': 4})
        for i_mol, ax in enumerate(axs.flatten()):

            if i_mol < len(arn_group):
                plt.setp(ax, frame_on=False)
                plt.setp(plt.getp(ax, 'xaxis'), visible=False)
                plt.setp(plt.getp(ax, 'yaxis'), visible=False)
                row = arn_group.iloc[i_mol]
                cas_number = row['CASRN'] if not pd.isnull(row['CASRN']) else '-'
                mol_title = f"{cas_number}"
                h = ax.text(0.5, -.1, mol_title, ha="center", va="center", rotation=0, size=4.0, bbox=None, transform=ax.transAxes)
                im = Chem.Draw.MolToImage(row['mol'])
                ax.imshow(im)

            else:
                plt.setp(ax, visible=False)

        fpath = impath/(re.sub(r'[/\<\\\s,\(\)]+', '_', arn_group_name)+'.png')
        fig.savefig(fname = fpath)
        plt.close(fig)
        log.info(f'saved figure {str(fpath)} created')
        plt.close(fig)

    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')