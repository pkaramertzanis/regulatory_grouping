# setup logging
import logger
log = logger.get_logger(__name__)

from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm

from rdkit.Chem import AllChem
from cheminfo_toolkit import Fingerprint_engine

import matplotlib.pyplot as plt
import seaborn as sns

output_folder = Path('output') / 'iteration12'

# load the molecules (whole dataset)
with open(output_folder/'molecules_all.pickle', 'rb') as handle:
    molecules = pickle.load(handle)

fingerprint_lengths = list(range(1024, 1024 + 60*512, 512))
fingerprint_options = [{'radius': radius, 'nBits': nBits} for radius in [2,3,4,5] for nBits in fingerprint_lengths]
results = []
for fingerprint_option in tqdm(fingerprint_options):
    radius = fingerprint_option['radius']
    nBits = fingerprint_option['nBits']

    # compute the number of on bits
    FPs = []
    for molecule in molecules:
        mol = molecule.rdkit_mol
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits, bitInfo=info, useFeatures=False)
        FPs.append(fp.ToList())
    onBits = np.logical_or.reduce(FPs).sum()
    results.append({'radius': radius, 'nBits': nBits, 'onBits': onBits})
results = pd.DataFrame(results)

# plot the number of on bits as a function of the fingerprint length for each radius
fig = plt.figure()
ax = fig.subplots()
markers = ['o', 's', '<', '*', '>']
for i_radius, radius in enumerate(results['radius'].drop_duplicates()):
    msk = results['radius'] == radius
    ax.plot(results.loc[msk, 'nBits'], results.loc[msk, 'onBits'], color='k', marker=markers[i_radius], label=radius)
ax.legend(loc='upper left')
ax.set_xlabel('fingerprint length')
ax.set_ylabel('number of on bits')
ax.set_xticks(fingerprint_lengths, labels=fingerprint_lengths, rotation=90)
fig.savefig(output_folder/'fingerprint_length_effect.png', dpi=600)