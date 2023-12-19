# setup logging
import logger
log = logger.get_logger(__name__)

import pickle
from pathlib import Path

import shap
import numpy as np

from rdkit.Chem import Draw, AllChem
from rdkit import Chem
from model_domain import Domain

from PIL import Image, ImageDraw, ImageFont
import io
import re

import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from cheminfo_toolkit import Molecule

output_folder = Path('output') / 'iteration13'

# load the training set
with open(output_folder/'training_set.pickle', 'rb') as handle:
    molecules_train = pickle.load(handle)

# load the test set
with open(output_folder/'test_set.pickle', 'rb') as handle:
    molecules_test = pickle.load(handle)

# load the best rf model
with open(output_folder/'best_model_rf.pickle', 'rb') as handle:
    best_model_rf = pickle.load(handle)

# load the domain
with open(output_folder/'domain_rf.pickle', 'rb') as handle:
    domain_rf = pickle.load(handle)


output_folder = Path('output') / 'iteration13' / 'feature_importance_per_group'


preprocessing = best_model_rf['models details']['best estimator'][0]
model = best_model_rf['models details']['best estimator'][1]
fingerprint_engine = best_model_rf['models details']['fingerprint engine']
group_numbering = {group: i for i, group in enumerate(model.classes_)}
feature_names_out = preprocessing.get_feature_names_out()


# create the explainer for all molecules in the group
groups = list({mol.group for mol in molecules_train})
mols = [mol for mol in molecules_train]
fingerprints_train = [mol.compute_fingerprint(fingerprint_engine).ToList() for mol in molecules_train]
X_train = preprocessing.transform(fingerprints_train)
explainer = shap.TreeExplainer(model, X_train, model_output='probability')


plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
for group in set([mol.group for mol in molecules_train]):
    log.info(f'processing group {group}')

    mols_group = [mol for mol in molecules_train if mol.group==group]
    fingerprints_group = [mol.compute_fingerprint(fingerprint_engine).ToList() for mol in mols_group]
    X_group = preprocessing.transform(fingerprints_group)


    # compute the shap values of the group
    shap_values = explainer.shap_values(X_group, check_additivity=True)

    # create bar plot with feature importance for each group
    plt.close('all')
    fig = plt.figure()
    shap.summary_plot(shap_values[group_numbering[group]], X_group, feature_names=feature_names_out, plot_type='bar', show=False)
    plt.gcf()
    plt.tight_layout()
    fpath = output_folder / ('feature_importance_'+re.sub(r'[/\<\\\s,\(\)]+', '_', group)+'.png')
    fig.savefig(fpath, dpi=600)
    log.info(f'figure saved in {str(fpath)}')

    # identify the important features
    n_important_features = 10 # will plot the 10 most important features
    feature_importances = np.abs(shap_values[group_numbering[group]]).mean(axis=0)
    important_feature_ids = np.argsort(feature_importances)[::-1][:n_important_features] # indices of array elements from maximum to minimum
    feature_importances[important_feature_ids]
    important_fingerprint_bits = [int(fbit.replace('x', '')) for fbit in feature_names_out[important_feature_ids]]


    # create the images with the structures of the important fingeprint bits
    images = []
    annots = []
    molSize = 400
    for i_bit, fbit in enumerate(important_fingerprint_bits):
        for mol in mols_group + molecules_train: # we put first the group molecules due to fingerprint clashes
            bi = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol.rdkit_mol,
                                                       radius=best_model_rf['models details']['fingerprint engine'].params['radius'],
                                                       nBits=best_model_rf['models details']['fingerprint engine'].params['nBits'],
                                                       bitInfo=bi)
            if fp[fbit]:
                im = Draw.DrawMorganBit(mol.rdkit_mol, fbit, bi, useSVG=False, molSize=(molSize, molSize))
                images.append(im)
                annots.append(f'x{fbit}, mean(|SHAP value|)={feature_importances[important_feature_ids[i_bit]]:0.4f}')
                print(i_bit, fbit)
                break
    images = [Image.open(io.BytesIO(image)) for image in images]

    # create a single figure with all fingerprint bit images
    n_columns = 5
    n_rows = 2
    pad = 20
    font = ImageFont.truetype("arial.ttf", 20)
    combined_image = Image.new('RGB', (n_columns * molSize, n_rows * (molSize + pad) + pad), (255, 255, 255))
    for i_image in range(n_important_features):
        # add the fingerprint bit visualisation
        x_pos = i_image % n_columns
        y_pos = i_image // n_columns
        combined_image.paste(images[i_image], (x_pos * molSize, y_pos * (molSize + pad)))
    # add the annotations
    combined_image_draw = ImageDraw.Draw(combined_image)
    for i_image in range(n_important_features):
        # add the fingerprint bit visualisation
        x_pos = i_image % n_columns
        y_pos = i_image // n_columns
        position = (x_pos * molSize + pad, (y_pos + 1) * (molSize + pad) - pad)
        combined_image_draw.text(position, annots[i_image], font=font, fill=(0, 0, 0))
        # get the text size using the font
        left, top, right, bottom = combined_image_draw.textbbox(position, annots[i_image], font=font)
        combined_image_draw.rectangle([left - pad / 2., top - pad / 2., right + pad / 2., bottom + pad / 2.],
                                      outline="black")

        fpath = output_folder / ('feature_structure_' + re.sub(r'[/\<\\\s,\(\)]+', '_', group) + '.png')
    combined_image.save(fpath)
    combined_image.close()
    log.info(f'figure saved in {str(fpath)}')
plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')


# SHAP waterfall plot for selected molecules in a groups
groups = ['Paraben acid, salts and esters', 'chlorinated aromatic hydrocarbons']
for group in groups:
    log.info(f'SHAP waterfall plots for group {group}')
    output_folder = Path('output') / 'iteration13' / 'feature_importance_per_group' / group.replace(' ','_').replace(',', '_')
    output_folder.mkdir(exist_ok=True)
    mols_group = [mol for mol in molecules_train+molecules_test if mol.group == group]
    fingerprints_group = [mol.compute_fingerprint(fingerprint_engine).ToList() for mol in mols_group]
    X_group = preprocessing.transform(fingerprints_group)

    # compute the shap values of the group
    sv = explainer(X_group, check_additivity=True)
    exp = shap.Explanation(sv[:,:,group_numbering[group]], sv.base_values[:,group_numbering[group]], X_group, feature_names=feature_names_out)
    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
    for i_mol in range(len(X_group)):
        log.info(f'waterfall for molecule {i_mol} with CAS {mols_group[i_mol].CASRN} and SMILES {Chem.MolToSmiles(mols_group[i_mol].rdkit_mol)}')
        plt.close('all')
        fig = plt.figure()
        shap.waterfall_plot(exp[i_mol], show=False)
        plt.gcf()
        plt.tight_layout()
        fpath = output_folder / (f'waterfall_{mols_group[i_mol].CASRN}.png')
        fig.savefig(fpath, dpi=600)
        log.info(f'figure saved in {str(fpath)}')
    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')



# -------------------------------- project ends here

# SHAP waterfall plot for selected molecule (not in the training set)
group = 'chlorinated aromatic hydrocarbons'
output_folder = Path('output') / 'iteration12'
mol = Molecule.from_smiles('C1=C(Cl)C(Cl)=C(Cl)C=C1C=O') # CAS 56961-76-3
fingerprint = mol.compute_fingerprint(fingerprint_engine).ToList()
X_mol = preprocessing.transform(np.array([fingerprint]))

# compute the shap values of the group
sv = explainer(X_mol, check_additivity=True)
exp = shap.Explanation(sv[:,:,group_numbering[group]], sv.base_values[:,group_numbering[group]], X_mol, feature_names=feature_names_out)
plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
plt.close('all')
fig = plt.figure()
shap.waterfall_plot(exp[0], show=False)
plt.gcf()
plt.tight_layout()
fpath = output_folder / (f'waterfall_56961-76-3.png')
fig.savefig(fpath, dpi=600)
log.info(f'figure saved in {str(fpath)}')
plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')



# ----
bi = {}
fp = AllChem.GetMorganFingerprintAsBitVect(mols_group[0].rdkit_mol,
                                           radius=best_model_rf['models details']['fingerprint engine'].params[
                                               'radius'],
                                           nBits=best_model_rf['models details']['fingerprint engine'].params['nBits'],
                                           bitInfo=bi)

molSize = 400
fbit = 2429
im = Image.open(io.BytesIO(Draw.DrawMorganBit(mols_group[0].rdkit_mol, fbit, bi, useSVG=False, molSize=(molSize, molSize))))
plt.imshow(im)
plt.show()

images.append(im)
annots.append(f'x{fbit}, mean(|SHAP value|)={feature_importances[important_feature_ids[i_bit]]:0.4f}')


fbit=242
for mol in mols_group:
    bi = {}
    fp = AllChem.GetMorganFingerprintAsBitVect(mol.rdkit_mol,
                                               radius=best_model_rf['models details']['fingerprint engine'].params[
                                                   'radius'],
                                               nBits=best_model_rf['models details']['fingerprint engine'].params[
                                                   'nBits'],
                                               bitInfo=bi)
    if fp[fbit]:
        mol
        im = Draw.DrawMorganBit(mol.rdkit_mol, fbit, bi, useSVG=False, molSize=(molSize, molSize))
        break
im = Image.open(io.BytesIO(im))
plt.imshow(im)
plt.show()