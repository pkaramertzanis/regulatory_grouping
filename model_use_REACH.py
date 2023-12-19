# setup logging
import logger
log = logger.get_logger(__name__)

import matplotlib
matplotlib.use('Tkagg')
# %matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from cheminfo_toolkit import Molecule, Fingerprint_engine
from build_model import group_predictor_rf, group_predictor_kn
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from model_domain import Domain
from glob import glob
import textwrap

from build_model import build_random_forest_classifier, group_predictor_rf, select_groups

from rdkit import Chem

from visualise_ARN_groups import visualise_ARN_groups
from tqdm import tqdm

output_folder = Path('output') / 'iteration13'

# load the molecules (whole dataset)
with open(output_folder/'molecules_all.pickle', 'rb') as handle:
    molecules = pickle.load(handle)
molecules_regrouped = select_groups(molecules,
                                    minimum_group_size=10,
                                    small_groups_as_negative=True,
                                    pulled_small_group_name="miscellaneous chemistry")


# build the RF model using all molecules in the dataset and the optimal model parameters
fingerprint_engine = Fingerprint_engine.Morgan(radius=2, nBits=2560)
parameters = {'rf__n_estimators': [150],
              'rf__max_features': [0.01],
              'rf__min_samples_split': [3]}
parameters = None
model_details = build_random_forest_classifier(molecules_regrouped, fingerprint_engine, random_state=0, parameters=parameters)

# compute the domain of the model
domain_rf = Domain(molecules_regrouped, fingerprint_engine=fingerprint_engine)


# # load the best rf model
# with open(output_folder/'best_model_rf.pickle', 'rb') as handle:
#     best_model_rf = pickle.load(handle)
# mod = best_model_rf['models details']['best estimator']
# mol = Chem.MolFromSmiles('ClC1CCCc2ccccc21')
# Chem.SanitizeMol(mol)
# Chem.MolToSmiles(mol)
# x  =fingerprint_engine.compute(mol)
# x = np.array(x.ToList()).reshape(1, -1)
# mod.predict(x)
# # load the domain
# with open(output_folder/'domain_rf.pickle', 'rb') as handle:
#     domain_rf = pickle.load(handle)


# read the arn groups as produced in app.py
arn_groups = pd.read_excel(output_folder / 'ARN_groups.xlsx')


# merge the DSSTox structure and run the predictions
fpath = r'D:\myApplications\local\2023_CompTox_structural_information\input\2023_01_02_structures'
# locate all CompTox sdfs
sdfs = glob(fpath + '/*sdf')
# .. keep the structures related to registered substances (using RDkit)
fname = r'input/2023_03_17_registrations_latest_submission.xlsx'
reg_data = pd.read_excel(fname)
reg_cas_numbers = set(reg_data['cas_number'].drop_duplicates())
mol_entries = []
mol_count = 0
for sdf in sdfs:
    print(f'processing {sdf}')
    # Load an SDF file
    mols = Chem.SDMolSupplier(sdf)
    # Iterate over the molecules in the file
    for mol in tqdm(mols):
        mol_entry = dict()
        mol_entry['file'] = sdf
        if mol:
            mol_count += 1
            # fetch the sdf properties
            for prop in list(mol.GetPropNames()):
                mol_entry[prop] = mol.GetProp(prop)
            if mol_entry.get('CASRN', '-') in reg_cas_numbers:
                mol = Molecule.from_rdkit_mol(mol)
                mol.CASRN = mol_entry.get('CASRN')
                mol_entry['mol'] = mol
                mol_entry['SMILES'] = Chem.MolToSmiles(mol.rdkit_mol)
                # predicted_group = group_predictor_rf(mol, best_model_rf['models details']) # <--
                predicted_groups = (
                    pd.Series(group_predictor_rf(mol, model_details=model_details, all_groups=True))
                    .sort_values(ascending=False)
                    .head(3)
                    .rename('group probability')
                    .reset_index()
                    .rename({'index': 'group name'}, axis='columns'))
                mol_entry['predicted group 1'], mol_entry['predicted group 2'], mol_entry['predicted group 3'] = predicted_groups['group name'].to_list()
                mol_entry['predicted group 1 probability'], mol_entry['predicted group 2 probability'], mol_entry['predicted group 3 probability'] = predicted_groups['group probability'].to_list()
                # mol_entry['predicted group'] = predicted_group
                # mol_entry['probability'] = best_model_rf['models details']['best estimator'].predict_proba(np.array(mol.fingerprint).reshape(1,-1)).max()
                mol_entry['in domain'] = domain_rf.in_domain(mol)
                log.info(f"mol {mol_count} (CAS {mol_entry.get('CASRN', '-')}) belongs is predicted to belong to group {mol_entry['predicted group 1']} with probability {mol_entry['predicted group 1 probability']: .3f}")
                mol_entries.append(mol_entry)
    #             if mol_count > 100:
    #                 break
    # break
mol_entries = pd.DataFrame(mol_entries)
mol_entries.drop('mol', axis='columns').to_parquet(output_folder/'rf_application_1_predicted_groups_only.parquet')


# join with the registration data
reg_data = reg_data.groupby(['ec_number', 'cas_number', 'substance_name'], dropna=False)[['asset_status', 'file_registered_full_tonnage', 'file_registered_tii_tonnage', 'file_registered_osii_tonnage']].apply(lambda df: df.drop_duplicates().to_json(orient='records'))
reg_data = reg_data.rename('registration details').reset_index()
modelled_groups = {group for group in model_details['best estimator'][1].classes_ if group!='miscellaneous chemistry'}
res_ap1 = (reg_data
            .merge(right=arn_groups[['EC_number_ARN', 'Group_name_ARN']], left_on='ec_number', right_on='EC_number_ARN', how='left')
            .drop('EC_number_ARN', axis='columns')
            .rename({'Group_name_ARN': 'true group'}, axis='columns'))
res_ap1['true group (regrouped)'] = res_ap1['true group'].loc[res_ap1['true group'].isin(modelled_groups)]
res_ap1.loc[res_ap1['true group'].notnull() & ~res_ap1['true group'].isin(modelled_groups), 'true group (regrouped)'] = 'miscellaneous chemistry'
cols = ['SMILES', 'predicted group 1', 'predicted group 2', 'predicted group 3', 'predicted group 1 probability', 'predicted group 2 probability', 'predicted group 3 probability']
res_ap1 = (res_ap1
            .merge(mol_entries[['CASRN'] + cols + ['in domain']].assign(**{'structure available': True}), left_on='cas_number', right_on='CASRN', how='left')
            .drop('CASRN', axis='columns'))
res_ap1['structure available'] = res_ap1['structure available'].fillna(False)


# export the results
res_ap1.to_excel(output_folder / 'rf_application_1_results.xlsx')


# function to format group names
def group_name_format(group_name):
    group_name = group_name.lower()
    group_name = textwrap.shorten(group_name, width=50, placeholder="...")
    return group_name


# check how many substances are added to groups as a function to probability threshold
plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
for in_domain in ['enforce domain', 'do not enforce domain']:
    for prob_threshold in np.linspace(0., 1., 11):
        log.info(f'processing the prediction results with probability threshold {prob_threshold:.2f}')
        tmp1 = res_ap1.groupby('true group (regrouped)').agg(**{
            'total number of substances': pd.NamedAgg(column="ec_number", aggfunc=lambda s: s.dropna().nunique()),
            'structure available': pd.NamedAgg(column="ec_number", aggfunc=lambda s: s.loc[res_ap1['structure available']].dropna().nunique()),
            'true positives': pd.NamedAgg(column="ec_number", aggfunc=lambda s: s.loc[res_ap1['structure available'] & (res_ap1['true group (regrouped)']==res_ap1['predicted group 1']) & (res_ap1['predicted group 1 probability']>=prob_threshold)].dropna().nunique())
        })
        if in_domain == 'do not enforce domain':
            tmp2 = res_ap1.groupby('predicted group 1').agg(**{
                'putative new members': pd.NamedAgg(column="ec_number", aggfunc=lambda s: s.loc[
                    res_ap1['structure available'] & res_ap1['true group (regrouped)'].isnull() & (
                                res_ap1['predicted group 1 probability'] >= prob_threshold)].dropna().nunique()),
            })
        elif in_domain == 'enforce domain':
            tmp2 = res_ap1.groupby('predicted group 1').agg(**{
                'putative new members': pd.NamedAgg(column="ec_number", aggfunc=lambda s: s.loc[
                    res_ap1['structure available']
                  & res_ap1['true group (regrouped)'].isnull()
                  & (res_ap1['predicted group 1 probability'] >= prob_threshold)
                  & res_ap1['in domain']].dropna().nunique())
            })
        tmp = tmp1.join(tmp2, how='left')
        data = tmp.reset_index().rename({'true group (regrouped)': 'group'}, axis='columns').melt(id_vars=['group'], value_vars=['total number of substances', 'structure available', 'true positives', 'putative new members'], var_name='count', value_name='value')

        data['group'] = data['group'].apply(group_name_format)
        fig = plt.figure(figsize=(6, 3), dpi=600)
        ax = fig.subplots()
        p = sns.barplot(ax=ax, data=data, y='group', x='value', hue='count', orient='h')
        p.set_xscale("log")
        sns.move_legend(
            ax, "lower center",
            bbox_to_anchor=(.5, 1), ncol=4, title=None, frameon=False, fontsize=4
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
        fpath = output_folder/f"rf_application_1_prob_{prob_threshold:0.2f}_{in_domain.replace(' ','_')}.png"
        fig.savefig(fpath)
        log.info(f'saved figure {fpath}')
        plt.close(fig)
plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
# visualise the putative new members
prob_threshold = 0.5
msk = (res_ap1['structure available']
      & res_ap1['true group (regrouped)'].isnull()
      & (res_ap1['predicted group 1 probability'] >= prob_threshold)
      & res_ap1['in domain']
       )
new_members_in_groups = res_ap1.loc[msk]
molecules_reach = []
group_number_name_mapping = arn_groups[['Group_name_ARN', 'Group_number']].drop_duplicates().set_index('Group_name_ARN').squeeze().to_dict()
group_number_name_mapping.update({'miscellaneous chemistry': '-'})
for idx, new_member in new_members_in_groups.iterrows():
    cas = new_member['cas_number']
    mol = mol_entries.loc[mol_entries['CASRN']==cas, 'mol'].iloc[0]
    mol.CASRN = cas
    mol.group = new_member['predicted group 1']
    mol.group_number = group_number_name_mapping[mol.group]
    molecules_reach.append(mol)
visualise_ARN_groups(molecules_reach, impath = output_folder/'rf_application1_ARN_visualisation_new_members', minimum_group_size=1)



