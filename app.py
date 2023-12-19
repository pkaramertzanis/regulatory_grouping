# setup logging
import logger
log = logger.setup_applevel_logger(file_name ='logs/ARN_groupings.log')

import matplotlib
matplotlib.use('Tkagg')
# %matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
import math
import re
from pathlib import Path
import textwrap
from inspect import getmembers, isfunction
from typing import List
from pickle import dump, load
import zipfile

import pickle

from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import Fragments
from rdkit.Chem import Draw
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRGroupDecomposition
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat

from cheminfo_toolkit import Molecule, Fingerprint_engine
from model_domain import Domain

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import classification_report

from visualise_structural_coverage import visualise_structural_coverage
from visualise_fingerprint_heatmap import visualise_fingerprint_heatmap
from visualise_ARN_groups import visualise_ARN_groups
from visualise_network import visualise_network
from visualise_confusion_matrix import visualise_confusion_matrix
from visualise_structure_proximity_2D import visualise_structure_proximity_2D
from visualise_rf_feature_importance import visualise_rf_feature_importance
from visualise_classification_report import visualise_classification_report
from visualise_OneVsRest_AUC_ROC import visualise_OneVsRest_AUC_ROC
from visualise_OneVsOne_ROC import visualise_OneVsOne_AUC_ROC

from build_model import select_groups, split_molecules_train_test
from build_model import build_random_forest_classifier, group_predictor_rf
from build_model import build_kNeighbours_classifier, group_predictor_kn
from build_model import build_gradient_boosting_classifier



# pandas display options
# do not fold dataframes
pd.set_option('expand_frame_repr', False)
# maximum number of columns
pd.set_option("display.max_columns",50)
# maximum number of rows
pd.set_option("display.max_rows",500)
# precision of float numbers
pd.set_option("display.precision",3)
# maximum column width
pd.set_option("max_colwidth", 120)

# set the output folder
output_folder = Path('output') / 'iteration13'


# read the dataset
f = f'input/2023_03_24_ARN_grouping.xlsx'
cols_ARN = ['entry ID', 'Group_name_ARN', 'Substance_name_ARN', 'EC_number_ARN', 'CAS_number_ARN']
cols_added = ['DSSTox_ID', 'DSSTox_structure_ID', 'DSSTox_QC_Level', 'Substance_name_DSSTox',
              'CAS_number_DSSTox', 'Substance_type_DSSTox', 'Substance_Note_DSSTox', 'SMILES_DSSTox',
              'InChI_DSSTox', 'InChIKey_DSSTox', 'Formula_DSSTox', 'MolWeight_DSSTox', 'SMILES_2D_QSAR_DSSTox']
arn_groups = pd.read_excel(f, usecols=cols_ARN+cols_added)
# add the group number for sorting purposes
arn_groups.insert(1, 'Group_number', arn_groups['Group_name_ARN'].map(pd.Series(data=range(0, arn_groups['Group_name_ARN'].nunique()), index=arn_groups['Group_name_ARN'].drop_duplicates().to_list())))

# read in molecules and put them in a list
molecules = []
n_succeeded = 0
n_failed = 0
fingerprint_engine = Fingerprint_engine.Morgan(radius=5, nBits=2028)
for idx, row in arn_groups.iterrows():
    smiles = row['SMILES_DSSTox']
    try:
        mol = Molecule.from_smiles(smiles)
        mol.group = row['Group_name_ARN']
        mol.group_number = row['Group_number']
        mol.CASRN = row['CAS_number_ARN']
        mol.compute_fingerprint(fingerprint_engine)
        molecules.append(mol)
        n_succeeded += 1
        arn_groups.loc[idx, 'structure processable'] = True
    except Exception as e:
        log.error(e)
        n_failed += 1
        arn_groups.loc[idx, 'structure processable'] = False
arn_groups['structure processable'] = arn_groups['structure processable'].astype(bool)
arn_groups.to_excel(output_folder / 'ARN_groups.xlsx')
log.info(f'Number of molecules that can be processed {n_succeeded} out of {n_succeeded+n_failed} in total')
with open(output_folder/'molecules_all.pickle', 'wb') as handle:
    pickle.dump(molecules, handle, protocol=pickle.HIGHEST_PROTOCOL)




# overall statistics (structural coverage)
arn_stats = (arn_groups
             .groupby(['Group_number', 'Group_name_ARN'])
             .agg(**{
                      "number of substances": pd.NamedAgg(column="entry ID", aggfunc=pd.Series.nunique),
                      "found in CompTox": pd.NamedAgg(column="DSSTox_ID", aggfunc=lambda s: len(s.dropna())),
                      "substance type": pd.NamedAgg(column="Substance_type_DSSTox", aggfunc=lambda s: s.value_counts().to_dict()),
                      "DSSTox QC level": pd.NamedAgg(column="DSSTox_QC_Level", aggfunc=lambda s: s.value_counts().to_dict()),
                      "SMILES available": pd.NamedAgg(column="SMILES_DSSTox", aggfunc=lambda s: len(s.dropna())),
                      "SMILES (2D-QSAR) available": pd.NamedAgg(column="SMILES_2D_QSAR_DSSTox", aggfunc=lambda s: len(s.dropna())),
                      "structure processable": pd.NamedAgg(column="structure processable", aggfunc=lambda s: s.sum()),
                })
             )
arn_stats['% of substances with SMILES (2D-QSAR) available'] = 100*arn_stats['SMILES (2D-QSAR) available']/arn_stats['number of substances']
arn_stats['% of substances with processable structure'] = 100*arn_stats['structure processable']/arn_stats['number of substances']
arn_stats = arn_stats.sort_values(by='% of substances with processable structure', ascending=False)
arn_stats.to_excel(output_folder / 'ARN_stats.xlsx')


# visualise structural coverage (we only keep the processable structures)
visualise_structural_coverage(arn_groups, impath=output_folder/'2023_03_20_groups_structural_information_show_quality.png', minimum_group_size=1, indicate_quality=True)
visualise_structural_coverage(arn_groups, impath=output_folder/'2023_03_20_groups_structural_information_indicate_small_show_quality.png', minimum_group_size=10, indicate_quality=True)
visualise_structural_coverage(arn_groups, impath=output_folder/'2023_03_20_groups_structural_information.png', minimum_group_size=1, indicate_quality=False)
visualise_structural_coverage(arn_groups, impath=output_folder/'2023_03_20_groups_structural_information_indicate_small.png', minimum_group_size=10, indicate_quality=False)

# visualise the ARN groups
visualise_ARN_groups(molecules, impath = output_folder/'ARN_visualisation', minimum_group_size=10)


# visualise fingerprint heatmap
fingerprint_engine = Fingerprint_engine.Morgan(radius=3, nBits=2048)
visualise_fingerprint_heatmap(molecules, impath = output_folder/'fingerprint_distance_heatmap_r3.png', minimum_group_size=10, fingerprint_engine=fingerprint_engine, label_minimum_group_size=1, tick_label_size=2.2)
visualise_fingerprint_heatmap(molecules, impath = output_folder/'fingerprint_distance_heatmap_r3_only_large_groups.png', minimum_group_size=10, fingerprint_engine=fingerprint_engine, label_minimum_group_size=30, tick_label_size=4)
fingerprint_engine = Fingerprint_engine.Morgan(radius=5, nBits=2048)
visualise_fingerprint_heatmap(molecules, impath = output_folder/'fingerprint_distance_heatmap_r5_minus_r3.png', minimum_group_size=10, fingerprint_engine=fingerprint_engine, label_minimum_group_size=1, tick_label_size=2.2)


# visualise the network
visualise_network(molecules, impath = output_folder/'ARN_network', minimum_group_size=20, number_closest_neighbours=2)


# visualise structure proximity in 2-D space using t_SNE
fingerprint_engine = Fingerprint_engine.Morgan(radius=3, nBits=2048)
for min_group_size in [10, 20, 30]:
    visualise_structure_proximity_2D(molecules, impath = output_folder/f'structure_proximity_2D_tSNE_{min_group_size}_structures_r3.png', minimum_group_size=min_group_size, random_state=0, fingerprint_engine=fingerprint_engine)



# ------------------  method 1: random forest classifier ------------------
# build random forest classifiers using different fingerprint options
molecules_regrouped = select_groups(molecules,
                                    minimum_group_size=10,
                                    small_groups_as_negative=True,
                                    pulled_small_group_name="miscellaneous chemistry")
molecules_train, molecules_test = split_molecules_train_test(molecules_regrouped, random_state=0, test_size=0.2, stratify=True) # this shuffles the data points by default
with open(output_folder/'training_set.pickle', 'wb') as handle:
    pickle.dump(molecules_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(output_folder/'test_set.pickle', 'wb') as handle:
    pickle.dump(molecules_test, handle, protocol=pickle.HIGHEST_PROTOCOL)
fingerprint_options = [{'radius': radius, 'nBits': nBits} for radius in [2,3,4,5] for nBits in [1536, 2048, 2560]]
# fingerprint_options = [{'radius': radius, 'nBits': nBits} for radius in [2] for nBits in [2560]]
models = []
for fingerprint_option in fingerprint_options:
    log.info(f'attempting finger print options {fingerprint_option}')
    model = {}
    fingerprint_engine = Fingerprint_engine.Morgan(radius=fingerprint_option['radius'], nBits=fingerprint_option['nBits'])
    model_details = build_random_forest_classifier(molecules_train, fingerprint_engine, random_state=0)
    model['finger options'] = fingerprint_option
    model['models details'] = model_details

    # examine the performance on the train set
    y_train, y_train_pred = zip(*[(mol.group, group_predictor_rf(mol, model_details)) for mol in molecules_train])
    f1_score_train = f1_score(y_train, y_train_pred, average='macro')
    model['f1 score (train)'] = f1_score_train

    # examine the performance on the test set
    y_test, y_test_pred = zip(*[(mol.group, group_predictor_rf(mol, model_details)) for mol in molecules_test])
    f1_score_test = f1_score(y_test, y_test_pred, average='macro')
    model['f1 score (test)'] = f1_score_test
    models.append(model)
# details of the outer and inner grid search
outer_inner_grid_details = []
for model in models:
    tmp =  model['models details']['grid search results'].copy()
    tmp.insert(0, column='fingerprint radius', value=model['finger options']['radius'])
    tmp.insert(1, column='fingerprint nBits', value=model['finger options']['nBits'])
    outer_inner_grid_details.append(tmp)
outer_inner_grid_details = pd.concat(outer_inner_grid_details, axis='index', ignore_index=True, sort=False)
outer_inner_grid_details.to_excel(output_folder/'outer_inner_grid_details_rf.xlsx')
# overview of fingerprint tuning effects
models_overview_fingerprint_tuning_rf = pd.DataFrame([(model['finger options']['radius'], model['finger options']['nBits'], model['models details']['best mean cross-validation score'], model['f1 score (train)'], model['f1 score (test)']) for model in models],
                               columns=['radius', 'nBits', 'f1 score (macro) (best cross-validation mean)', 'f1 score (macro) (train)', 'f1 score (macro) (test)'])
models_overview_fingerprint_tuning_rf.to_excel(output_folder/'models_overview_fingerprint_tuning_rf.xlsx')
# set the best model
best_model_rf = max([model for model in models], key=lambda model: model['models details']['best mean cross-validation score'])
best_five_cross_validations_rf = best_model_rf['models details']['grid search results'].sort_values(by='mean_test_score',ascending=False).iloc[:5][['params', 'mean_test_score','std_test_score']].to_markdown()
log.info('\n'+best_five_cross_validations_rf)
# plot the fingerprint tuning heatmaps
plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
for metric in ['f1 score (macro) (best cross-validation mean)', 'f1 score (macro) (train)', 'f1 score (macro) (test)']:
    fig = plt.figure(figsize=(4,2))
    ax = fig.subplots()
    sns.heatmap(models_overview_fingerprint_tuning_rf.pivot(index='radius', columns='nBits', values=metric), annot=True, fmt='.3f', cmap='Blues', ax=ax)
    fname = "fingerprint_tuning_"+re.sub(r'[\(\)\s\-]+','_',metric)+"_rf.png"
    fig.tight_layout()
    fig.savefig(output_folder/fname, dpi=600)
plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
# compute the domain of the model
domain_rf = Domain(molecules_train, fingerprint_engine=best_model_rf['models details']['fingerprint engine'])



# examine how many molecules in the training set are out of domain
count_mol_out_of_domain = 0
for mol in tqdm(molecules_train):
    in_domain = domain_rf.in_domain(mol)
    if not in_domain:
        count_mol_out_of_domain += 1
log.info(f'Number of molecules in the training set out of domain: {count_mol_out_of_domain} out of {len(molecules_train)}')


# examine how many molecules in the test set are out of domain
count_mol_out_of_domain = 0
domain_test = []
for mol in tqdm(molecules_test):
    in_domain = domain_rf.in_domain(mol)
    domain_test.append(in_domain)
    if not in_domain:
        count_mol_out_of_domain += 1
log.info(f'Number of molecules in the test set out of domain: {count_mol_out_of_domain} out of {len(molecules_test)}')

# visualise the feature importance with the best model
impath_importances = output_folder/'RF_feature_importance.png'
impath_structural_moieties = output_folder/'RF_feature_importance_structural_representation.png'
n_most_important_features = 40
# visualise_rf_feature_importance(best_model_rf=best_model_rf, impath=impath, molecules=molecules_train)
visualise_rf_feature_importance(best_model_rf,
                                impath_importances=impath_importances,
                                impath_structural_moieties=impath_structural_moieties,
                                n_most_important_features=n_most_important_features,
                                molecules=molecules_train)


# visualise the grid search scores
data = best_model_rf['models details']['grid search results'][['params', 'mean_test_score', 'std_test_score']]
data = pd.concat([data, pd.json_normalize(data['params'])], axis='columns').drop('params', axis='columns')
data = data.rename(lambda col: col.replace('rf__',''), axis='columns')
for min_samples_split in data['min_samples_split'].drop_duplicates().sort_values(ascending=True):
    msk = data['min_samples_split'] == min_samples_split
    res = data.loc[msk].pivot(index='n_estimators', columns='max_features', values='mean_test_score')
    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
    fig = plt.figure(figsize=(4,3), dpi=600)
    ax = fig.subplots()
    import seaborn as sns
    sns.heatmap(res, ax=ax, cmap='Blues', square=False, vmin=data['mean_test_score'].min(), vmax=data['mean_test_score'].max(),
                annot=True, fmt='.3f', annot_kws={'fontsize': 4},
                cbar_kws={"shrink": 0.5})
    ax.set_title(f'min_samples_split = {min_samples_split}')
    fig.tight_layout()
    fig.savefig(output_folder/f'grid_search_min_samples_split_{min_samples_split}_rf.png')
    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')



# visualise the confusion matrix for the training set and test set
group_name_number_mapping = {mol.group: f'({mol.group_number}) {mol.group}' for mol in molecules}
all_groups = pd.Series(list({mol.group for mol in molecules_regrouped})).sort_values(key=lambda s: s.str.lower())
y_train, y_train_pred = zip(*[(mol.group, group_predictor_rf(mol, best_model_rf['models details'])) for mol in molecules_train])
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) '+g)),
                           impath=output_folder/'confusion_training_set_rf.png')
y_test, y_test_pred = zip(*[(mol.group, group_predictor_rf(mol, best_model_rf['models details'])) for mol in molecules_test])
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_test_set_rf.png')
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train+y_test)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train_pred+y_test_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_all_data_rf.png')

# prepare and visualise the confusion report (in domain)
y_test_in_domain, y_test_pred_in_domain = zip(*[(mol.group, group_predictor_rf(mol, best_model_rf['models details'])) for mol in molecules_test if domain_rf.in_domain(mol)])
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test_in_domain)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test_pred_in_domain)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_test_set_rf_in_domain.png')


# prepare and visualise the classification report
report_train = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)).T.reindex(all_groups)
report_train.index = report_train.index.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
report_test = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T.reindex(all_groups)
report_test.index = report_test.index.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
report_all = pd.DataFrame(classification_report(y_train+y_test, y_train_pred+y_test_pred, output_dict=True)).T.reindex(all_groups)
report_all.index = report_all.index.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_classification_report(report_train, output_folder/'training_set_classification_report_rf.png')
report_train.to_excel(output_folder/'training_set_classification_report_rf.xlsx')
visualise_classification_report(report_train, output_folder/'training_set_classification_report_rf.png')
report_test.to_excel(output_folder/'test_set_classification_report_rf.xlsx')
visualise_classification_report(report_all, output_folder/'all_data_classification_report_rf.png')


# prepare and visualise the classification report (in domain)
y_test_in_domain, y_test_pred_in_domain = zip(*[(mol.group, group_predictor_rf(mol, best_model_rf['models details'])) for mol in molecules_test if domain_rf.in_domain(mol)])
report_test = pd.DataFrame(classification_report(y_test_in_domain, y_test_pred_in_domain, output_dict=True)).T.reindex(all_groups)
report_test.index = report_test.index.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_classification_report(report_test, output_folder/'test_set_classification_report_rf_in_domain.png')


# OneVsRest AUC ROC
molecules = [mol for mol in molecules_test]
y_scores = pd.DataFrame([group_predictor_rf(mol, best_model_rf['models details'], all_groups=True) for mol in molecules])
y_true = pd.Series([mol.group for mol in molecules])
y_scores.columns = y_scores.columns.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
y_true = y_true.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_OneVsRest_AUC_ROC(y_scores, y_true, impath=output_folder/'oneVsRest_AUC_ROC_test_set.png', maximum_group_name_length=50)
molecules = [mol for mol in molecules_train]
y_scores = pd.DataFrame([group_predictor_rf(mol, best_model_rf['models details'], all_groups=True) for mol in molecules])
y_true = pd.Series([mol.group for mol in molecules])
y_scores.columns = y_scores.columns.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
y_true = y_true.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_OneVsRest_AUC_ROC(y_scores, y_true, impath=output_folder/'oneVsRest_AUC_ROC_training_set.png', maximum_group_name_length=50)
molecules = [mol for mol in molecules_regrouped]
y_scores = pd.DataFrame([group_predictor_rf(mol, best_model_rf['models details'], all_groups=True) for mol in molecules])
y_true = pd.Series([mol.group for mol in molecules])
y_scores.columns = y_scores.columns.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
y_true = y_true.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_OneVsRest_AUC_ROC(y_scores, y_true, impath=output_folder/'oneVsRest_AUC_ROC_all_data.png', maximum_group_name_length=50)

# OneVsOne ROC, most confused classes
molecules = [mol for mol in molecules_test]
y_scores = pd.DataFrame([group_predictor_rf(mol, best_model_rf['models details'], all_groups=True) for mol in molecules])
y_true = pd.Series([mol.group for mol in molecules])
y_scores.columns = y_scores.columns.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
y_true = y_true.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_OneVsOne_AUC_ROC(y_scores, y_true, impath=output_folder/'oneVsOne_AUC_ROC_test_set.png', n_most_confused_groups=8, maximum_group_name_length=50)
molecules = [mol for mol in molecules_train]
y_scores = pd.DataFrame([group_predictor_rf(mol, best_model_rf['models details'], all_groups=True) for mol in molecules])
y_true = pd.Series([mol.group for mol in molecules])
y_scores.columns = y_scores.columns.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
y_true = y_true.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_OneVsOne_AUC_ROC(y_scores, y_true, impath=output_folder/'oneVsOne_AUC_ROC_training_set.png', n_most_confused_groups=8, maximum_group_name_length=50)
molecules = [mol for mol in molecules_regrouped]
y_scores = pd.DataFrame([group_predictor_rf(mol, best_model_rf['models details'], all_groups=True) for mol in molecules])
y_true = pd.Series([mol.group for mol in molecules])
y_scores.columns = y_scores.columns.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
y_true = y_true.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_OneVsOne_AUC_ROC(y_scores, y_true, impath=output_folder/'oneVsOne_AUC_ROC_all_data.png', n_most_confused_groups=8, maximum_group_name_length=50)




# ------------------  method 2: gradient boosting classifier  ------------------
# build nearest neighbour classifiers using different fingerprint options
molecules_regrouped = select_groups(molecules)
molecules_train, molecules_test = split_molecules_train_test(molecules_regrouped, random_state=0, test_size=0.2, stratify=True) # this shuffles the data points by default
# fingerprint_options = [{'radius': radius, 'nBits': nBits} for radius in [2,3,4,5] for nBits in [1536, 2048, 2560]]
fingerprint_options = [{'radius': radius, 'nBits': nBits} for radius in [2] for nBits in [2560]]
models = []
for fingerprint_option in fingerprint_options:
    log.info(f'attempting finger print options {fingerprint_option}')
    model = {}
    fingerprint_engine = Fingerprint_engine.Morgan(radius=fingerprint_option['radius'], nBits=fingerprint_option['nBits'])
    model_details = build_gradient_boosting_classifier(molecules_train, fingerprint_engine, random_state=0)
    model['finger options'] = fingerprint_option
    model['models details'] = model_details

    # examine the performance on the train set
    y_train, y_train_pred = zip(*[(mol.group, group_predictor_kn(mol, model_details)) for mol in molecules_train])
    f1_score_train = f1_score(y_train, y_train_pred, average='macro')
    model['f1 score (train)'] = f1_score_train

    # examine the performance on the test set
    y_test, y_test_pred = zip(*[(mol.group, group_predictor_kn(mol, model_details)) for mol in molecules_test])
    f1_score_test = f1_score(y_test, y_test_pred, average='macro')
    model['f1 score (test)'] = f1_score_test
    models.append(model)
# details of the outer and inner grid search
outer_inner_grid_details = []
for model in models:
    tmp =  model['models details']['grid search results'].copy()
    tmp.insert(0, column='fingerprint radius', value=model['finger options']['radius'])
    tmp.insert(1, column='fingerprint nBits', value=model['finger options']['nBits'])
    outer_inner_grid_details.append(tmp)
outer_inner_grid_details = pd.concat(outer_inner_grid_details, axis='index', ignore_index=True, sort=False)
outer_inner_grid_details.to_excel(output_folder/'outer_inner_grid_details_gb.xlsx')
# overview of fingerprint tuning effects
models_overview_fingerprint_tuning_gb = pd.DataFrame([(model['finger options']['radius'], model['finger options']['nBits'], model['models details']['best mean cross-validation score'], model['f1 score (train)'], model['f1 score (test)']) for model in models],
                               columns=['radius', 'nBits', 'f1 score (macro) (best cross-validation mean)', 'f1 score (macro) (train)', 'f1 score (macro) (test)'])
models_overview_fingerprint_tuning_gb.to_excel(output_folder/'models_overview_fingerprint_tuning_gb.xlsx')
# set the best model
best_model_gb = max([model for model in models], key=lambda model: model['models details']['best mean cross-validation score'])
best_five_cross_validations_gb = best_model_gb['models details']['grid search results'].sort_values(by='mean_test_score',ascending=False).iloc[:5][['params', 'mean_test_score','std_test_score']].to_markdown()
log.info('\n'+best_five_cross_validations_gb)
# plot the fingerprint tuning heatmaps (low resolution)
plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
for metric in ['f1 score (macro) (best cross-validation mean)', 'f1 score (macro) (train)', 'f1 score (macro) (test)']:
    fig = plt.figure(figsize=(4,2))
    ax = fig.subplots()
    sns.heatmap(models_overview_fingerprint_tuning_rf.pivot(index='radius', columns='nBits', values=metric), annot=True, fmt='.3f', cmap='Blues', ax=ax)
    fname = "fingerprint_tuning_"+re.sub(r'[\(\)\s\-]+','_',metric)+"_gb.png"
    fig.tight_layout()
    fig.savefig(output_folder/fname, dpi=600)
plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
# compute the domain of the model
domain_gb = Domain(molecules_train, fingerprint_engine=best_model_rf['models details']['fingerprint engine'])

# visualise the confusion matrix for the training set and test set
group_name_number_mapping = {mol.group: f'({mol.group_number}) {mol.group}' for mol in molecules}
all_groups = pd.Series(list({mol.group for mol in molecules_regrouped})).sort_values(key=lambda s: s.str.lower())
y_train, y_train_pred = zip(*[(mol.group, group_predictor_rf(mol, best_model_gb['models details'])) for mol in molecules_train])
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) '+g)),
                           impath=output_folder/'confusion_training_set_gb.png')
y_test, y_test_pred = zip(*[(mol.group, group_predictor_rf(mol, best_model_rf['models details'])) for mol in molecules_test])
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_test_set_gb.png')
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train+y_test)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train_pred+y_test_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_all_data_gb.png')




# ------------------  method 3: nearest neighbours ------------------
# build nearest neighbour classifiers using different fingerprint options
molecules_regrouped = select_groups(molecules)
molecules_train, molecules_test = split_molecules_train_test(molecules_regrouped, random_state=0, test_size=0.2, stratify=True) # this shuffles the data points by default
fingerprint_options = [{'radius': radius, 'nBits': nBits} for radius in [2,3,4,5] for nBits in [1536, 2048, 2560]]
# fingerprint_options = [{'radius': radius, 'nBits': nBits} for radius in [2] for nBits in [2560]]
models = []
for fingerprint_option in fingerprint_options:
    log.info(f'attempting finger print options {fingerprint_option}')
    model = {}
    fingerprint_engine = Fingerprint_engine.Morgan(radius=fingerprint_option['radius'], nBits=fingerprint_option['nBits'])
    model_details = build_kNeighbours_classifier(molecules_train, fingerprint_engine, random_state=0)
    model['finger options'] = fingerprint_option
    model['models details'] = model_details

    # examine the performance on the train set
    y_train, y_train_pred = zip(*[(mol.group, group_predictor_kn(mol, model_details)) for mol in molecules_train])
    f1_score_train = f1_score(y_train, y_train_pred, average='macro')
    model['f1 score (train)'] = f1_score_train

    # examine the performance on the test set
    y_test, y_test_pred = zip(*[(mol.group, group_predictor_kn(mol, model_details)) for mol in molecules_test])
    f1_score_test = f1_score(y_test, y_test_pred, average='macro')
    model['f1 score (test)'] = f1_score_test
    models.append(model)
# details of the outer and inner grid search
outer_inner_grid_details = []
for model in models:
    tmp =  model['models details']['grid search results'].copy()
    tmp.insert(0, column='fingerprint radius', value=model['finger options']['radius'])
    tmp.insert(1, column='fingerprint nBits', value=model['finger options']['nBits'])
    outer_inner_grid_details.append(tmp)
outer_inner_grid_details = pd.concat(outer_inner_grid_details, axis='index', ignore_index=True, sort=False)
outer_inner_grid_details.to_excel(output_folder/'outer_inner_grid_details_kn.xlsx')
# overview of fingerprint tuning effects
models_overview_fingerprint_tuning_kn = pd.DataFrame([(model['finger options']['radius'], model['finger options']['nBits'], model['models details']['best mean cross-validation score'], model['f1 score (train)'], model['f1 score (test)']) for model in models],
                               columns=['radius', 'nBits', 'f1 score (macro) (best cross-validation mean)', 'f1 score (macro) (train)', 'f1 score (macro) (test)'])
models_overview_fingerprint_tuning_kn.to_excel(output_folder/'models_overview_fingerprint_tuning_kn.xlsx')
# set the best model
best_model_kn = max([model for model in models], key=lambda model: model['models details']['best mean cross-validation score'])
best_five_cross_validations_kn = best_model_kn['models details']['grid search results'].sort_values(by='mean_test_score',ascending=False).iloc[:5][['params', 'mean_test_score','std_test_score']].to_markdown()
log.info('\n'+best_five_cross_validations_kn)
# plot the fingerprint tuning heatmaps (low resolution)
plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
for metric in ['f1 score (macro) (best cross-validation mean)', 'f1 score (macro) (train)', 'f1 score (macro) (test)']:
    fig = plt.figure(figsize=(4,2))
    ax = fig.subplots()
    sns.heatmap(models_overview_fingerprint_tuning_kn.pivot(index='radius', columns='nBits', values=metric), annot=True, fmt='.3f', cmap='Blues', ax=ax)
    fname = "fingerprint_tuning_"+re.sub(r'[\(\)\s\-]+','_',metric)+"_kn.png"
    fig.tight_layout()
    fig.savefig(output_folder/fname, dpi=600)
plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
# compute the domain of the model
domain_kn = Domain(molecules_train, fingerprint_engine=best_model_kn['models details']['fingerprint engine'])




# visualise the grid search scores
data = best_model_kn['models details']['grid search results'][['params', 'mean_test_score', 'std_test_score']]
data = pd.concat([data, pd.json_normalize(data['params'])], axis='columns').drop('params', axis='columns')
data = data.rename(lambda col: col.replace('kn__',''), axis='columns')
res = data.pivot(index='weights', columns='n_neighbors', values='mean_test_score')
plt.interactive('off')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')
fig = plt.figure(figsize=(4,3), dpi=600)
ax = fig.subplots()
sns.heatmap(res, ax=ax, cmap='Blues', square=False, vmin=data['mean_test_score'].min(), vmax=data['mean_test_score'].max(),
            annot=True, fmt='.3f', annot_kws={'fontsize': 4},
            cbar_kws={"shrink": 0.5})
fig.tight_layout()
fig.savefig(output_folder/f'grid_search_kn.png')
plt.interactive('on')
log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

# visualise the confusion matrix for the training set and test set
group_name_number_mapping = {mol.group: f'({mol.group_number}) {mol.group}' for mol in molecules}
all_groups = pd.Series(list({mol.group for mol in molecules_regrouped})).sort_values(key=lambda s: s.str.lower())
y_train, y_train_pred = zip(*[(mol.group, group_predictor_kn(mol, best_model_kn['models details'])) for mol in molecules_train])
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_training_set_kn.png')
y_test, y_test_pred = zip(*[(mol.group, group_predictor_kn(mol, best_model_kn['models details'])) for mol in molecules_test])
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_test_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_test_set_kn.png')
visualise_confusion_matrix(list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train+y_test)),
                           list(map(lambda g: group_name_number_mapping.get(g, '(-) '+g), y_train_pred+y_test_pred)),
                           all_groups=all_groups.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g)),
                           impath=output_folder/'confusion_all_data_kn.png')

# prepare and visualise the classification report
report_train = pd.DataFrame(classification_report(y_train, y_train_pred, output_dict=True)).T.reindex(all_groups)
report_train.index = report_train.index.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
report_test = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T.reindex(all_groups)
report_test.index = report_test.index.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
report_all = pd.DataFrame(classification_report(y_train+y_test, y_train_pred+y_test_pred, output_dict=True)).T.reindex(all_groups)
report_all.index = report_all.index.map(lambda g: group_name_number_mapping.get(g, '(-) ' + g))
visualise_classification_report(report_train, output_folder/'training_set_classification_report_kn.png')
report_train.to_excel(output_folder/'training_set_classification_report_kn.xlsx')
visualise_classification_report(report_test, output_folder/'test_set_classification_report_kn.png')
report_test.to_excel(output_folder/'test_set_classification_report_kn.xlsx')
visualise_classification_report(report_all, output_folder/'all_data_classification_report_kn.png')


# create a box plot with the F1 scores with the different group sizes, external (hold-out) set
group_size = pd.Series([mol.group for mol in molecules_regrouped if mol.group != 'miscellaneous chemistry']).value_counts().rename('group size').to_frame()
# kNN
y_test, y_test_pred = zip(*[(mol.group, group_predictor_kn(mol, best_model_kn['models details'])) for mol in molecules_test])
report_test_kNN = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T.reindex(all_groups)
# RF
y_test, y_test_pred = zip(*[(mol.group, group_predictor_kn(mol, best_model_rf['models details'])) for mol in molecules_test])
report_test_RF = pd.DataFrame(classification_report(y_test, y_test_pred, output_dict=True)).T.reindex(all_groups)
data = (group_size
        .join(report_test_kNN[['f1-score']]).rename({'f1-score': 'f1-score (kNN)'}, axis='columns')
        .join(report_test_RF[['f1-score']]).rename({'f1-score': 'f1-score (RF)'}, axis='columns')
        )
data['group size (bin)'] = pd.qcut(data['group size'], q=[0., 0.2, 0.4, 0.6, 0.8, 1], precision=0)
data['group size (bin, str)'] = None
for idx, row in data.iterrows():
    int_left = int(row['group size (bin)'].left)
    int_right = int(row['group size (bin)'].right)
    int_left_bracket = '[' if row['group size (bin)'].closed_left else '('
    int_right_bracket = ']' if row['group size (bin)'].closed_right else ')'
    if int_left <= data['group size'].min():
        int_left = int(data['group size'].min())
        int_left_bracket = '['
    data.loc[idx, 'group size (bin, str)'] = f'{int_left_bracket}{int_left}, {int_right}{int_right_bracket}'
data = (data[['group size (bin, str)', 'group size (bin)', 'f1-score (kNN)', 'f1-score (RF)']]
        .melt(id_vars=['group size (bin, str)', 'group size (bin)'], var_name='model', value_name='F1 score')
        .sort_values(by='group size (bin)')
        .rename({'group size (bin, str)': 'number of substances in the group'}, axis='columns'))
fig = plt.figure(figsize=(8,4))
ax = fig.subplots()
sns.swarmplot(data, x='number of substances in the group', y='F1 score', hue='model', ax=ax)
fig.savefig(output_folder/'F1_score_vs_group_size.png', dpi=600)

# find good examples for SHAP analysis, Paraben acid, salts and esters
# y_train, y_train_pred = zip(*[(mol.group, group_predictor_kn(mol, best_model_rf['models details'])) for mol in molecules_train])
# res_train = pd.Series(y_test_pred).loc[pd.Series(y_test)=='Paraben acid, salts and esters']
# res_train = pd.Series(y_train_pred).loc[pd.Series(y_train)=='Paraben acid, salts and esters']
# y_test, y_test_pred = zip(*[(mol.group, group_predictor_kn(mol, best_model_rf['models details'])) for mol in molecules_test])
# res_test = pd.Series(y_test_pred).loc[pd.Series(y_test)=='Paraben acid, salts and esters']
# pd.Series(group_predictor_rf(molecules_test[290], model_details=best_model_rf['models details'], all_groups=True)).sort_values()



# pickle the best rf model and all associated details
import pickle
with open(output_folder/'best_model_rf.pickle', 'wb') as handle:
    pickle.dump(best_model_rf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pickle the best kn model and all associated details
with open(output_folder/'best_model_kn.pickle', 'wb') as handle:
    pickle.dump(best_model_kn, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pickle the best gb model and all associated details
with open(output_folder/'best_model_gb.pickle', 'wb') as handle:
    pickle.dump(best_model_gb, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pickle the domain for rf
with open(output_folder/'domain_rf.pickle', 'wb') as handle:
    pickle.dump(domain_rf, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pickle the domain for kn
with open(output_folder/'domain_kn.pickle', 'wb') as handle:
    pickle.dump(domain_kn, handle, protocol=pickle.HIGHEST_PROTOCOL)

# pickle the domain for gb
with open(output_folder/'domain_gb.pickle', 'wb') as handle:
    pickle.dump(domain_gb, handle, protocol=pickle.HIGHEST_PROTOCOL)
