# setup logging
import logger
log = logger.get_logger(__name__)

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use('Tkagg')
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

import math

from cheminfo_toolkit import Molecule, Fingerprint_engine

from rdkit.Chem import Draw, AllChem

from PIL import Image, ImageDraw, ImageFont
import io

def visualise_rf_feature_importance(best_model_rf: Dict,
                                    impath_importances: Path,
                                    impath_structural_moieties: Path,
                                    n_most_important_features: int,
                                    molecules: List[Molecule]) -> None:
    '''
    Visualises the feature importance for the best random forest model

    :param best_model_rf: dictionary with the best random forest model
    :param impath_importances: Path to export the figure with feature importances as png
    :param impath_structural_moieties: Path to export the figure with the structural representation of the important features as png
    :param n_most_important_features: number of important features to visualise
    :param molecules: List of molecules to use for visualising bits

    :return:
    '''

    # feature names
    feature_names_in = np.array([f'x{i}' for i in range(best_model_rf['finger options']['nBits'])])
    feature_names_out = best_model_rf['models details']['best estimator'][0].get_feature_names_out()

    # obtain the random forest estimator of the best model
    forest = best_model_rf['models details']['best estimator']._final_estimator

    # arrange feature importances in descending importance order
    importances = forest.feature_importances_
    ids = np.argsort(importances)[::-1]
    forest_importances_mean = pd.Series(importances, index=feature_names_out).iloc[ids]
    forest_importances_std = pd.Series(np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0), index=feature_names_out).iloc[ids]



    plt.interactive('off')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')

    # create the feature importance plot
    fig = plt.figure(figsize=(4,4), dpi=600)
    axs = fig.subplots(2,1)
    ax=axs[0]
    ax.plot(forest_importances_mean.reset_index(drop=True))

    loc = plticker.MultipleLocator(base=200.0)
    plt.setp(ax.xaxis, major_locator=loc)
    ax.tick_params(axis='x',  # changes apply to the x-axis
               which='major',  # major ticks are affected
               bottom=True,  # ticks along the bottom edge are on
               top=False,  # ticks along the top edge are off
               left=False,
               right=False,
               labelbottom=True,
               labelleft=False,
               labeltop=False,
               labelright=False,
               labelsize=5,
               length=0,
               pad=2.
               )
    ax.tick_params(axis='y',  # changes apply to the x-axis
               which='major',  # major ticks are affected
               bottom=True,  # ticks along the bottom edge are on
               top=False,  # ticks along the top edge are off
               left=False,
               right=False,
               labelbottom=False,
               labelleft=True,
               labeltop=False,
               labelright=False,
               labelsize=5,
               length=0,
               pad=2.
               )
    plt.setp(ax.spines['top'], visible=False)
    plt.setp(ax.spines['right'], visible=False)
    ax.set_ylabel("mean decrease in impurity")
    plt.setp(ax.yaxis.label, fontsize=5)
    ax.set_xlabel("feature number")
    plt.setp(ax.xaxis.label, fontsize=5)

    ax=axs[1]
    ax.barh(y=np.arange(0, n_most_important_features), width=forest_importances_mean.iloc[:n_most_important_features])
    ax.errorbar(forest_importances_mean.iloc[:n_most_important_features], np.arange(0, n_most_important_features), xerr=forest_importances_std.iloc[:n_most_important_features],
                fmt="o", color="k", capsize=1, elinewidth=0.2, capthick=0.2, markersize=1)

    ax.tick_params(axis='x',  # changes apply to the x-axis
                   which='major',  # major ticks are affected
                   bottom=True,  # ticks along the bottom edge are on
                   top=False,  # ticks along the top edge are off
                   left=False,
                   right=False,
                   labelbottom=True,
                   labelleft=False,
                   labeltop=False,
                   labelright=False,
                   labelsize=5,
                   length=0,
                   pad=2.
                   )
    ax.tick_params(axis='y',  # changes apply to the x-axis
                   which='major',  # major ticks are affected
                   bottom=True,  # ticks along the bottom edge are on
                   top=False,  # ticks along the top edge are off
                   left=False,
                   right=False,
                   labelbottom=False,
                   labelleft=True,
                   labeltop=False,
                   labelright=False,
                   labelsize=5,
                   length=0,
                   pad=2.
                   )
    plt.setp(ax.spines['top'], visible=False)
    plt.setp(ax.spines['right'], visible=False)
    ax.set_ylabel("feature number")
    plt.setp(ax.yaxis.label, fontsize=5)
    ax.set_xlabel("mean decrease in impurity")
    plt.setp(ax.xaxis.label, fontsize=5)

    # forest_importances_mean.plot.bar(yerr=forest_importances_std, ax=ax)
    fig.tight_layout()
    fig.savefig(impath_importances)
    plt.close(fig)
    log.info(f'figure saved in {str(impath_importances)}')



    plt.interactive('on')
    log.info(f'matplotlib interactivity set to {plt.isinteractive()}')



    # save the figure with structural features
    radius = best_model_rf['finger options']['radius']
    nBits = best_model_rf['finger options']['nBits']
    molSize = 400
    # Define the font and font size
    font = ImageFont.truetype("arial.ttf", 20)

    images = []
    annots = []
    for i_feature in range(n_most_important_features):
        log.info(f'drawing the structural feature {i_feature}')
        i_bit_out = ids[i_feature]
        i_bit_in = int(feature_names_out[i_bit_out].replace('x', ''))
        n_feature_occurrences = 0
        im = None
        for mol in molecules:
            bi = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol.rdkit_mol, radius=radius, nBits=nBits, bitInfo=bi)
            if fp[i_bit_in]:
                n_feature_occurrences += 1
                if im is None:
                    im = Draw.DrawMorganBit(mol.rdkit_mol, i_bit_in, bi, useSVG=False, molSize=(molSize,molSize))
        if im is not None:
            images.append(im)
            annots.append(f'imp: {forest_importances_mean.iloc[i_feature]:.4f}, {n_feature_occurrences} mols')

    images = [Image.open(io.BytesIO(image)) for image in images]
    n_columns = 10
    n_rows = math.ceil(n_most_important_features/n_columns)
    # merge the fingerprint bit images
    pad = 20
    combined_image = Image.new('RGB', (n_columns*molSize, n_rows*(molSize+pad)+pad), (255, 255, 255))
    for i_image in range(min(n_most_important_features,len(images))):
        # add the fingerprint bit visualisation
        x_pos = i_image%n_columns
        y_pos = i_image//n_columns
        combined_image.paste(images[i_image],(x_pos*molSize, y_pos*(molSize+pad)))
    # add the annotations
    combined_image_draw = ImageDraw.Draw(combined_image)
    for i_image in range(min(n_most_important_features,len(images))):
        # add the fingerprint bit visualisation
        x_pos = i_image%n_columns
        y_pos = i_image//n_columns
        position = (x_pos*molSize+pad, (y_pos+1)*(molSize+pad)-pad)
        combined_image_draw.text(position, annots[i_image], font=font, fill =(0, 0, 0))
        # get the text size using the font
        left, top, right, bottom = combined_image_draw.textbbox(position, annots[i_image], font=font)
        combined_image_draw.rectangle([left-pad/2., top-pad/2., right+pad/2., bottom+pad/2.], outline="black")


    combined_image.save(impath_structural_moieties)
    combined_image.close()
    log.info(f'figure saved in {str(impath_structural_moieties)}')

