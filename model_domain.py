# setup logging
import logger
log = logger.get_logger(__name__)

import numpy as np
from typing import List

from rdkit.DataManip.Metric.rdMetricMatrixCalc import GetTanimotoDistMat
from rdkit import DataStructs
from cheminfo_toolkit import Molecule, Fingerprint_engine

class Domain:
    '''
    Class to define the domain of a model by using a set of molecules.

    For now the domain is simply defined based on the atoms present in the molecule set.
    '''
    n_closest_neighbours = 3
    percentile_cutoff = 95

    def __init__(self, molecules: List[Molecule], fingerprint_engine: Fingerprint_engine) -> None:
        '''
        Initialises the domain class by computing the element list
        and the distance threshold, so that 95% of the training set molecules are in domain, i.e. the average distance
        from the three closest analogues is less or equal to the threshold
        :param molecules: molecules in the training set
        :param fingerprint_engine: fingerprint engine to compute fingerprints
        '''
        # compute the element list
        self.__atom_list = []
        for mol in molecules:
            self.__atom_list.extend(mol.get_atom_list())
        self.__atom_list = sorted(set(self.__atom_list))
        log.info(f'elements present in molecule list: {self.__atom_list}')

        # compute the distance threshold
        fingerprints = [mol.compute_fingerprint(fingerprint_engine) for mol in molecules]
        distance_matrix_lower_triang = GetTanimotoDistMat(fingerprints)
        distance_matrix = np.zeros((len(fingerprints),len(fingerprints)))
        ind = np.tril_indices(len(fingerprints), -1)
        distance_matrix[ind] = distance_matrix_lower_triang
        distance_matrix = distance_matrix + distance_matrix.T
        # compute the 2nd-4th closest neighbour, the diagonal is zero and
        # corresponds to the distance of the molecule from itself
        idx = np.argpartition(distance_matrix, Domain.n_closest_neighbours+1, axis=1)[:, 1:Domain.n_closest_neighbours+1]
        min_distances = distance_matrix[np.arange(len(fingerprints))[:, np.newaxis], idx]
        min_distances = min_distances.mean(axis=1)
        self.__distance_threshold = np.percentile(min_distances, Domain.percentile_cutoff)
        log.info(f'distance threshold: {self.__distance_threshold}')

        # store the fingerprint engine and the fingerprints in the training set to apply the domain for new structures
        self.__fingerprint_engine = fingerprint_engine
        self.__fingerprints = fingerprints


    def get_domain(self) -> dict:
        '''
        Returns the domain of the model
        :return: List of atom symbols
        '''
        return {'allowed elements': self.__atom_list,
                'distance threshold': self.__distance_threshold}

    def in_domain(self, molecule: Molecule) -> bool:
        '''
        Examines if a molecule is in or out of the model domain. In order for a molecule to be in domain, two conditions
        need to be fulfilled, namely:
        - the molecule does not contain any elements not in training set
        - the average distance of the molecule to the Domain.n_closest_neighbours closest neighbours in the training set
          is less than the Domain.percentile_cutoff% percentile of the same distance for the molecules in the training set
        :param molecule:
        :return: boolean to indicate whether the molecule is in or out of domain
        '''

        # examine if the molecule contains any new elements
        mol_atoms = molecule.get_atom_list()
        condition1 = all([atom in self.__atom_list for atom in mol_atoms])

        # examine how far the molecule is from the molecules in the training set compared to the threshold
        GetTanimotoDistMat(self.__fingerprints+[molecule.compute_fingerprint(self.__fingerprint_engine)])

        distances = 1. - np.array(DataStructs.BulkTanimotoSimilarity(molecule.compute_fingerprint(self.__fingerprint_engine), self.__fingerprints))
        idx = distances.argpartition(Domain.n_closest_neighbours)[:Domain.n_closest_neighbours]
        average_min_distance = distances[idx].mean()
        condition2 = average_min_distance <= self.__distance_threshold
        log.info(f'all elements known: {condition1}, distance threshold met: {condition2}')

        return condition1 and condition2