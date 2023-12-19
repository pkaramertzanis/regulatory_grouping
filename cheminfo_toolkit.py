# setup logging
import logger
log = logger.get_logger(__name__)

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
from typing import Dict, List
from enum import Enum, auto

from collections import Counter

from io import StringIO
import sys


# create a class based context manager to capture rdkit error and warnings that typically go to the standard error
class Rdkit_operation:
    def __enter__(self):
        # redirect the standard error to a memory buffer
        rdkit.rdBase.LogToPythonStderr()
        sio = sys.stderr = StringIO()
        return sio
    def __exit__(self, exc_type, exc_value, exc_tb):
        # print(exc_type, exc_value, exc_tb, sep="\n")
        # set the standard error back to the default
        sys.stderr = sys.__stderr__
        return False # this propagates exceptions out of the working context (default)



class Molecule_format(Enum):
    '''Enum to define valid molecule input formats'''
    SMILES = auto()
    RDKIT_MOL = auto()


class Chem_exception(Exception):
    '''Custom exception used in the cheminformatics toolkit'''
    ...




class Fingerprint_engine():
    '''
    Class to compute fingerprints
    '''

    def __init__(self, type: str, params: Dict):
        self.type = type
        self.params = params

    @classmethod
    def Morgan(cls, radius: int=5, nBits: int=2048):
        '''
        Factory method to construct a Morgan fingerprint calculation engine

        :param radius: fingerprint radius
        :param nBits: number of bits
        :return: Fingerprint engine instance
        '''
        return cls(type='Morgan', params={'radius':radius, 'nBits': nBits})

    def compute(self, mol: Chem.Mol) -> np.array:
        '''Computes the fingerprint of an rdkit molecule

        :param mol: rdkit molecule
        :return: fingerprint bit vector
        '''

        if self.type == 'Morgan':
            try:
                radius = self.params['radius']
                nBits = self.params['nBits']
                with Rdkit_operation() as sio:
                    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
                    error_warning = sio.getvalue()
                    return fp
            except Exception as e:
                log.error(e)
                raise


class Molecule():
    '''
    Molecule objects are initiated once and have their fingerprints and other properties computed once.
    '''

    def __init__(self, rdkit_mol: Chem.Mol, creation_error_warning: str=None):
        '''
        Initialises a molecule using an rdkit molecule

        :param rdkit_mol: rdkit molecule
        :param creation_error_warning: string with error and/or warnings when the rdkit molecule is created
        :return: None
        '''
        self.rdkit_mol = rdkit_mol
        self.creation_error_warning = creation_error_warning
        self.compute_fingerprint()
        self.group = None
        self.__atom_list = []

    def compute_fingerprint(self, fingerprint_engine: Fingerprint_engine=None) -> None:
        '''
        Computes molecular fingerprint. It also sets the fingerprint instance attribute
        :param fingerprint_engine: Fingerprint engine instance. If None Morgan fingerprints with
                                   radius 3 and 2048 bits are used.
        :return: fingerprint (numpy array)
        '''
        if fingerprint_engine is None:
            default_fingerprint_engine = Fingerprint_engine.Morgan(radius=3, nBits=2048)
            self.fingerprint = default_fingerprint_engine.compute(self.rdkit_mol)
            return self.fingerprint
        else:
            self.fingerprint = fingerprint_engine.compute(self.rdkit_mol)
            return self.fingerprint


    def get_atom_list(self) -> List[str]:
        '''
        Returns the atom list. The atom list is only computed once
        :return: List of atom symbols as strings
        '''
        if not self.__atom_list:
            self.__atom_list = sorted(list({atom.GetSymbol() for atom in self.rdkit_mol.GetAtoms()}))
        return self.__atom_list

    @classmethod
    def from_smiles(cls, smiles: str):
        '''
        Factory method to instantiate a molecule from a smiles

        :param smiles: input smiles
        :return: molecule instance
        '''
        try:
            with Rdkit_operation() as sio:
                rdkit_mol = Chem.MolFromSmiles(smiles)
                error_warning = sio.getvalue()
                return cls(rdkit_mol=rdkit_mol, creation_error_warning=error_warning)
        except Exception as e:
            log.error(e)
            raise

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol: Chem.Mol):
        '''
        Factory method to instantiate a molecule from am rdkit molecule

        :param rdkit_mol: input rdkit molecule
        :return: molecule instance
        '''
        try:
            with Rdkit_operation() as sio:
                rdkit_mol = Chem.Mol(rdkit_mol)
                error_warning = sio.getvalue()
                return cls(rdkit_mol=rdkit_mol, creation_error_warning=error_warning)
        except Exception as e:
            log.error(e)
            raise

    @property
    def group(self):
        return self._group

    @group.setter
    def group(self, group_name: str):
        self._group = group_name

    @property
    def CASRN(self):
        return self._CASRN

    @CASRN.setter
    def CASRN(self, CASRN: str):
        self._CASRN = CASRN


molecule = Molecule.from_smiles('CCCC')
molecule.CASRN='asd'
