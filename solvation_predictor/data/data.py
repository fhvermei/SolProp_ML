import csv

import numpy as np
import torch
#from features.calculated_features import morgan_fingerprint
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch import nn
from solvation_predictor.features.molecule_encoder import MolEncoder
from torch.utils.data.dataset import Dataset
import random
import pandas as pd
import pickle

from solvation_predictor.inp import InputArguments


class DataPoint:
    """A datapoint contains a list of smiles, a list of targets and a list of features
    the smiles are converted to inchi (including a fixed H layer) and moles
    After operations it also contains a list of scaled targets, scaled features, scaled predictions and predictions"""

    def __init__(self, smiles, targets, features, molefracs, inp: InputArguments):
        if "InChI" in smiles[0]:
            self.mol = [Chem.MolFromInchi(i) if i else None for i in smiles]
            self.inchi = [Chem.MolToInchi(i, options='/fixedH') for i in self.mol]
            self.smiles = [Chem.MolToSmiles(i) if i else None for i in self.mol]
        else:
            self.smiles = smiles
            self.mol = [Chem.MolFromSmiles(i) if i else None for i in smiles]
            self.inchi = [Chem.MolToInchi(i, options='/fixedH') if i else None for i in self.mol]

        self.prop = inp.property
        self.add_hydrogens_to_solvent = inp.add_hydrogens_to_solvent
        self.mol_encoders = []
        self.updated_mol_vecs = []
        self.updated_atom_vecs = []
        self.targets = targets
        self.features = features
        self.molefracs = molefracs
        self.scaled_targets = []
        self.scaled_features = []
        self.scaled_predictions = []
        self.predictions = []
        self.scaffold = []
        self.epistemic_uncertainty = []

    def get_mol(self):
        if not self.mol:
            self.mol = [Chem.MolFromSmiles(i) if i else None for i in self.smiles]
            if self.add_hydrogens_to_solvent:
                self.mol[0] = Chem.AddHs(self.mol[0])
        return self.mol

    def get_scaffold(self):
        if not self.scaffold:
            if not self.mol:
                self.get_mol()
            self.scaffold = [MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
                             if mol else None for mol in self.mol]
        return self.scaffold

    def get_mol_encoder(self):
        if not self.mol_encoders:
            if not self.mol:
                self.get_mol()
            self.mol_encoders = [MolEncoder(m, dummy=False, property=self.prop) if m else None for m in self.mol]
        return self.mol_encoders

    def get_mol_encoder_components(self):
        if not self.mol_encoders:
            me = self.get_mol_encoder()
        else:
            me = self.mol_encoders
        fa = []
        fb = []
        fm = []
        a2b = []
        b2b = []
        b2revb = []
        for m in me:
            fa.append(m.get_components()[0])
            fb.append(m.get_components()[1])
            fm.append(m.get_components()[2])
            a2b.append(m.get_components()[3])
            b2b.append(m.get_components()[4])
            b2revb.append(m.get_components()[5])
        return fa, fb, fm, a2b, b2b, b2revb


class DatapointList(Dataset):
    """A DatapointList is simply a list of datapoints and allows for operations on the dataset"""

    def __init__(self, data):
        self.data = data

    def get_data(self):
        return self.data

    def get_targets(self):
        return [d.targets for d in self.data]

    def get_scaled_targets(self):
        return [d.scaled_targets for d in self.data]

    def get_features(self):
        return [d.features for d in self.data]

    def get_scaled_features(self):
        return [d.scaled_features for d in self.data]

    def get_molefracs(self):
        return [d.molefracs for d in self.data]

    def set_scaled_targets(self, l):
        for i in range(0, len(l)):
            self.data[i].scaled_targets = l[i]

    def set_scaled_features(self, l):
        for i in range(0, len(l)):
            self.data[i].scaled_features = l[i]

    def get_mol_encoders(self):
        return [d.get_mol_encoder() for d in self.data]

    def shuffle(self, seed: int = None):
        """
        Shuffles the dataset.

        :param seed: Optional random seed.
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.data)


class DataTensor:
    def __init__(self, list_mol_encoders: MolEncoder, property: str = "solvation"):
        self.list_mol_encoders = list_mol_encoders
        self.prop = property
        self.make_tensor()

    def make_tensor(self):
        fa_size, fb_size, fm_size = self.sizes()
        n_atoms = 1
        n_bonds = 1
        a_scope = []
        b_scope = []
        fa = [[0] * fa_size]
        fb = [[0] * (fa_size + fb_size)]
        fm = []
        a2b = [[]]
        b2a = [0]
        b2revb = [0]
        for enc in self.list_mol_encoders:
            if enc:
                fa.extend(enc.f_atoms)
                fb.extend(enc.f_bonds)
                fm.append(enc.f_mol)
                for a in range(0, enc.n_atoms):
                    a2b.append([b + n_bonds for b in enc.a2b[a]])
                for b in range(enc.n_bonds):
                    b2a.append(n_atoms + enc.b2a[b])
                    b2revb.append(n_bonds + enc.b2revb[b])
                a_scope.append((n_atoms, enc.n_atoms))
                b_scope.append((n_bonds, enc.n_bonds))
                n_atoms += enc.n_atoms
                n_bonds += enc.n_bonds
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        self.f_atoms = torch.FloatTensor(fa)
        self.f_bonds = torch.FloatTensor(fb)
        self.f_mols = torch.FloatTensor(fm)
        self.a2b = torch.LongTensor([a2b[a] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.b2revb = torch.LongTensor(b2revb)
        self.ascope = a_scope
        self.bscope = b_scope

    def sizes(self):
        dummy_smiles = "CC"
        dummy_mol = Chem.MolFromSmiles(dummy_smiles)
        dummy_encoder = MolEncoder(dummy_mol, property=self.prop)
        return dummy_encoder.get_sizes()

    def set_current_a_hiddens(self, a_hiddens: torch.FloatTensor):
        for i, (a_start, a_size) in enumerate(self.ascope):
            if a_size == 0:
                self.list_mol_encoders[i].set_updated_a_messages(
                    nn.Parameter(torch.zeros(a_hiddens.size()), requires_grad=False))
            else:
                cur_hiddens = a_hiddens.narrow(0, a_start, a_size)
                self.list_mol_encoders[i].set_current_a_hiddens(cur_hiddens)

    def get_current_a_hiddens(self):
        return [i.get_current_a_hiddens() for i in self.list_mol_encoders]


def read_data(inp: InputArguments, encoding='utf-8', file=None):
    """
    Reading in the data, assume input, features and next targets. The header should contain 'mol', 'feature', 'frac' or
    'target' as a keyword and headers without these keywords are not allowed. Input are either smiles or inchi, but
    should be the same if multiple molecules are read
    """
    if file is None:
        file = inp.input_file

    f = open(file, 'r', encoding=encoding)
    reader = csv.reader(f, delimiter=';')

    header = next(reader)
    all_data = list()
    solutes_count = []
    solvents_count = []
    targets_count = []
    features_count = []
    molefracs_count = []

    for i in header:
        if "solute" in i:
            solutes_count.append(header.index(i))
        if "solvent" in i:
            solvents_count.append(header.index(i))
        if "target" in i:
            targets_count.append(header.index(i))
        if "feature" in i:
            features_count.append(header.index(i))
        if "molefrac" in i:
            molefracs_count.append(header.index(i))

    if len(solutes_count) > 1:
        raise NotImplementedError("The SolProp package is not yet able to handle multiple solutes")
    if len(solutes_count) == 0 or len(solvents_count) == 0:
        raise NameError("The solute or solvent header is not found.")

    for line in reader:
        smiles = list()
        features = list()
        targets = list()
        molefracs = list()
        for count in solutes_count:
            smiles.append(line[count]) if line[count] != '' else smiles.append(None)
        for count in solvents_count:
            smiles.append(line[count]) if line[count] != '' else smiles.append(None)
        for count in targets_count:
            if len(line[count]) > 0:
                targets.append(float(line[count]))
            else:
                targets.append(np.NaN)
        for count in features_count:
            features.append(float(line[count])) \
                if line[count] else features.append(None)
        for count in molefracs_count:
            molefracs.append(float(line[count])) if line[count] != '' else molefracs.append(None)
            if line[count] != '' and len(molefracs_count) == 1:
                molefracs.append(1. - float(line[count]))
            if line[count] != '' and len(molefracs_count) > 1:
                raise NotImplementedError("Importing mole fractions for more than 2 solvent not possible yet.")
        # try:
        all_data.append(DataPoint(smiles, targets, features, molefracs, inp))
        # except:
        #     continue
    f.close()
    return all_data


def read_data_from_df(inp: InputArguments, encoding='utf-8', file=None, df=None, max=None):
    """
    Reading in the data, assume input, features and next targets. The header should contain 'mol', 'feature', 'frac' or
    'target' as a keyword and headers without these keywords are not allowed. Input are either smiles or inchi, but
    should be the same if multiple molecules are read
    """
    if file is None and df is None:
        file = inp.input_file
    elif df is None:
        if 'pickle' in file:
            with open(file, 'rb') as f:
                df = pickle.load(f)
        else:
            df = pd.read_csv(file)

    #shuffle data
    if max is not None:
        df = df.sample(n=max)

    header = df.columns

    all_data = list()
    species_columns = [i for i in header if 'mol' in i]
    target_columns = [i for i in header if 'target' in i]
    feature_columns = [i for i in header if 'feature' in i]

    for i, row in df.iterrows():
        smiles = list()
        fractions = list()
        features = list()
        targets = list()
        kmeans = list()
        for c in species_columns:
            smiles.append(row[c]) if row[c] != '' else smiles.append(None)
        for c in feature_columns:
            features.append(float(row[c])) if row[c] else features.append(None)
        for c in target_columns:
            targets.append(float(row[c]))

        try:
            all_data.append(DataPoint(smiles, targets, features, inp))
        except:
            continue
    return df, all_data
