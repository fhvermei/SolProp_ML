from rdkit import Chem
from .calculated_features import *


class AtomFeatureVector:
    """Constructs a atom feature vector
    All atoms are supported, but more encoding for B, C, N, O, F, Si, P, S, Cl, Br, I"""

    def __init__(self, atom: Chem.rdchem.Atom, property: str = "solvation"):
        self.atom = atom
        self.vector = []
        self.property = property

    def construct_vector(self):
        self.vector += oneHotVector(self, get_atomic_number(self.atom),
                                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 35, 53])

        self.vector += oneHotVector(self, self.atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5])
        self.vector += [float(self.atom.GetFormalCharge())]  # make it positive and scale
        self.vector += [float(self.atom.GetTotalNumHs())]
        self.vector += oneHotVector(self, self.atom.GetHybridization(), [Chem.rdchem.HybridizationType.S,
                                                                         Chem.rdchem.HybridizationType.SP,
                                                                         Chem.rdchem.HybridizationType.SP2,
                                                                         Chem.rdchem.HybridizationType.SP3,
                                                                         Chem.rdchem.HybridizationType.SP3D,
                                                                         Chem.rdchem.HybridizationType.SP3D2])
        self.vector += [1 if self.atom.GetIsAromatic() else 0]
        self.vector += [self.atom.GetMass() * 0.01]

        if self.property == "solvation":
            self.vector += [float(get_num_lone_pairs(self.atom))]
            self.vector += oneHotVector(self, get_h_bond_donor(self.atom), [0, 1, 2, 3])
            self.vector += oneHotVector(self, get_h_bond_acceptor(self.atom), [0, 1, 2, 3])
            self.vector += [get_in_ring_size(self.atom)]
            self.vector += [get_electronegativity(self.atom) * 0.1]

        return self.vector

    def get_vector(self):
        return self.vector

    def get_atom_feature_dim(self):
        return len(self.vector)


class BondFeatureVector:
    """Constructs a bond feature vector"""

    def __init__(self, bond: Chem.rdchem.Bond, property: str = "solvation"):
        self.bond = bond
        self.vector = []
        self.property = property

    def construct_vector(self):
        if self.bond is None:
            self.vector = [1] + [0] * 13
        else:
            bt = self.bond.GetBondType()
            self.vector += oneHotVector(self, self.bond.GetBondType(), [0,
                                                                         Chem.rdchem.BondType.SINGLE,
                                                                         Chem.rdchem.BondType.DOUBLE,
                                                                         Chem.rdchem.BondType.TRIPLE,
                                                                         Chem.rdchem.BondType.AROMATIC])
            self.vector += [1 if self.bond.GetIsConjugated() else 0]
            self.vector += oneHotVector(self, int(self.bond.GetStereo()), list(range(6)))

            if self.property == "solvation":
                self.vector += [1 if self.bond.IsInRing() else 0]

        return self.vector

    def get_vector(self):
        return self.vector

    def get_bond_feature_dim(self):
        return len(self.vector)


class MolFeatureVector:

    def __init__(self, mol: Chem.rdchem.Mol, property: str = "solvation"):
        self.mol = mol
        self.vector = []
        self.property = property

    def construct_vector(self):
        if self.property == "solvation":
            self.vector += [topological_polar_surface_area(self.mol)]
            self.vector += [molecular_radius(self.mol)]
        return self.vector

    def get_vector(self):
        return self.vector

    def get_mol_feature_dim(self):
        return len(self.vector)


def oneHotVector(self, value, choices):
    encoding = [0] * (len(choices) + 1) #if value is not there, last index will be 1
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding
