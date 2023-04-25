from .feature_vectors import *
from rdkit import Chem
import torch


class MolEncoder:
    """Computes the featurization from a molecule considering all atom and bond features, adapted from chemprop"""

    def __init__(self, mol: Chem.rdchem.Mol, property: str = "solvation", dummy=False):
        smiles = Chem.MolToSmiles(mol)
        self.mol = mol
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        f_atoms = []  # mapping from atom index to atom features
        f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        f_mol = []  # mol features
        a2b = []  # mapping from atom index to incoming bond indices
        b2a = (
            []
        )  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = []  # mapping from bond index to the index of the reverse bond
        self.fa_size = 0
        self.fb_size = 0
        self.fm_size = 0
        self.curr_a_hidden = None
        self.property = property

        # fake the number of "atoms" if we are collapsing substructures
        self.n_atoms = self.mol.GetNumAtoms()
        # Get atom features
        for atom in self.mol.GetAtoms():
            # if atom.GetSymbol() is not "H":
            af = AtomFeatureVector(atom, property=self.property)
            f_atoms.append(af.construct_vector())
            self.fa_size = af.get_atom_feature_dim()

        for _ in range(self.n_atoms):
            a2b.append([])
        if self.n_atoms == 1:
            a2b.append([])
            a2b.append([])
        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = self.mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue
                bf = BondFeatureVector(bond, property=self.property)
                f_bond = bf.construct_vector()
                self.fb_size = bf.get_bond_feature_dim()

                f_bonds.append(f_atoms[a1] + f_bond)
                f_bonds.append(f_atoms[a2] + f_bond)
                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                a2b[a2].append(b1)
                b2a.append(a1)
                a2b[a1].append(b2)
                b2a.append(a2)
                b2revb.append(b2)
                b2revb.append(b1)
                self.n_bonds += 2
        mf = MolFeatureVector(self.mol, property=self.property)
        f_mol = mf.construct_vector()
        self.fm_size = mf.get_mol_feature_dim()
        self.f_atoms = f_atoms
        self.f_bonds = f_bonds
        self.f_mol = f_mol
        self.a2b = a2b
        self.b2a = b2a
        self.b2revb = b2revb

    def get_components(self):
        return self.f_atoms, self.f_bonds, self.f_mol, self.a2b, self.b2a, self.b2revb

    def get_sizes(self):
        return self.fa_size, self.fb_size, self.fm_size

    def set_current_a_hiddens(self, a_message: torch.FloatTensor):
        self.curr_a_hidden = a_message

    def get_current_a_hiddens(self):
        return self.curr_a_hidden
