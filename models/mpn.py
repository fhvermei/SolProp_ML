import torch
import torch.nn as nn
from rdkit import Chem

from solvation_predictor.data.data import DataTensor
from solvation_predictor.features.molecule_encoder import MolEncoder


# from memory_profiler import profile


class MPN(nn.Module):

    def __init__(self, depth=3, hidden_size=100, dropout=0.1,
                 activation="ReLU", bias=False, cuda=False, atomMessage=False, property: str = "solvation",
                 aggregation: str = 'mean'):
        super(MPN, self).__init__()
        self.depth = depth
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.activation = get_activation_function(activation)
        self.bias = bias
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.cuda = cuda
        self.atom_messages = atomMessage
        self.prop = property
        self.aggregation = aggregation
        self.fa_size, self.fb_size, self.fm_size = self.sizes()
        if not self.atom_messages:
            self.W_i = nn.Linear(self.fb_size + self.fa_size, self.hidden_size, bias=self.bias)
            self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_o = nn.Linear(self.hidden_size + self.fa_size, self.hidden_size, bias=self.bias)
        else:
            self.W_i = nn.Linear(self.fa_size, self.hidden_size, bias=self.bias)
            self.W_h = nn.Linear(self.hidden_size, self.hidden_size, bias=self.bias)
            self.W_o = nn.Linear(self.hidden_size + self.fb_size, self.hidden_size, bias=self.bias)

    def sizes(self):
        dummy_smiles = "CC"
        dummy_mol = Chem.MolFromSmiles(dummy_smiles)
        dummy_encoder = MolEncoder(dummy_mol, property=self.prop)
        return dummy_encoder.get_sizes()

    def forward(self, data: DataTensor):
        f_atoms = data.f_atoms
        f_bonds = data.f_bonds
        f_mol = data.f_mols
        a2b = data.a2b
        b2a = data.b2a
        b2revb = data.b2revb
        ascope = data.ascope
        bscope = data.bscope

        # if self.atom_messages:
        #    a2a = mol_graph.get_a2a()
        if self.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, f_mol, a2b, b2a, b2revb = f_atoms.cuda(), f_bonds.cuda(), f_mol.cuda(), \
                                                        a2b.cuda(), b2a.cuda(), b2revb.cuda()
            # if self.atom_messages:
            #   a2a = a2a.cuda()

        # Input
        if self.atom_messages:
            input = self.W_i(f_atoms)  # num_atoms x hidden_size
        else:
            input = self.W_i(f_bonds)  # num_bonds x hidden_size

        message = self.activation(input)  # num_bonds x hidden_size

        # Message passing
        for depth in range(self.depth - 1):
            if self.atom_messages:
                print("atom messages not supported")
                # nei_a_message = index_select_ND(message, a2a)  # num_atoms x max_num_bonds x hidden
                # nei_f_bonds = index_select_ND(f_bonds, a2b)  # num_atoms x max_num_bonds x bond_fdim
                # nei_message = torch.cat((nei_a_message, nei_f_bonds), dim=2)  # num_atoms x max_num_bonds x hidden + bond_fdim
                # message = nei_message.sum(dim=1)  # num_atoms x hidden + bond_fdim
            else:
                # m(a1 -> a2) = [sum_{a0 \in nei(a1)} m(a0 -> a1)] - m(a2 -> a1)
                # message      a_message = sum(nei_a_message)      rev_message
                nei_a_message = index_select_ND(message, a2b)  # num_atoms x max_num_bonds x hidden
                a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden
                rev_message = message[b2revb]  # num_bonds x hidden
                message = a_message[b2a] - rev_message  # num_bonds x hidden

            message = self.W_h(message)
            message = self.activation(input + message)  # num_bonds x hidden_size
            message = self.dropout(message)  # num_bonds x

        # a2x = a2a if self.atom_messages else a2b

        a2x = a2b
        nei_a_message = index_select_ND(message, a2x)  # num_atoms x max_num_bonds x hidden
        a_message = nei_a_message.sum(dim=1)  # num_atoms x hidden

        a_input = torch.cat([f_atoms, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.activation(self.W_o(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden

        # ReadoutFalse
        mol_vecs = []
        atoms_vecs = []

        for i, (a_start, a_size) in enumerate(ascope):
            if a_size == 0:
                mol_vecs.append(self.cached_zero_vector)
            else:
                cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
                mol_vec = cur_hiddens  # (num_atoms, hidden_size)
                atoms_vecs.append(cur_hiddens)
                if self.aggregation == 'mean':
                    mol_vec = mol_vec.sum(dim=0) / a_size
                elif self.aggregation == 'sum':
                    mol_vec = mol_vec.sum(dim=0)
                else:
                    raise ValueError(f'aggregation function {self.aggregation} not defined')
                mol_vecs.append(mol_vec)

        mol_vecs = torch.stack(mol_vecs, dim=0)  # (num_molecules, hidden_size)
        mol_vecs = torch.cat([mol_vecs, f_mol], dim=1)  # (num_molecules, hidden_size)

        return mol_vecs, atoms_vecs


def get_activation_function(activation) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.
       :param activation: The name of the activation function.
       :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.
    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    # target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = torch.index_select(source, 0, index.view(-1))
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)
    return target
    # return source.index_select(dim=0, index=index.view(-1)).view(final_size)
