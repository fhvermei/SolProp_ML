import torch
import torch.nn as nn

from solvation_predictor.data.data import DataTensor
from solvation_predictor.inp import InputArguments

from solvation_predictor.models.ffn import FFN
from solvation_predictor.models.mpn import MPN
from logging import Logger


class Model(nn.Module):
    def __init__(self, inp: InputArguments, logger: Logger = None):
        super(Model, self).__init__()
        logger = logger.debug if logger is not None else print
        self.postprocess = inp.postprocess
        self.shared = inp.shared if not inp.num_mols == 1 else True
        self.feature_size = inp.num_features
        self.cudap = inp.cuda
        self.property = inp.property

        if not self.shared:
            #self.shared = False
            logger(f"Make {inp.num_mols} MPN models (no shared weight) with depth {inp.depth}, "
                   f"hidden size {inp.mpn_hidden}, dropout {inp.mpn_dropout}, "
                   f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}")
            #only works for 2 molecules
            self.mpn_1 = MPN(depth=inp.depth, hidden_size=inp.mpn_hidden, dropout=inp.mpn_dropout,
                             activation=inp.mpn_activation, bias=inp.mpn_bias, cuda=inp.cuda, atomMessage=False,
                             property=self.property, aggregation=inp.aggregation)
            self.mpn_2 = MPN(depth=inp.depth, hidden_size=inp.mpn_hidden, dropout=inp.mpn_dropout,
                             activation=inp.mpn_activation, bias=inp.mpn_bias, cuda=inp.cuda, atomMessage=False,
                             property=self.property, aggregation=inp.aggregation)
        else:
            logger(f"Make MPN model with depth {inp.depth}, hidden size {inp.mpn_hidden}, dropout {inp.mpn_dropout}, "
                   f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}")
            self.mpn = MPN(depth=inp.depth, hidden_size=inp.mpn_hidden, dropout=inp.mpn_dropout,
                           activation=inp.mpn_activation, bias=inp.mpn_bias, cuda=inp.cuda, atomMessage=False,
                           property=self.property, aggregation=inp.aggregation)

        logger(f"Make FFN model with number of layers {inp.ffn_num_layers}, hidden size {inp.ffn_hidden}, "
               f"dropout {inp.ffn_dropout}, activation function {inp.ffn_activation} and bias {inp.ffn_bias}")
        self.ffn = FFN((inp.mpn_hidden + inp.f_mol_size) * inp.num_mols + inp.num_features, inp.num_targets,
                       ffn_hidden_size=inp.ffn_hidden, num_layers=inp.ffn_num_layers, dropout=inp.ffn_dropout,
                       activation=inp.ffn_activation, bias=inp.ffn_bias)

    def forward(self, data):
        datapoints = data.get_data()

        if not self.shared:
            tensor_1 = []
            tensor_2 = []
            for d in data.get_data():
                tensor_1.append(d.get_mol_encoder()[0])
                tensor_2.append(d.get_mol_encoder()[1])
            tensor_1 = DataTensor(tensor_1, property=self.property)
            tensor_2 = DataTensor(tensor_2, property=self.property)
            mol_encoding_1, atoms_vecs_1 = self.mpn_1(tensor_1)
            mol_encoding_2, atoms_vecs_2 = self.mpn_2(tensor_2)
            input = torch.cat([mol_encoding_1, mol_encoding_2], dim=1)

        else:
            tensor = []
            for d in data.get_data():
                for enc in d.get_mol_encoder():
                    if enc:
                        tensor.append(enc)
            tensor = DataTensor(tensor, property=self.property)
            mol_encoding, atoms_vecs = self.mpn(tensor)
            num_mols = len(datapoints[0].get_mol())
            sizes = list(mol_encoding.size())
            new = sizes[0] / num_mols
            sizes[1] = int(sizes[0] * sizes[1] / new)
            sizes[0] = int(new)
            input = mol_encoding.view(sizes)

        if self.feature_size > 0:
            features = data.get_scaled_features()
            features = torch.FloatTensor(features)
            if self.cudap or next(self.parameters()).is_cuda:
                features = features.cuda()
            input = torch.cat([input, features], dim=1)

        if self.postprocess and not self.shared:
            for i in range(0, len(datapoints)):
                datapoints[i].updated_mol_vecs = [mol_encoding_1[i], mol_encoding_2[i]]

            for i in range(0, len(datapoints)):
                datapoints[i].updated_atom_vecs = [atoms_vecs_1[i], atoms_vecs_2[i]]

        if self.postprocess and self.shared:
            for i in range(0, len(datapoints)):
                datapoints[i].updated_mol_vecs = [mol_encoding[i]]

            for i in range(0, len(datapoints)):
                datapoints[i].updated_atom_vecs = [atoms_vecs[i]]

        output = self.ffn(input)
        del input

        for i in range(0, len(datapoints)):
            datapoints[i].scaled_predictions = output[i]

        return output

