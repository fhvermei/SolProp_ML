import torch
import torch.nn as nn

from solvation_predictor.data.data import DataTensor
from solvation_predictor.inp import InputArguments

from solvation_predictor.models.ffn import FFN
from solvation_predictor.models.mpn import MPN
from logging import Logger


class Model(nn.Module):
    """
    A class object that is a model which contains a message passing network following by feed-forward layers.
    """
    def __init__(self, inp: InputArguments, logger: Logger = None):
        super(Model, self).__init__()
        logger = logger.debug if logger is not None else print
        self.postprocess = inp.postprocess
        self.shared = inp.shared if not inp.num_mols == 1 else True
        self.feature_size = inp.num_features
        self.cudap = inp.cuda
        self.property = inp.property
        self.num_mols = inp.num_mols

        if not self.shared:
            # self.shared = False
            logger(
                f"Make {inp.num_mols} MPN models (no shared weight) with depth {inp.depth}, "
                f"hidden size {inp.mpn_hidden}, dropout {inp.mpn_dropout}, "
                f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}"
            )
            # only works for 2 molecules
            self.mpns = []
            for i in range(inp.num_mols):
                mpn = MPN(
                    depth=inp.depth,
                    hidden_size=inp.mpn_hidden,
                    dropout=inp.mpn_dropout,
                    activation=inp.mpn_activation,
                    bias=inp.mpn_bias,
                    cuda=inp.cuda,
                    atomMessage=False,
                    property=self.property,
                    aggregation=inp.aggregation,
                )
                self.mpns.append(mpn)
        else:
            logger(
                f"Make MPN model with depth {inp.depth}, hidden size {inp.mpn_hidden}, dropout {inp.mpn_dropout}, "
                f"activation function {inp.mpn_activation} and bias {inp.mpn_bias}"
            )
            self.mpn = MPN(
                depth=inp.depth,
                hidden_size=inp.mpn_hidden,
                dropout=inp.mpn_dropout,
                activation=inp.mpn_activation,
                bias=inp.mpn_bias,
                cuda=inp.cuda,
                atomMessage=False,
                property=self.property,
                aggregation=inp.aggregation,
            )

        logger(
            f"Make FFN model with number of layers {inp.ffn_num_layers}, hidden size {inp.ffn_hidden}, "
            f"dropout {inp.ffn_dropout}, activation function {inp.ffn_activation} and bias {inp.ffn_bias}"
        )
        self.ffn = FFN(
            (inp.mpn_hidden + inp.f_mol_size) * 2 + inp.num_features,
            inp.num_targets,
            ffn_hidden_size=inp.ffn_hidden,
            num_layers=inp.ffn_num_layers,
            dropout=inp.ffn_dropout,
            activation=inp.ffn_activation,
            bias=inp.ffn_bias,
        )

    def forward(self, data):
        """
        Runs the Model Class on input.

        :param data: Parameter containing the data on which the model needs to be run.
        :return: The output of the Class Model, containing a list of property predictions.
        """
        datapoints = data.get_data()

        if not self.shared:
            tensors = []
            molefracs = []

            for i in range(self.num_mols):
                tensors.append([])
            for d in datapoints:
                molefracs.append(d.molefracs)
                for i in tensors:
                    i.append([d.get_mol_encoder()[tensors.index(i)]])

            mol_encodings = []
            atoms_vecs = []
            for i in tensors:
                tensor = DataTensor(i[0], property=self.property)
                mol_encoding, atoms_vecs = self.mpns[tensors.index(i)](tensor)
                mol_encodings.append(mol_encoding)

            if len(molefracs) == 0:
                vec = mol_encodings[0]
            else:
                vec = torch.empty(mol_encodings[0].size())
                for i in range(1, self.num_mols):
                    vec = torch.add(vec, torch.mul(mol_encodings[i], molefracs[0][i-1]))
            input = torch.cat([mol_encodings[0], vec], dim=1)
        else:
            tensor = []
            for d in data.get_data():
                for enc in d.get_mol_encoder():
                    if enc:
                        tensor.append(enc)
            tensor = DataTensor(tensor, property=self.property)
            mol_encodings, atoms_vecs = self.mpn(tensor)
            num_mols = len(datapoints[0].get_mol())
            sizes = list(mol_encodings.size())
            new = sizes[0] / num_mols
            sizes[1] = int(sizes[0] * sizes[1] / new)
            sizes[0] = int(new)
            input = mol_encodings.view(sizes)
            mol_encodings, atoms_vecs = [mol_encodings], [atoms_vecs]

        if self.feature_size > 0:
            features = data.get_scaled_features()
            features = torch.FloatTensor(features)
            if self.cudap or next(self.parameters()).is_cuda:
                features = features.cuda()
            input = torch.cat([input, features], dim=1)

        if self.postprocess:
            for i in range(0, len(datapoints)):
                datapoints[i].updated_mol_vecs = mol_encodings[i]

            for i in range(0, len(datapoints)):
                datapoints[i].updated_atom_vecs = atoms_vecs[i]

        output = self.ffn(input)
        del input

        for i in range(0, len(datapoints)):
            datapoints[i].scaled_predictions = output[i]

        return output
