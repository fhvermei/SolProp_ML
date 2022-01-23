"""
This module defines the SolubilityHandler for use in Torchserve.
"""
from ts.torch_handler.base_handler import BaseHandler

from solvation_predictor.solubility.solubility_models import SolubilityModels
from utils import make_solubility_prediction


class SolubilityHandler(BaseHandler):

    def __init__(self):
        super().__init__()

    def initialize(self, context):
        self.models = SolubilityModels(
            reduced_number=False,  # if False, all 10 ML models are used per property to make predictions more accurate
            load_g=True,  # load models for solvation free energy
            load_h=True,  # load models for solvation enthalpy
            load_saq=True,  # load models for aqueous solid solubility
            load_solute=True,  # load models for Abraham solute parameters
            logger=None,  # logger file if one wants to save the logging information, else None
            verbose=True,  # whether to show logger info or not
        )
        self.initialized = True

    def preprocess(self, data):
        return data[0].get('data') or data[0].get('body')

    def inference(self, data, *args, **kwargs):
        return make_solubility_prediction(
            models=self.models,
            **data,
        )

    def postprocess(self, inference_output):
        return [inference_output]


if __name__ == '__main__':

    from ts.context import Context

    context = Context('model_name', '.', None, None, None, None)
    service = SolubilityHandler()
    service.initialize(context)

    input_data = [{'data': {
        'solvent_list': ['CC1=CC=CC=C1'],
        'solute_list': ['CC(C1=CC(=CC=C1)C(=O)C2=CC=CC=C2)C(=O)O'],
        'temp_list': [298],
        'ref_solvent_list': [None],
        'ref_solubility_list': [None],
        'ref_temp_list': [None],
        'hsub298_list': [None],
        'cp_gas_298_list': [None],
        'cp_solid_298_list': [None],
    }}]
    out = service.postprocess(service.inference(service.preprocess(input_data)))
    print(out)
