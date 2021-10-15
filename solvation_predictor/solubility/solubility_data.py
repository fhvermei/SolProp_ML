import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.rdmolops as rdmolops


class SolubilityData:
    def __init__(self, df: pd.DataFrame, validate_smiles: bool = True, logger=None, verbose=True):
        """Reads in data used to calculate the solubility of neutral solutes in organic solvents
            :param df: pandas dataframe with columns 'solvent', 'solute', 'temperature' 'reference_solubility', and 'reference_solvent'.
            In general only the 'solute' column is mandatory, for solubility predictions at room temperature 'solute'
            and 'solvent', for other temperatures 'temperature'.
            If calculations use experimental reference solubility 'reference_solubility' and 'reference_solvent' have to be specified
            :param validate_smiles: validate the smiles and convert the inchis
        """
        self.df = df
        self.smiles_pairs = []
        self.temperatures = None
        self.reference_solubility = None
        self.reference_solvents = None
        self.validate = validate_smiles
        self.validate_smiles_error_messages = None
        self.df_wrongsmiles = None
        self.logger = logger.info if logger is not None else print
        
        if df is not None:
            self.get_data_from_df(verbose=verbose)
        else:
            raise ValueError('A dataframe needs to be provided with columns \'solvent\', \'solute\', '
                             '\'temperature\' \'logS_ref\', and \'solvent_ref\', '
                             'currently there is only one possibility to reading data')

    def get_data_from_df(self, verbose=False):
        self.df.columns = self.df.columns.str.strip().str.lower()
        if verbose:
            self.logger('Reading dataframe')
        if not 'solute' in self.df.columns:
            raise ValueError('CSV column names must have at least \'solvent\'')
        elif 'solute' in self.df.columns and not 'solvent' in self.df.columns:
            if verbose:
                self.logger('Reading only solute smiles, no solvent smiles are provided')
            self.smiles_pairs = [(None, row.solute) for i, row in self.df.iterrows()]
        else:
            if verbose:
                self.logger('Reading solute and solvent smiles')
            self.smiles_pairs = [(row.solvent, row.solute) for i, row in self.df.iterrows()]

        if self.validate:
            if verbose:
                self.logger('Validating smiles (or converting inchis)')
            self.validate_smiles()
            if len(self.validate_smiles_error_messages.keys()) > 0:
                self.df_wrongsmiles = pd.DataFrame()
                keys = np.array([np.array(i) for i in self.validate_smiles_error_messages.keys()])
                values = np.array([self.validate_smiles_error_messages[k] for k in self.validate_smiles_error_messages.keys()])
                self.df_wrongsmiles['solvent'] = keys[:, 0]
                self.df_wrongsmiles['solute'] = keys[:, 1]
                self.df_wrongsmiles['error'] = values
                idx = [self.df[(self.df['solvent'] == pair[0]) & (self.df['solute'] == pair[1])].index for pair in keys]
                for id in idx:
                    self.df.drop(id, inplace=True)
                if verbose:
                    self.logger(f'Validation done, {len(self.df_wrongsmiles)} smiles pairs are not correct')

        if 'temperature' in self.df.columns:
            if verbose:
                self.logger('Reading temperature')
            self.temperatures = self.df.temperature.values

        if 'reference_solubility' in self.df.columns and 'reference_solvent' in self.df.columns:
            if verbose:
                self.logger('Reading reference solubility')
            self.reference_solubility = self.df.reference_solubility.values
            self.reference_solvents = self.df.reference_solvent.values

        if verbose:
            self.logger('Done reading data')

    def validate_smiles(self):
        new_smiles_pairs = []
        wrong_smiles = dict()
        for pair in self.smiles_pairs:
            solvent_smiles = None
            solvent_charge = 0
            if pair[0] is not None:
                if 'InChI' in pair[0]:
                    mol = Chem.MolFromInchi(pair[0])
                else:
                    mol = Chem.MolFromSmiles(pair[0])

                try:
                    solvent_smiles = Chem.MolToSmiles(mol)
                    solvent_charge = rdmolops.GetFormalCharge(mol)
                except:
                    wrong_smiles[pair] = f'solvent id {pair[0]} cannot be converted by RDKit'
                if solvent_charge > 0 or solvent_charge < 0:
                    solvent_smiles = None
                    wrong_smiles[pair] = f'solvent id {pair[0]} has charge {solvent_charge} calculated by RDKit, ' \
                                            f'only neutral molecules are allowed'
            solute_smiles = None
            solute_charge = 0
            if 'InChI' in pair[1]:
                mol = Chem.MolFromInchi(pair[1])
            else:
                mol = Chem.MolFromSmiles(pair[1])
            try:
                solute_smiles = Chem.MolToSmiles(mol)
                solute_charge = rdmolops.GetFormalCharge(mol)
            except:
                wrong_smiles[pair] = f'solute id {pair[1]} cannot be converted by RDKit'

            if solute_charge > 0 or solute_charge < 0:
                solvent_smiles = None
                wrong_smiles[pair] = f'solvent id {pair[1]} has charge {solute_charge} calculated by RDKit, ' \
                                        f'only neutral molecules are allowed'
            if solvent_smiles is not None and solute_smiles is not None:
                new_smiles_pairs.append((solvent_smiles, solute_smiles))
        self.smiles_pairs = new_smiles_pairs
        self.validate_smiles_error_messages = wrong_smiles
