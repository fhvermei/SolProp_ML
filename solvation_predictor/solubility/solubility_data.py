import pandas as pd
import rdkit.Chem as Chem


class SolubilityData:
    def __init__(self, df: pd.DataFrame, validate_smiles: bool = False, logger=None):
        """Reads in data used to calculate the solubility of neutral solutes in organic solvents
            :param df: pandas dataframe with columns 'solvent', 'solute', 'temperature' 'reference_solubility', and 'reference_solvent'.
            In general only the 'solute' column is mandatory, for solubility predictions at room temperature 'solute'
            and 'solvent', for other temperatures 'temperature'.
            If calculations use experimental reference solubility 'reference_solubility' and 'reference_solvent' have to be specified
            :param validate_smiles: validate the smiles and convert the inchis
        """
        self.smiles_pairs = []
        self.temperatures = None
        self.reference_solubility = None
        self.reference_solvents = None
        self.validate = validate_smiles
        if df is not None:
            self.get_data_from_df(logger=logger)
            self.df = df
        else:
            raise ValueError('A dataframe needs to be provided with columns \'solvent\', \'solute\', '
                             '\'temperature\' \'logS_ref\', and \'solvent_ref\', '
                             'currently there is only one possibility to reading data')

    def get_data_from_df(self, logger=None):
        self.df.columns = self.df.columns.str.strip().str.lower()
        logger.info('Reading dataframe')
        if not 'solute' in self.df.columns:
            raise ValueError('CSV column names must have at least \'solvent\'')
        elif 'solute' in self.df.columns and not 'solvent' in self.df.columns:
            logger.info('Reading only solute smiles, no solvent smiles are provided')
            self.smiles_pairs = [(None, row.solute) for i, row in self.df.iterrows()]
        else:
            logger.info('Reading solute and solvent smiles')
            self.smiles_pairs = [(row.solvent, row.solute) for i, row in self.df.iterrows()]

        if 'temperature' in self.df.columns:
            logger.info('Reading temperature')
            self.temperatures = self.df.temperature.values

        if 'reference_solubility' in self.df.columns and 'reference_solvent' in self.df.columns:
            logger.info('Reading reference solubility')
            self.reference_solubility = self.df.reference_solubility.values
            self.reference_solvents = self.df.reference_solvent.values

        if self.validate:
            logger.info('Validating smiles (or converting inchis)')
            self.validate_smiles()
            logger.info('All identifiers are valid')
        logger.info('Done reading data')

    def validate_smiles(self):
        new_smiles_pairs = []
        for pair in self.smiles_pairs:
            solvent_smiles = None
            if pair[0] is not None:
                if 'InChI' in pair[0]:
                    mol = Chem.MolFromInchi(pair[0])
                else:
                    mol = Chem.MolFromSmiles(pair[0])
                try:
                    solvent_smiles = Chem.MolToSmiles(mol)
                except:
                    raise ValueError(f'solvent id {pair[0]} cannot be converted by RDKit')
            if 'InChI' in pair[1]:
                mol = Chem.MolFromInchi(pair[1])
            else:
                mol = Chem.MolFromSmiles(pair[1])
            try:
                solute_smiles = Chem.MolToSmiles(mol)
            except:
                raise ValueError(f'solute id {pair[1]} cannot be converted by RDKit')
            new_smiles_pairs.append((solvent_smiles, solute_smiles))
        self.smiles_pairs = new_smiles_pairs
