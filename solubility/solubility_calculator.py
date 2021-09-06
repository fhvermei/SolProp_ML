import numpy as np
import rdkit.Chem as Chem
from solvation_predictor.solubility.solubility_predictions import SolubilityPredictions


class SolubilityCalculations:
    def __init__(self, predictions: SolubilityPredictions,
                 calculate_aqueous: bool = None,
                 calculate_reference_solvents: bool = None,
                 calculate_t_dep: bool = None,
                 logger=None):

        logger.info('Start making logS calculations')
        self.gsolv_298 = np.array(predictions.gsolv[0]) if predictions.gsolv else None  # in kcal/mol
        self.unc_gsolv_298 = np.sqrt(np.array(predictions.gsolv[1])) if predictions.gsolv else None

        self.logk_298 = self.calculate_logk(gsolv=self.gsolv_298)
        self.unc_logk_298 = self.calculate_logk(gsolv=self.unc_gsolv_298, uncertainty=True)

        if calculate_aqueous:
            logger.info('Calculating logS at 298K from predicted aqueous solubility')
            self.gsolv_aq_298 = np.array(predictions.gsolv_aq[0]) if predictions.gsolv_aq else None  # in kcal/mol
            self.unc_gsolv_aq_298 = np.sqrt(np.array(predictions.gsolv_aq[1])) if predictions.gsolv_aq else None

            self.logk_aq_298 = self.calculate_logk(gsolv=self.gsolv_aq_298)
            self.unc_logk_aq_298 = self.calculate_logk(gsolv=self.unc_gsolv_aq_298, uncertainty=True)

            self.logs_aq_298 = np.array(predictions.saq[0]) if predictions.saq else None  # in log10(mol/L)
            self.unc_logs_aq_298 = np.sqrt(np.array(predictions.saq[1])) if predictions.saq else None

            self.logs_298_from_aq = self.calculate_logs_298(logk=self.logk_298,
                                                            logk_ref=self.logk_aq_298,
                                                            logs_ref=self.logs_aq_298)
            self.unc_logs_298_from_aq = self.calculate_logs_298(logk=self.unc_logk_298,
                                                                logk_ref=self.unc_logk_aq_298,
                                                                logs_ref=self.unc_logs_aq_298,
                                                                uncertainty=True)

        if calculate_reference_solvents:
            logger.info('Calculating logS at 298K from reference solubility')
            self.gsolv_ref_298 = np.array(predictions.gsolv_ref[0]) if predictions.gsolv_ref else None  # in kcal/mol
            self.unc_gsolv_ref_298 = np.sqrt(np.array(predictions.gsolv_ref[1])) if predictions.gsolv_ref else None

            self.logk_ref_298 = self.calculate_logk(gsolv=self.gsolv_ref_298)
            self.unc_logk_ref_298 = self.calculate_logk(gsolv=self.unc_gsolv_ref_298, uncertainty=True)

            self.logs_ref_298 = np.array(predictions.data.reference_solubility)
            self.logs_298_from_ref = self.calculate_logs_298(logk=self.logk_298,
                                                             logk_ref=self.logk_ref_298,
                                                             logs_ref=self.logs_ref_298)
            self.unc_logs_298_from_ref = self.calculate_logs_298(logk=self.unc_logk_298,
                                                                 logk_ref=self.unc_logk_ref_298,
                                                                 logs_ref=0.0,
                                                                 uncertainty=True)

        if calculate_t_dep:
            self.hsolv_298 = np.array(predictions.hsolv[0]) if predictions.hsolv else None  # in kcal/mol
            self.unc_hsolv_298 = np.sqrt(np.array(predictions.hsolv[1])) if predictions.hsolv else None

            if predictions.solute_parameters:
                self.E, self.S, self.A, self.B, self.L = self.get_solute_parameters(predictions.solute_parameters[0])
                self.unc_E, self.unc_S, self.unc_A, self.unc_B, self.unc_L = self.get_solute_parameters(predictions.solute_parameters[1])
            else:
                self.E, self.S, self.A, self.B, self.L = None, None, None, None, None
                self.unc_E, self.unc_S, self.unc_A, self.unc_B, self.unc_L = None, None, None, None, None

            self.V = np.array([self.calculate_solute_parameter_v(sm[1]) for sm in predictions.data.smiles_pairs])
            self.I_OHadj, self.I_OHnonadj, self.I_NH = self.get_diol_amine_ids(predictions.data.smiles_pairs)
            self.hsubl_298 = self.get_hsubl_298(self.E, self.S, self.A, self.B, self.V,
                                                I_OHadj=self.I_OHadj,
                                                I_OHnonadj=self.I_OHnonadj,
                                                I_NH=self.I_NH)

            if calculate_aqueous:
                logger.info('Calculating T-dep logS from predicted aqueous solubility using H_solu(298K) approximation')
                self.logs_T_from_aq = self.calculate_logs_t(hsolv_298=self.hsolv_298,
                                                            hsubl_298=self.hsubl_298,
                                                            logs_298=self.logs_298_from_aq,
                                                            temperatures=predictions.data.temperatures)

            if calculate_reference_solvents:
                logger.info('Calculating T-dep logS from reference solubility using H_solu(298K) approximation')
                self.logs_T_from_aq = self.calculate_logs_t(hsolv_298=self.hsolv_298,
                                                            hsubl_298=self.hsubl_298,
                                                            logs_298=self.logs_298_from_ref,
                                                            temperatures=predictions.data.temperatures)

    def calculate_logs_t(self, hsolv_298=None, hsubl_298=None, logs_298=None, temperatures=None):
        hsolu_298 = hsolv_298 + hsubl_298
        return logs_298 - hsolu_298/2.303/8.314*4.184*1000*(1/temperatures-1/298.)

    def calculate_logs_298(self, logk=None, logk_ref=None, logs_ref=None, uncertainty: bool = False):
        if uncertainty:
            return np.abs(logk) + np.abs(logk_ref) + np.abs(logs_ref)
        else:
            return logs_ref + logk - logk_ref

    def calculate_logk(self, gsolv=None, uncertainty: bool = False):
        if uncertainty:
            return np.sqrt(gsolv) * 4.184 * 1000. / 8.314 / 298. / 2.303
        else:
            return -gsolv * 4.184 * 1000. / 8.314 / 298. / 2.303  # in log10

    def get_solute_parameters(self, predictions):
        E, S, A, B, L = [], [], [], [], []
        for i in predictions:
            E.append(i[0])
            S.append(i[1])
            A.append(i[2])
            B.append(i[3])
            L.append(i[4])
        return np.array(E), np.array(S), np.array(A), np.array(B), np.array(L)

    def get_Cp_solid(self, E, S, A, B, V, I_OHnonadj=False, in_cal=True):
        '''
        From Acree. N = 406, SD = 19.0, R 2 = 0.976, F = 1799.2, PRESS = 153,144,
        Q2 = 0.974, PSD = 19.7.
        I_OHnonadj: indicator for aliphatic diols with non-adjacent OH groups. Either True or False.
        Cp at 298.15 K in J/K/mol
        '''
        Cp = 11.63 - 34.18 * E - 1.20 * S - 1.09 * A + 12.28 * B + 181.69 * V \
             + 2.32 * S * S + 4.24 * A * B - 1.85 * V * V - 28.50 * I_OHnonadj
        if in_cal == True:
            Cp = Cp / 4.184  # convert to cal/K/mol
        return Cp

    def get_Cp_gas(self, E, S, A, B, V, in_cal=True):
        '''
        From Acree. N = 1014, SD = 7.86, R2 = 0.994, F = 22,597.7, PRESS = 63,725.7, Q2 =0.994, PSD= 7.96.
        Cp at 298.15 K in J/K/mol
        '''
        Cp = -8.62 - 24.33 * E - 15.83 * S + 12.35 * A + 13.27 * B + 160.00 * V + 10.66 * S * S - 2.11 * A * B + 0.41 * V * V
        if in_cal == True:
            Cp = Cp / 4.184  # convert to cal/K/mol
        return Cp

    def get_hsubl_298(self, E, S, A, B, V, I_OHadj=None, I_OHnonadj=None, I_NH=None, in_kcal=True):
        '''
        From Acree. N = 898, SD = 9.90, R2 = 0.868, F = 528.6, PRESS = 90315.5,
        Q2 = 0.863, PSD = 10.09.
        I_OHadj: indicator for aliphatic diols with adjacent OH groups. Either True or False
        I_OHnonadj: indicator for aliphatic diols with non-adjacent OH groups. Either True or False.
        I_amine: indicator for alkyl amine compounds
        '''
        # the correlation unit is in kJ/mol.
        dHsub = 9.96 - 2.10 * E + 24.10 * S + 13.70 * A + 0.79 * B + 38.71 * V - 1.36 * S * S \
                + 36.90 * A * B + 1.86 * V * V - 10.89 * I_OHadj + 14.74 * I_OHnonadj + 9.69 * I_NH  # kJ/mol
        if in_kcal == True:
            dHsub = dHsub / 4.184  # convert to kcal/mol
        return dHsub

    def calculate_solute_parameter_v(self, solute_smiles):
        V = None
        #Yunsie can you do this
        return 0.0

    def get_diol_amine_ids(self, smiles_pairs):
        solutes = [sm[1] for sm in smiles_pairs]
        unique_solutes = set(solutes)
        dict_diol_amine = dict()
        for i in unique_solutes:
            dict_diol_amine[i] = self.get_individual_diol_amine_ids(i)
        I_OHadj = [1 if dict_diol_amine[i][0] else 0 for i in solutes]
        I_OHnonadj = [1 if dict_diol_amine[i][1] else 0 for i in solutes]
        I_NH = [1 if dict_diol_amine[i][2] else 0 for i in solutes]
        return np.array(I_OHadj), np.array(I_OHnonadj), np.array(I_NH)

    def get_individual_diol_amine_ids(self, solute):
        smarts_aliphatic_OH = ['[C;X4v4]-[O;H1]', '[!O]=[C;X3v4]-[O;H1]']
        mol_OHnon_list = [Chem.MolFromSmarts(i) for i in smarts_aliphatic_OH]

        smarts_aliphatic_adj_OH = ['[O;H1]-[C;X4v4]-[C;X4v4][O;H1]',
                                   '[O;H1]-[C;X3v4](=[!O])-[C;X4v4]-[O;H1]',
                                   '[O;H1]-[C;X3v4](=[!O])-[C;X3v4](=[!O])-[O;H1]']
        mol_OHajd_list = [Chem.MolFromSmarts(i) for i in smarts_aliphatic_adj_OH]

        smarts_aliphatic_amine = ['[C;v4X4]-[N;H1]-[C;v4X4]', '[C;v4X4]-[N;H2]',
                                  '[!O]=[C;v4X3]-[N;H1]-[C;v4X4]',
                                  '[!O]=[C;v4X3]-[N;H1]-[C;v4X3]=[!O]',
                                  '[!O]=[C;v4X3]-[N;H2]']
        mol_amine_list = [Chem.MolFromSmarts(i) for i in smarts_aliphatic_amine]

        mol = Chem.MolFromSmiles(solute)
        mol = Chem.rdmolops.AddHs(mol)

        OH_adj_found = False
        OH_non_found = False
        amine_found = False

        # only consider aliphatic molecules
        # future improvements should include aliphatic side-chain of aromatic molecules
        if len(mol.GetAromaticAtoms()) > 0:
            pass
        else:
            # get OH non match
            OH_non_match_tup = ()
            for mol_template in mol_OHnon_list:
                OH_non_match_tup += mol.GetSubstructMatches(mol_template)
            # get OH adj match
            OH_adj_match_tup = ()
            for mol_template in mol_OHajd_list:
                OH_adj_match_tup += mol.GetSubstructMatches(mol_template)
            # get NH and NH2 match
            amine_match_tup = ()
            for mol_template in mol_amine_list:
                amine_match_tup += mol.GetSubstructMatches(mol_template)

            if len(OH_adj_match_tup) > 0:
                OH_adj_found = True
            else:
                if len(OH_non_match_tup) >= 2:  # make sure they are diols
                    OH_non_found = True
            if len(amine_match_tup) > 0:
                amine_found = True

        return OH_adj_found, OH_non_found, amine_found




