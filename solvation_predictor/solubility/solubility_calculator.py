import json
import numpy as np
import rdkit.Chem as Chem
import CoolProp.CoolProp as CP
from solvation_predictor.solubility.solubility_predictions import SolubilityPredictions


class SolubilityCalculations:

    # Set class variable with McGowan volumes for each atomic number
    mcgowan_volumes = {
        1: 8.71, 2: 6.75,
        3: 22.23, 4: 20.27, 5: 18.31, 6: 16.35, 7: 14.39, 8: 12.43, 9: 10.47, 10: 8.51,
        11: 32.71, 12: 30.75, 13: 28.79, 14: 26.83, 15: 24.87, 16: 22.91, 17: 20.95, 18: 18.99,
        19: 51.89, 20: 50.28, 21: 48.68, 22: 47.07, 23: 45.47, 24: 43.86, 25: 42.26, 26: 40.65, 27: 39.05,
        28: 37.44, 29: 35.84, 30: 34.23, 31: 32.63, 32: 31.02, 33: 29.42, 34: 27.81, 35: 26.21, 36: 24.60,
        37: 60.22, 38: 58.61, 39: 57.01, 40: 55.40, 41: 53.80, 42: 52.19, 43: 50.59, 44: 48.98, 45: 47.38,
        46: 45.77, 47: 44.17, 48: 42.56, 49: 40.96, 50: 39.35, 51: 37.75, 52: 36.14, 53: 34.54, 54: 32.93,
        55: 77.25, 56: 76.00, 57: 74.75, 72: 55.97, 73: 54.71, 74: 53.46, 75: 52.21, 76: 50.96, 77: 49.71,
        78: 48.45, 79: 47.20, 80: 45.95, 81: 44.70, 82: 43.45, 83: 42.19, 84: 40.94, 85: 39.69, 86: 38.44,
    }

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
            self.Cp_solid = self.get_Cp_solid(self.E, self.S, self.A, self.B, self.V,
                                              I_OHnonadj=self.I_OHnonadj)
            self.Cp_gas = self.get_Cp_gas(self.E, self.S, self.A, self.B, self.V)

            # load solvent's CoolProp name, critical temperature, and critical density data
            with open('solvent_crit_data.json') as f:
                self.solv_info_dict = json.load(f)  # inchi is used as a solvent key

            self.coolprop_name_list, self.crit_t_list, self.crit_d_list = \
                self.get_solvent_info(predictions.data.smiles_pairs, self.solv_info_dict)

            if calculate_aqueous:
                logger.info('Calculating T-dep logS from predicted aqueous solubility using H_solu(298K) approximation')
                self.logs_T_from_aq = self.calculate_logs_t(hsolv_298=self.hsolv_298,
                                                            hsubl_298=self.hsubl_298,
                                                            logs_298=self.logs_298_from_aq,
                                                            temperatures=predictions.data.temperatures)

            if calculate_reference_solvents:
                logger.info('Calculating T-dep logS from reference solubility using H_solu(298K) approximation')
                self.logs_T_from_ref = self.calculate_logs_t(hsolv_298=self.hsolv_298,
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
        mol = Chem.MolFromSmiles(solute_smiles)
        mol = Chem.rdmolops.AddHs(mol)
        V_tot = 0.0
        for atom in mol.GetAtoms():
            try:
                V_tot += self.mcgowan_volumes[atom.GetAtomicNum()]
            except KeyError:
                raise ValueError('McGowan volume not available for element {}'.format(atom.GetAtomicNum()))
            # divide contribution in half since all bonds would be counted twice this way
            V_tot -= len(atom.GetBonds()) * 6.56 / 2
        return V_tot / 100  # division by 100 to get units correct

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

    def get_solvent_info(self, smiles_pairs, solv_info_dict):
        solvents_smiles_list = [sm[0] for sm in smiles_pairs]
        coolprop_name_list, crit_t_list, crit_d_list = [], [], []
        for smi in solvents_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            inchi = Chem.MolToInchi(mol, options='/FixedH')
            if inchi in solv_info_dict:
                coolprop_name_list.append(solv_info_dict[inchi]['coolprop_name'])
                crit_t_list.append(solv_info_dict[inchi]['Tc'])  # in K
                crit_d_list.append(solv_info_dict[inchi]['rho_c'])  # in mol/m^3
            else:
                coolprop_name_list.append(None)
                crit_t_list.append(None)
                crit_d_list.append(None)
        return coolprop_name_list, crit_t_list, crit_d_list

    def check_valid_t(self, T, Tc, coolprop_name=None, ref_solvent='n-Heptane'):
        if coolprop_name is None:
            Tc_ref = CP.PropsSI('T_critical', ref_solvent)  # in K
            T_min_ref = CP.PropsSI('T_min', ref_solvent)
            T_max = Tc
            T_min_red = T_min_ref / Tc_ref
            T_min = T_min_red * Tc
        else:
            T_max = CP.PropsSI('T_critical', coolprop_name)
            T_min = CP.PropsSI('T_min', coolprop_name)

        valid = True
        const_hsolu_T = None
        error_message = None
        if T > T_max:
            error_message = f"Temperature {T} K is above the critical temperature {T_max} K."
            valid = False
        elif T > T_max - 15:
            error_message = f"Warning! Temperature {T} K is too close to the critical temperature {T_max} K."
            error_message += ' The prediction may not be reliable.'
        elif T < T_min:
            const_hsolu_T = T_min
            if coolprop_name is None:
                error_message = f"Unable to predict dHsoluT for T < {'%.3f' % T_min} K. dHsoluT at {'%.3f' % T_min} K is used instead for lower temperatures."
            else:
                error_message = f"Warning! Temperature {T} K is below the minimum limit. It should be in range [{T_min} K, {T_max} K]."
                error_message += f" Constant dHsoluT at {'%.3f' % T_min} K is used instead for lower temperatures."
        return valid, const_hsolu_T, error_message, T_min, T_max

    def get_gas_liq_sat_density(self, T, Tc, rho_c, coolprop_name=None, ref_solvent='n-Heptane'):
        if coolprop_name is None:
            return self.get_gas_liq_sat_density_from_ref(T, Tc, rho_c, ref_solvent=ref_solvent)
        else:
            return self.get_gas_liq_sat_density_from_cp(T, coolprop_name)

    def get_gas_liq_sat_density_from_cp(self, T, coolprop_name):
        gas_density = CP.PropsSI('Dmolar', 'T', T, 'Q', 1, coolprop_name)  # in mol/m^3
        liq_density = CP.PropsSI('Dmolar', 'T', T, 'Q', 0, coolprop_name)  # in mol/m^3
        return gas_density, liq_density

    def get_gas_liq_sat_density_from_ref(self, T, Tc, rho_c, ref_solvent='n-Heptane'):
        # convert temperature to reduced temperature and then calculate corresponding temperature for the reference solvent
        T_red = T / Tc
        Tc_ref = CP.PropsSI('T_critical', ref_solvent)  # K
        T_ref = T_red * Tc_ref
        # get densities for the reference solvent
        gas_density_ref, liq_density_ref = self.get_gas_liq_sat_density_from_cp(T_ref, ref_solvent)
        # convert densities to reduced densities and then calculate corresponding densities for the solvent of interest.
        rhoc_ref = CP.PropsSI('rhomolar_critical', ref_solvent)  # mol/m^3
        gas_density_red = gas_density_ref / rhoc_ref
        gas_density = gas_density_red * rho_c  # mol/m^3
        liq_density_red = liq_density_ref / rhoc_ref
        liq_density = liq_density_red * rho_c  # mol/m^3
        return gas_density, liq_density