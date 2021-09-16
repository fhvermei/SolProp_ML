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

    def get_gas_liq_sat_density(self, T, Tc, rho_c, coolprop_name=None, ref_solvent='n-Heptane', T_adjustment=True):
        if coolprop_name is None:
            return self.get_gas_liq_sat_density_from_ref(T, Tc, rho_c, ref_solvent=ref_solvent,
                                                         T_adjustment=T_adjustment)
        else:
            return self.get_gas_liq_sat_density_from_cp(T, coolprop_name, T_adjustment=T_adjustment)

    def get_gas_liq_sat_density_from_cp(self, T, coolprop_name, T_adjustment=True, give_T_max_min=False):
        # initialize the output values
        gas_density, liq_density, new_T, error_message = None, None, None, None

        # get the maximum (critical temperature) and minimum temperatures available for CoolProp calculation
        T_max = CP.PropsSI('T_critical', coolprop_name)  # K
        T_min = CP.PropsSI('T_min', coolprop_name)  # K

        # check whether the temperature is in a valid range.
        if T > T_max:
            new_T = T_max - 5
            error_message = f"Temperature {T} K is above or too close to the critical temperature {T_max} K."
        elif T > T_max - 5:
            error_message = f"Warning! Temperature {T} K is too close to the critical temperature {T_max} K and "
            error_message += 'the prediction may not be reliable.'
        elif T < T_min:
            new_T = T_min
            error_message = f"Temperature {T} K must be in range [{T_min} K, {T_max} K]."

        if new_T is None:
            gas_density = CP.PropsSI('Dmolar', 'T', T, 'Q', 1, coolprop_name)  # in mol/m^3
            liq_density = CP.PropsSI('Dmolar', 'T', T, 'Q', 0, coolprop_name)  # in mol/m^3
        elif T_adjustment is True:
            error_message += f" {'%.3f' % new_T} K is used instead for calculation."
            gas_density = CP.PropsSI('Dmolar', 'T', new_T, 'Q', 1, coolprop_name)  # in mol/m^3
            liq_density = CP.PropsSI('Dmolar', 'T', new_T, 'Q', 0, coolprop_name)  # in mol/m^3

        if give_T_max_min:
            return gas_density, liq_density, new_T, error_message, False, T_max, T_min
        else:
            return gas_density, liq_density, new_T, error_message, False

    def get_gas_liq_sat_density_from_ref(self, T, Tc, rho_c, ref_solvent='n-Heptane', T_adjustment=True):
        # convert temperature to reduced temperature and calculate corresponding temperature for the reference solvent
        T_red = T / Tc
        Tc_ref = CP.PropsSI('T_critical', ref_solvent)
        T_ref = T_red * Tc_ref
        # get densities for the reference solvent
        gas_density_ref, liq_density_ref, new_T_ref, error_message, dummy_var, T_max_ref, T_min_ref = \
            self.get_gas_liq_sat_density_from_cp(T_ref, ref_solvent, T_adjustment=True, give_T_max_min=True)
        # convert densities to reduced densities and calculate corresponding densities for the solvent of interest.
        rhoc_ref = CP.PropsSI('rhomolar_critical', ref_solvent)  # mol/m^3
        gas_density_red = gas_density_ref / rhoc_ref
        gas_density = gas_density_red * rho_c  # mol/m^3
        liq_density_red = liq_density_ref / rhoc_ref
        liq_density = liq_density_red * rho_c  # mol/m^3

        # initialize some output values
        new_T, use_const_dHsolv_for_low_T = None, False
        # Update the error message with a valid temperature range
        if new_T_ref is not None:
            new_T_red = new_T_ref / Tc_ref
            new_T = new_T_red * Tc
            if T_adjustment is True:
                # Overwrite the error message
                if new_T > T:
                    error_message = f"Unable to predict dHsoluT for T < {'%.3f' % new_T} K. dHsoluT at {'%.3f' % new_T} K is used instead for lower temperatures."
                    use_const_dHsolv_for_low_T = True
                elif T > new_T:
                    error_message = f"Temperature {T} K is too close to or above the critical temperature {Tc} K."
                    error_message += f" {'%.3f' % new_T} K is used instead for calculation."
                elif T > Tc - 5:
                    error_message = f"Warning! Temperature {T} K is too close to the critical temperature {Tc} K. "
                    error_message += 'The prediction may not be reliable.'
            else:
                # Overwrite the error message and set the output to None
                T_max_red = T_max_ref / Tc_ref
                T_max = T_max_red * Tc
                T_min_red = T_min_ref / Tc_ref
                T_min = T_min_red * Tc
                error_message = f"Temperature {T} K must be in range [{T_min} K, {T_max} K]."
                gas_density = None
                liq_density = None

        return gas_density, liq_density, new_T, error_message, use_const_dHsolv_for_low_T

    def get_Kfactor_parameters(self, gsolv_298, hsolv_298, Tc, rho_c, coolprop_name, T_trans_factor=0.75):
        T1 = 298
        gsolv_298 = gsolv_298 * 4184  # convert from kcal/mol to J/mol
        hsolv_298 = hsolv_298 * 4184  # convert from kcal/mol to J/mol
        dSsolv298 = (hsolv_298 - gsolv_298) / T1
        T_transition = Tc * T_trans_factor  # T_trans_factor is empirically set to 0.75 by default

        # Generate Amatrix and bvector for Ax = b
        Amatrix = np.zeros((4, 4))
        bvec = np.zeros((4, 1))

        # 1. Tr*ln(K-factor) value at T = 298 K
        rho_g_298, rho_l_298, dummy1, dummy2, dummy3 = \
            self.get_gas_liq_sat_density(T1, Tc, rho_c, coolprop_name=coolprop_name, T_adjustment=False)
        K298 = np.exp(gsolv_298 / (T1 * 8.314)) / rho_g_298 * rho_l_298  # K-factor
        x298 = T1 / Tc * np.log(K298)  # Tr*ln(K-factor), in K
        Amatrix[0][0] = 1
        Amatrix[0][1] = (1 - T1 / Tc) ** 0.355
        Amatrix[0][2] = np.exp(1 - T1 / Tc) * (T1 / Tc) ** 0.59
        Amatrix[0][3] = 0
        bvec[0] = x298
        # 2. d(Tr*ln(K-factor)) / dT at T = 298 Use finite difference method to get the temperature gradient from
        # delG, delH, and delS at 298 K
        T2 = T1 + 1
        delG_T2 = hsolv_298 - dSsolv298 * T2
        rho_g_T2, rho_l_T2, dummy1, dummy2, dummy3 = \
            self.get_gas_liq_sat_density(T2, Tc, rho_c, coolprop_name=coolprop_name, T_adjustment=False)
        K_T2 = np.exp(delG_T2 / (T2 * 8.314)) / rho_g_T2 * rho_l_T2
        x_T2 = T2 / Tc * np.log(K_T2)  # Tr*ln(K-factor) at 299 K, in K
        slope298 = (x_T2 - x298) / (T2 - T1)
        Amatrix[1][0] = 0
        Amatrix[1][1] = -0.355 / Tc * ((1 - T1 / Tc) ** (-0.645))
        Amatrix[1][2] = 1 / Tc * np.exp(1 - T1 / Tc) * (0.59 * (T1 / Tc) ** (-0.41) - (T1 / Tc) ** 0.59)
        Amatrix[1][3] = 0
        bvec[1] = slope298
        # 3. Tr*ln(K-factor) continuity at T = T_transition
        rho_g_Ttran, rho_l_Ttran, dummy1, dummy2, dummy3 = \
            self.get_gas_liq_sat_density(T_transition, Tc, rho_c, coolprop_name=coolprop_name, T_adjustment=False)
        Amatrix[2][0] = 1
        Amatrix[2][1] = (1 - T_transition / Tc) ** 0.355
        Amatrix[2][2] = np.exp(1 - T_transition / Tc) * (T_transition / Tc) ** 0.59
        Amatrix[2][3] = -(rho_l_Ttran - rho_c) / rho_c
        bvec[2] = 0
        # 4. d(Tr*ln(K-factor)) / dT smooth transition at T = T_transition
        T3 = T_transition + 1
        rho_g_T3, rho_l_T3, dummy1, dummy2, dummy3 = \
            self.get_gas_liq_sat_density(T3, Tc, rho_c, coolprop_name=coolprop_name, T_adjustment=False)
        Amatrix[3][0] = 0
        Amatrix[3][1] = -0.355 / Tc * ((1 - T_transition / Tc) ** (-0.645))
        Amatrix[3][2] = 1 / Tc * np.exp(1 - T_transition / Tc) * (
                0.59 * (T_transition / Tc) ** (-0.41) - (T_transition / Tc) ** 0.59)
        Amatrix[3][3] = - ((rho_l_T3 - rho_l_Ttran) / rho_c / (T3 - T_transition))
        bvec[3] = 0

        # solve for the parameters
        param, residues, ranks, s = np.linalg.lstsq(Amatrix, bvec, rcond=None)
        # store the results in kfactor_parameters class
        kfactor_parameters = KfactorParameters()
        kfactor_parameters.lower_T = [float(param[0]), float(param[1]), float(param[2])]
        kfactor_parameters.higher_T = float(param[3])
        kfactor_parameters.T_transition = T_transition
        return kfactor_parameters

    def get_Kfactor(self, T, Tc, rho_c, coolprop_name, kfactor_parameters):
        A = kfactor_parameters.lower_T[0]
        B = kfactor_parameters.lower_T[1]
        C = kfactor_parameters.lower_T[2]
        D = kfactor_parameters.higher_T
        T_transition = kfactor_parameters.T_transition
        rho_g, rho_l, dummy1, dummy2, dummy3 = self.get_gas_liq_sat_density(T, Tc, rho_c, coolprop_name=coolprop_name)
        if T < T_transition:
            kfactor = np.exp((A + B * (1 - T / Tc) ** 0.355 + C * np.exp(1 - T / Tc) * (T / Tc) ** 0.59) / (T / Tc))
        else:
            kfactor = np.exp(D * (rho_l / rho_c - 1) / (T / Tc))
        return kfactor


class KfactorParameters:
    """
    Stores the fitted parameters for K-factor calculation
    """
    def __init__(self, A=None, B=None, C=None, D=None, T_transition=None):
        self.lower_T = [A, B, C]
        self.higher_T = D
        self.T_transition = T_transition  # in K