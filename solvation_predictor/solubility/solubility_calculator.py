import os
import json
import numpy as np
import rdkit.Chem as Chem
import CoolProp.CoolProp as CP
from scipy.integrate import quad
from solvation_predictor.solubility.solubility_predictions import SolubilityPredictions
import pkgutil, io


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

    def __init__(self, predictions: SolubilityPredictions = None,
                 calculate_aqueous: bool = None,
                 calculate_reference_solvents: bool = None,
                 calculate_t_dep: bool = None,
                 calculate_t_dep_with_t_dep_hdiss: bool = None,
                 solv_crit_prop_dict: dict = None,
                 hsubl_298: np.array = None,
                 Cp_solid: np.array = None,
                 Cp_gas: np.array = None,
                 logger=None,
                 verbose=True):
        '''
        All uncertainty is reported as the standard deviation of machine learning model ensemble predictions.
        '''

        self.logger = logger.info if logger is not None else print
        self.solv_crit_prop_dict = solv_crit_prop_dict

        self.gsolv_298, self.unc_gsolv_298 = None, None
        self.logk_298, self.unc_logk_298 = None, None
        self.gsolv_aq_298, self.unc_gsolv_aq_298 = None, None
        self.logk_aq_298, self.unc_logk_aq_298 = None, None
        self.logs_aq_298, self.unc_logs_aq_298 = None, None
        self.logs_298_from_aq, self.unc_logs_298_from_aq = None, None
        self.gsolv_ref_298, self.unc_gsolv_ref_298 = None, None
        self.logk_ref_298, self.unc_logk_ref_298 = None, None
        self.logs_ref_298 = None
        self.logs_298_from_ref, self.unc_logs_298_from_ref = None, None
        self.hsolv_298, self.unc_hsolv_298 = None, None

        self.E, self.S, self.A, self.B, self.L = None, None, None, None, None
        self.unc_E, self.unc_S, self.unc_A, self.unc_B, self.unc_L = None, None, None, None, None
        self.V = None
        self.I_OHadj, self.I_OHnonadj, self.I_NH = None, None, None
        self.hsubl_298 = hsubl_298 if hsubl_298 is not None else None
        self.Cp_solid = Cp_solid if Cp_solid is not None else None
        self.Cp_gas = Cp_gas if Cp_gas is not None else None
        self.logs_T_with_const_hdiss_from_aq, self.logs_T_with_T_dep_hdiss_from_aq = None, None
        self.logs_T_with_const_hdiss_warning_message, self.logs_T_with_T_dep_hdiss_error_message = None, None
        self.hsolv_T, self.gsolv_T, self.ssolv_T = None, None, None
        self.logs_T_with_const_hdiss_from_ref, self.logs_T_with_T_dep_hdiss_from_ref = None, None

        if predictions is not None:
            if verbose:
                self.logger('Start making logS calculations')
            self.make_calculations_298(predictions=predictions,
                                       calculate_aqueous=calculate_aqueous,
                                       calculate_reference_solvents=calculate_reference_solvents,
                                       verbose=verbose)
            if calculate_t_dep:
                self.make_calculations_t(predictions=predictions,
                                         calculate_aqueous=calculate_aqueous,
                                         calculate_reference_solvents=calculate_reference_solvents,
                                         calculate_t_dep_with_t_dep_hdiss=calculate_t_dep_with_t_dep_hdiss,
                                         verbose=verbose)

    def make_calculations_298(self, predictions: SolubilityPredictions,
                              calculate_aqueous: bool = None,
                              calculate_reference_solvents: bool = None,
                              verbose=False):

        self.gsolv_298, self.unc_gsolv_298 = self.extract_predictions(predictions.gsolv)
        self.logk_298 = self.calculate_logk(gsolv=self.gsolv_298)
        self.unc_logk_298 = self.calculate_logk(gsolv=self.unc_gsolv_298, uncertainty=True)

        if calculate_aqueous:
            if verbose:
                self.logger('Calculating logS at 298K from predicted aqueous solubility')
            self.gsolv_aq_298,  self.unc_gsolv_aq_298 = self.extract_predictions(predictions.gsolv_aq)
            self.logk_aq_298 = self.calculate_logk(gsolv=self.gsolv_aq_298)
            self.unc_logk_aq_298 = self.calculate_logk(gsolv=self.unc_gsolv_aq_298, uncertainty=True)

            self.logs_aq_298, self.unc_logs_aq_298 = self.extract_predictions(predictions.saq)
            self.logs_298_from_aq = self.calculate_logs_298(logk=self.logk_298,
                                                            logk_ref=self.logk_aq_298,
                                                            logs_ref=self.logs_aq_298)
            self.unc_logs_298_from_aq = self.calculate_logs_298(logk=self.unc_logk_298,
                                                                logk_ref=self.unc_logk_aq_298,
                                                                logs_ref=self.unc_logs_aq_298,
                                                                uncertainty=True)

        if calculate_reference_solvents:
            if verbose:
                self.logger('Calculating logS at 298K from reference solubility')
            self.gsolv_ref_298, self.unc_gsolv_ref_298 = self.extract_predictions(predictions.gsolv_ref)

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

    def make_calculations_t(self, predictions: SolubilityPredictions,
                            calculate_aqueous: bool = None,
                            calculate_reference_solvents: bool = None,
                            calculate_t_dep_with_t_dep_hdiss: bool = None,
                            verbose=False):

        self.hsolv_298, self.unc_hsolv_298 = self.extract_predictions(predictions.hsolv)

        if predictions.solute_parameters:
            self.E, self.S, self.A, self.B, self.L = self.get_solute_parameters(predictions.solute_parameters[0])
            self.unc_E, self.unc_S, self.unc_A, self.unc_B, self.unc_L = self.get_solute_parameters(predictions.solute_parameters[1])

        self.V = np.array([self.calculate_solute_parameter_v(sm[1]) for sm in predictions.data.smiles_pairs])
        self.I_OHadj, self.I_OHnonadj, self.I_NH = self.get_diol_amine_ids(predictions.data.smiles_pairs)
        if self.hsubl_298 is None:
            self.hsubl_298 = self.get_hsubl_298(self.E, self.S, self.A, self.B, self.V,
                                                I_OHadj=self.I_OHadj,
                                                I_OHnonadj=self.I_OHnonadj,
                                                I_NH=self.I_NH)
        self.logs_T_with_const_hdiss_warning_message = self.get_logs_t_with_const_hdiss_warning_message(
            temperatures=predictions.data.temperatures)

        if calculate_t_dep_with_t_dep_hdiss:
            if self.Cp_solid is None:
                self.Cp_solid = self.get_Cp_solid(self.E, self.S, self.A, self.B, self.V,
                                                  I_OHnonadj=self.I_OHnonadj)  # in cal/mol/K
            if self.Cp_gas is None:
                self.Cp_gas = self.get_Cp_gas(self.E, self.S, self.A, self.B, self.V)  # in cal/mol/K

            # load solvent's CoolProp name, critical temperature, and critical density data
            # if the solvent critical property dictionary (solv_crit_prop_dict) is not provided, use the default one.
            if self.solv_crit_prop_dict is None:
                #load from package
                #current_path = os.path.dirname(os.path.abspath(__file__))
                #crit_data_path = os.path.join(current_path, 'solvent_crit_data.json')
                #with open(crit_data_path) as f:
                #    self.solv_crit_prop_dict = json.load(f)  # inchi with fixed H is used as a solvent key
                path = os.path.join('solubility', 'solvent_crit_data.json')
                crit_data_path = io.BytesIO(pkgutil.get_data('solvation_predictor', path))
                self.solv_crit_prop_dict = json.load(crit_data_path)  # inchi with fixed H is used as a solvent key

            coolprop_name_list, crit_t_list, crit_d_list = \
                self.get_solvent_info(predictions.data.smiles_pairs, self.solv_crit_prop_dict)

        if calculate_aqueous:
            if verbose:
                self.logger('Calculating T-dep logS from predicted aqueous solubility using H_diss(298K) approximation')
            self.logs_T_with_const_hdiss_from_aq = self.calculate_logs_t(hsolv_298=self.hsolv_298,
                                                        hsubl_298=self.hsubl_298,
                                                        logs_298=self.logs_298_from_aq,
                                                        temperatures=predictions.data.temperatures)
            if calculate_t_dep_with_t_dep_hdiss:
                if verbose:
                    self.logger('Calculating T-dep logS from predicted aqueous solubility using T-dep H_diss')
                self.logs_T_with_T_dep_hdiss_from_aq, self.logs_T_with_T_dep_hdiss_error_message, self.hsolv_T,\
                    self.gsolv_T, self.ssolv_T = self.calculate_logs_t_with_t_dep_hdiss_all(
                    gsolv_298_list=self.gsolv_298, hsolv_298_list=self.hsolv_298, hsubl_298_list=self.hsubl_298,
                    Cp_solid_list=self.Cp_solid, Cp_gas_list=self.Cp_gas, logs_298_list=self.logs_298_from_aq,
                    T_list=predictions.data.temperatures, coolprop_name_list=coolprop_name_list,
                    Tc_list=crit_t_list, rho_c_list=crit_d_list)

        if calculate_reference_solvents:
            if verbose:
                self.logger('Calculating T-dep logS from reference solubility using H_diss(298K) approximation')
            self.logs_T_with_const_hdiss_from_ref = self.calculate_logs_t(hsolv_298=self.hsolv_298,
                                                         hsubl_298=self.hsubl_298,
                                                         logs_298=self.logs_298_from_ref,
                                                         temperatures=predictions.data.temperatures)
            if calculate_t_dep_with_t_dep_hdiss:
                if verbose:
                    self.logger('Calculating T-dep logS from reference solubility using T-dep H_diss')
                # since `logs_T_with_T_dep_hdiss_error_message` and `hsolv_T` will be the same whether aqueous
                # or reference solubility is used, these can be overwritten.
                self.logs_T_with_T_dep_hdiss_from_ref, self.logs_T_with_T_dep_hdiss_error_message, self.hsolv_T,\
                    self.gsolv_T, self.ssolv_T= self.calculate_logs_t_with_t_dep_hdiss_all(
                    gsolv_298_list=self.gsolv_298, hsolv_298_list=self.hsolv_298, hsubl_298_list=self.hsubl_298,
                    Cp_solid_list=self.Cp_solid, Cp_gas_list=self.Cp_gas, logs_298_list=self.logs_298_from_ref,
                    T_list=predictions.data.temperatures, coolprop_name_list=coolprop_name_list,
                    Tc_list=crit_t_list, rho_c_list=crit_d_list)

    def extract_predictions(self, predictions):
        pred = np.array(predictions[0]) if predictions else None
        unc = np.sqrt(np.array(predictions[1])) if predictions else None  # uncertainty reported as standard deviation
        return pred, unc

    def calculate_logs_t(self, hsolv_298=None, hsubl_298=None, logs_298=None, temperatures=None):
        hdiss_298 = hsolv_298 + hsubl_298
        return logs_298 - hdiss_298/2.303/8.314*4.184*1000*(1/temperatures-1/298.)

    def get_logs_t_with_const_hdiss_warning_message(self, temperatures=None):
        warning_message = ['Warning! Above 350 K, `calculate_t_dep_with_t_dep_hdiss` option is recommended.'
                           if temp > 350 else '' for temp in temperatures]
        return warning_message

    def calculate_logs_298(self, logk=None, logk_ref=None, logs_ref=None, uncertainty: bool = False):
        if uncertainty:
            return np.sqrt(np.square(logk) + np.square(logk_ref) + np.square(logs_ref))
        else:
            return logs_ref + logk - logk_ref

    def calculate_logk(self, gsolv=None, uncertainty: bool = False):
        if uncertainty:
            return np.abs(gsolv * 4.184 * 1000. / 8.314 / 298. / 2.303)
        else:
            return -gsolv * 4.184 * 1000. / 8.314 / 298. / 2.303  # in log10

    def get_solute_parameters(self, predictions, uncertainty: bool = False):
        E, S, A, B, L = [], [], [], [], []
        for i in predictions:
            E.append(i[0])
            S.append(i[1])
            A.append(i[2])
            B.append(i[3])
            L.append(i[4])
        if uncertainty:
            return np.sqrt(np.array(E)), np.sqrt(np.array(S)), np.sqrt(np.array(A)), \
                   np.sqrt(np.array(B)), np.sqrt(np.array(L))
        else:
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

    def get_solvent_info(self, smiles_pairs, solv_crit_prop_dict):
        solvents_smiles_list = [sm[0] for sm in smiles_pairs]
        coolprop_name_list, crit_t_list, crit_d_list = [], [], []
        for smi in solvents_smiles_list:
            mol = Chem.MolFromSmiles(smi)
            inchi = Chem.MolToInchi(mol, options='/FixedH')
            if inchi in solv_crit_prop_dict:
                coolprop_name_list.append(solv_crit_prop_dict[inchi]['coolprop_name'])
                crit_t_list.append(solv_crit_prop_dict[inchi]['Tc'])  # in K
                crit_d_list.append(solv_crit_prop_dict[inchi]['rho_c'])  # in mol/m^3
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
        const_hdiss_T = None
        error_message = None
        if T > T_max:
            error_message = f"Temperature {T} K is above the critical temperature {T_max} K."
            valid = False
        elif T > T_max - 15:
            error_message = f"Warning! Temperature {T} K is too close to the critical temperature {T_max} K."
            error_message += ' The prediction may not be reliable.'
        elif T < T_min:
            const_hdiss_T = T_min
            if coolprop_name is None:
                error_message = f"Unable to predict dHdissT for T < {'%.3f' % T_min} K. dHdissT at {'%.3f' % T_min} K is used instead for lower temperatures."
            else:
                error_message = f"Warning! Temperature {T} K is below the minimum limit. It should be in range [{T_min} K, {T_max} K]."
                error_message += f" Constant dHdissT at {'%.3f' % T_min} K is used instead for lower temperatures."
        return valid, const_hdiss_T, error_message, T_min, T_max

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
        rho_g_298, rho_l_298 = self.get_gas_liq_sat_density(T1, Tc, rho_c, coolprop_name=coolprop_name)
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
        rho_g_T2, rho_l_T2, = self.get_gas_liq_sat_density(T2, Tc, rho_c, coolprop_name=coolprop_name)
        K_T2 = np.exp(delG_T2 / (T2 * 8.314)) / rho_g_T2 * rho_l_T2
        x_T2 = T2 / Tc * np.log(K_T2)  # Tr*ln(K-factor) at 299 K, in K
        slope298 = (x_T2 - x298) / (T2 - T1)
        Amatrix[1][0] = 0
        Amatrix[1][1] = -0.355 / Tc * ((1 - T1 / Tc) ** (-0.645))
        Amatrix[1][2] = 1 / Tc * np.exp(1 - T1 / Tc) * (0.59 * (T1 / Tc) ** (-0.41) - (T1 / Tc) ** 0.59)
        Amatrix[1][3] = 0
        bvec[1] = slope298
        # 3. Tr*ln(K-factor) continuity at T = T_transition
        rho_g_Ttran, rho_l_Ttran = self.get_gas_liq_sat_density(T_transition, Tc, rho_c, coolprop_name=coolprop_name)
        Amatrix[2][0] = 1
        Amatrix[2][1] = (1 - T_transition / Tc) ** 0.355
        Amatrix[2][2] = np.exp(1 - T_transition / Tc) * (T_transition / Tc) ** 0.59
        Amatrix[2][3] = -(rho_l_Ttran - rho_c) / rho_c
        bvec[2] = 0
        # 4. d(Tr*ln(K-factor)) / dT smooth transition at T = T_transition
        T3 = T_transition + 1
        rho_g_T3, rho_l_T3 = self.get_gas_liq_sat_density(T3, Tc, rho_c, coolprop_name=coolprop_name)
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

    def get_t_dep_gsolv_kfactor(self, T, Tc, rho_c, coolprop_name, kfactor_parameters, only_gsolv=True):
        A = kfactor_parameters.lower_T[0]
        B = kfactor_parameters.lower_T[1]
        C = kfactor_parameters.lower_T[2]
        D = kfactor_parameters.higher_T
        T_transition = kfactor_parameters.T_transition
        rho_g, rho_l = self.get_gas_liq_sat_density(T, Tc, rho_c, coolprop_name=coolprop_name)
        if T < T_transition:
            kfactor = np.exp((A + B * (1 - T / Tc) ** 0.355 + C * np.exp(1 - T / Tc) * (T / Tc) ** 0.59) / (T / Tc))
        else:
            kfactor = np.exp(D * (rho_l / rho_c - 1) / (T / Tc))
        gsolv = 8.314 * T * np.log(kfactor * rho_g / (rho_l)) / 4184  # in kcal/mol
        if only_gsolv:
            return gsolv
        else:
            return gsolv, kfactor

    def get_t_dep_hsolv(self, T, Tc, rho_c, coolprop_name, kfactor_parameters, T_min, T_max):
        # get the list of temperatures to evaluate gsolv
        if T - 1 < T_min:
            T_list = [T, T+1]
        elif T + 1 > T_max:
            T_list = [T, T-1]
        else:
            T_list = [T-1, T, T+1]
        gsolv_list = [self.get_t_dep_gsolv_kfactor(T, Tc, rho_c, coolprop_name, kfactor_parameters) for T in T_list]
        # estimate hsolv using finite difference
        if len(T_list) == 2:
            gsolv_T = gsolv_list[0]
            ssolv_T = - (gsolv_list[1] - gsolv_T) / (T_list[1] - T_list[0])
        else:
            gsolv_T = gsolv_list[1]
            ssolv_T = - (gsolv_list[2] - gsolv_list[0]) / (T_list[2] - T_list[0])
        hsolv_T = gsolv_T + T * ssolv_T
        return hsolv_T, gsolv_T, ssolv_T  # in kcal/mol, kcal/mol, kcal/K/mol

    def hsolv_integrand(self, T, Tc, rho_c, coolprop_name, kfactor_parameters, T_min, T_max):
        hsolv_T, _, _ = self.get_t_dep_hsolv(T, Tc, rho_c, coolprop_name, kfactor_parameters, T_min, T_max)  # in kcal/mol
        integrand = hsolv_T * 4.184 * 1000. / (8.314 * T ** 2)
        return integrand

    def integrate_hsolv_term(self, T, const_hdiss_T, Tc, rho_c, coolprop_name, kfactor_parameters, T_min, T_max):
        if const_hdiss_T is not None:
            T_for_integral = const_hdiss_T
        else:
            T_for_integral = T
        # do numerical integral from 298 K to T_for_integral
        I = quad(self.hsolv_integrand, 298, T_for_integral,
                 args=(Tc, rho_c, coolprop_name, kfactor_parameters, T_min, T_max))
        hsolv_integral = I[0]
        # Use a constant hsolvT at const_hdiss_T for T < const_hdiss_T if const_hdiss_T is not None
        if const_hdiss_T is not None:
            hsolv_T, _, _ = self.get_t_dep_hsolv(const_hdiss_T, Tc, rho_c, coolprop_name, kfactor_parameters, T_min, T_max)
            hsolv_integral += (- hsolv_T * 4.184 * 1000.) / 8.314 * (1 / T - 1 / const_hdiss_T)
        return hsolv_integral

    def integrate_t_dep_hdiss(self, hsolv_integral, hsubl_298, Cp_solid, Cp_gas, T):
        '''
        hsubl_298 in kcal/mol. Cp_solid and Cp_gas in cal/K/mol. T in K.
        '''
        hsubl_298 = hsubl_298 * 4.184 * 1000  # convert from kcal/mol to J/mol
        Cp_solid = Cp_solid * 4.184  # convert from cal/mol/K to J/mol/K
        Cp_gas = Cp_gas * 4.184  # convert from cal/mol/K to J/mol/K
        hsub_integral = (- hsubl_298) / 8.314 * (1 / T - 1 / 298)
        Cpsolid_integral = (-Cp_solid * 298) / 8.314 * (1 / T - 1 / 298) \
                           + (-Cp_solid) / 8.314 * np.log(T / 298)
        Cpgas_integral = (Cp_gas * 298) / 8.314 * (1 / T - 1 / 298) \
                         + (Cp_gas) / 8.314 * np.log(T / 298)
        total_integral = hsolv_integral + hsub_integral + Cpgas_integral + Cpsolid_integral
        return total_integral

    def calculate_logs_t_with_t_dep_hdiss(self, gsolv_298=None, hsolv_298=None, hsubl_298=None, Cp_solid=None,
                                          Cp_gas=None, logs_298=None, T=None, coolprop_name=None, Tc=None, rho_c=None):

        # check whether the given temperature is valid
        valid, const_hdiss_T, error_message, T_min, T_max = self.check_valid_t(T, Tc, coolprop_name=coolprop_name)

        if valid is False:
            return None, error_message, None, None, None

        kfactor_parameters = self.get_Kfactor_parameters(gsolv_298, hsolv_298, Tc, rho_c, coolprop_name)
        hsolv_integral = self.integrate_hsolv_term(T, const_hdiss_T, Tc, rho_c, coolprop_name, kfactor_parameters,
                                                   T_min, T_max)
        total_integral = self.integrate_t_dep_hdiss(hsolv_integral, hsubl_298, Cp_solid, Cp_gas, T)
        logs_t = logs_298 + total_integral / 2.303
        # get hsolv_T
        if const_hdiss_T is None:
            hsolv_T, gsolv_T, ssolv_T = self.get_t_dep_hsolv(T, Tc, rho_c, coolprop_name, kfactor_parameters,
                                                             T_min, T_max)
        else:
            hsolv_T, gsolv_T, ssolv_T = self.get_t_dep_hsolv(const_hdiss_T, Tc, rho_c, coolprop_name,
                                                             kfactor_parameters, T_min, T_max)
        return logs_t, error_message, hsolv_T, gsolv_T, ssolv_T

    def calculate_logs_t_with_t_dep_hdiss_all(self, gsolv_298_list=None, hsolv_298_list=None, hsubl_298_list=None,
                                              Cp_solid_list=None, Cp_gas_list=None, logs_298_list=None, T_list=None,
                                              coolprop_name_list=None, Tc_list=None, rho_c_list=None):

        logs_t_list, error_message_list, hsolv_t_list, gsolv_t_list, ssolv_t_list = [], [], [], [], []
        for i in range(len(gsolv_298_list)):
            if Tc_list[i] is None or rho_c_list is None:
                logs_t_list.append(None)
                error_message_list.append('The given solvent is not supported. Its critical temperature and'
                                          ' density are required for calculation.')
                hsolv_t_list.append(None)
                gsolv_t_list.append(None)
                ssolv_t_list.append(None)
            else:
                logs_t, error_message, hsolv_t, gsolv_T, ssolv_T = self.calculate_logs_t_with_t_dep_hdiss(
                    gsolv_298=gsolv_298_list[i], hsolv_298=hsolv_298_list[i], hsubl_298=hsubl_298_list[i],
                    Cp_solid=Cp_solid_list[i], Cp_gas=Cp_gas_list[i], logs_298=logs_298_list[i],
                    T=T_list[i], coolprop_name=coolprop_name_list[i], Tc=Tc_list[i], rho_c=rho_c_list[i])
                logs_t_list.append(logs_t)
                error_message_list.append(error_message)
                hsolv_t_list.append(hsolv_t)
                gsolv_t_list.append(gsolv_T)
                ssolv_t_list.append(ssolv_T)
        return logs_t_list, error_message_list, hsolv_t_list, gsolv_t_list, ssolv_t_list


class KfactorParameters:
    """
    Stores the fitted parameters for K-factor calculation
    """
    def __init__(self, A=None, B=None, C=None, D=None, T_transition=None):
        self.lower_T = [A, B, C]
        self.higher_T = D
        self.T_transition = T_transition  # in K