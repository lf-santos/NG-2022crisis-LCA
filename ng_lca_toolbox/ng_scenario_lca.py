import time
import brightway2 as bw
import numpy as np
import pandas as pd
import scipy
import copy


# NgScenarioLCA Class definition
class NgScenarioLCA():

    def __init__(self, base_data_dict, ei_ng_db=None, no_double_counting=False):
        # saving the database in the class <- check the memory usage of this
        self.ei_ng_db = ei_ng_db if ei_ng_db is not None else []
        self.no_double_counting = no_double_counting
        
        # TOTAL VALUES
        self.ELECTRICITY = base_data_dict['ELECTRICITY']
        self.HEATPOWER = base_data_dict['HEATPOWER']
        self.HEATPOWER_HEAT = base_data_dict['HEATPOWER_HEAT']
        self.HEAT_INDUSTRY = base_data_dict['HEAT_INDUSTRY']
        self.HEAT_HOUSEHOLDS = base_data_dict['HEAT_HOUSEHOLDS']

        # ELECTRICITY:
        self.NG_POWER = base_data_dict['NG_POWER']
        self.COAL_POWER = base_data_dict['COAL_POWER']
        self.NUCLEAR_POWER = base_data_dict['NUCLEAR_POWER']
        self.WIND_POWER = base_data_dict['WIND_POWER']
        self.SOLAR_POWER = base_data_dict['SOLAR_POWER']
        self.BIOMASS_POWER = base_data_dict['BIOMASS_POWER']
        self.OIL_POWER = base_data_dict['OIL_POWER']
        self.LIGNITE_POWER = base_data_dict['LIGNITE_POWER']
        self.HYDRO_POWER = base_data_dict['HYDRO_POWER']
        self.SAVINGS_POWER = base_data_dict['SAVINGS_POWER']

        # CHP ELECTRICITY:
        self.NG_CHP = base_data_dict['NG_CHP']
        self.COAL_CHP = base_data_dict['COAL_CHP']
        self.BIOMASS_CHP = base_data_dict['BIOMASS_CHP']
        self.OIL_CHP = base_data_dict['OIL_CHP']
        self.LIGNITE_CHP = base_data_dict['LIGNITE_CHP']
        
        # CHP:
        self.NG_CHP_HEAT = base_data_dict['NG_HEAT_CHP']
        self.COAL_CHP_HEAT = base_data_dict['COAL_HEAT_CHP']
        self.BIOMASS_CHP_HEAT = base_data_dict['BIOMASS_HEAT_CHP']
        self.OIL_CHP_HEAT = base_data_dict['OIL_HEAT_CHP']
        self.LIGNITE_CHP_HEAT = base_data_dict['LIGNITE_HEAT_CHP']
        
        # HEAT INDUSTRY:
        self.NG_INDUSTRY = base_data_dict['NG_INDUSTRY']
        self.COAL_INDUSTRY = base_data_dict['COAL_INDUSTRY']
        self.OIL_INDUSTRY = base_data_dict['OIL_INDUSTRY']
        self.EFFICIENCY_INDUSTRY = base_data_dict['EFFICIENCY_INDUSTRY']
        self.SAVINGS_INDUSTRY = base_data_dict['SAVINGS_INDUSTRY']

        # HEAT HH&S:
        self.NG_HOUSEHOLDS = base_data_dict['NG_HOUSEHOLDS']
        self.ELECTRICITY_HOUSEHOLDS = base_data_dict['ELECTRICITY_HOUSEHOLDS']
        self.WEATHER_HOUSEHOLDS = base_data_dict['WEATHER_HOUSEHOLDS']
        self.EFFICIENCY_HOUSEHOLDS = base_data_dict['EFFICIENCY_HOUSEHOLDS']
        self.SAVINGS_HOUSEHOLDS = base_data_dict['SAVINGS_HOUSEHOLDS']
        
        # NG SUPPLY MIX
        self.NG_DE = base_data_dict['NG_DE']
        self.NG_NL = base_data_dict['NG_NL']
        self.NG_RO = base_data_dict['NG_RO']
        self.NG_NO = base_data_dict['NG_NO']
        self.NG_RU = base_data_dict['NG_RU']
        self.NG_DZ = base_data_dict['NG_DZ']
        self.NG_AZ = base_data_dict['NG_AZ']
        self.LNG_US = base_data_dict['LNG_US']
        self.LNG_QA = base_data_dict['LNG_QA']
        self.LNG_RU = base_data_dict['LNG_RU']
        self.LNG_NG = base_data_dict['LNG_NG']
        self.LNG_DZ = base_data_dict['LNG_DZ']

        self.TOTAL_ENERGY = self.ELECTRICITY + self.HEATPOWER + self.HEAT_HOUSEHOLDS * 3.6 +\
                self.HEATPOWER_HEAT * 3.6 + self.HEAT_INDUSTRY * 3.6
        self.TOTAL_ELECTRICITY = self.ELECTRICITY + self.HEATPOWER
        total_electricity_sources = self.NG_POWER + self.COAL_POWER + self.NUCLEAR_POWER + self.WIND_POWER +\
               self.SOLAR_POWER + self.OIL_POWER + self.LIGNITE_POWER + self.BIOMASS_POWER +\
               self.NG_CHP + self.COAL_CHP + self.BIOMASS_CHP + self.OIL_CHP + self.LIGNITE_CHP +\
               self.HYDRO_POWER + self.SAVINGS_POWER
        if abs(total_electricity_sources - self.TOTAL_ELECTRICITY) > 1e-3:
            raise Exception(f"Electricity available of {total_electricity_sources} does not match with {self.TOTAL_ELECTRICITY}")

        if abs(self.NG_HOUSEHOLDS + self.SAVINGS_HOUSEHOLDS + self.ELECTRICITY_HOUSEHOLDS +\
               self.WEATHER_HOUSEHOLDS + self.EFFICIENCY_HOUSEHOLDS - self.HEAT_HOUSEHOLDS) > 1e-3:
            raise Exception(f"Heat available to households does not match with {self.HEAT_HOUSEHOLDS}")

        self.energy_from_ng = [act for act in ei_ng_db if 'energy from NG' in act['name']][0]
        self.heat_from_ng = [act for act in ei_ng_db if 'heat from NG, CHP' in act['name']][0]
        self.elec_from_ng = [act for act in ei_ng_db if 'electricity from NG' in act['name']][0]
        self.heat_industry = [act for act in ei_ng_db if 'heat from NG, industry' in act['name']][0]
        self.heat_households = [act for act in ei_ng_db if 'heat from NG, households' in act['name']][0]
        self.mkt_ng_EU = [act for act in ei_ng_db if 'market group for natural gas, high pressure' in act['name'] and 'EU27' in act['location']][0]

    def doLCA(self, methods, verbose=True, setup_database=False):
        if len(methods)>0:
            self.write_categories(methods)

        if verbose:
            print('Setting up the LCA database...')
            start = time.time()
        # 2) Set up exchanges
        # 2.1) market for natural gas in EU27 (flexible)
        mkt_ng_EU_exc = [exc for exc in self.mkt_ng_EU.exchanges()]
        mkt_ng_EU_exc[1]['amount'] = self.NG_NL
        mkt_ng_EU_exc[2]['amount'] = self.NG_DE
        mkt_ng_EU_exc[3]['amount'] = self.NG_RO
        mkt_ng_EU_exc[4]['amount'] = self.NG_NO
        mkt_ng_EU_exc[5]['amount'] = self.NG_RU
        mkt_ng_EU_exc[6]['amount'] = self.NG_DZ
        mkt_ng_EU_exc[7]['amount'] = self.NG_AZ
        mkt_ng_EU_exc[8]['amount'] = self.LNG_US
        mkt_ng_EU_exc[9]['amount'] = self.LNG_QA
        mkt_ng_EU_exc[10]['amount'] = self.LNG_RU
        mkt_ng_EU_exc[11]['amount'] = self.LNG_NG
        mkt_ng_EU_exc[12]['amount'] = self.LNG_DZ
        for i in range(len(mkt_ng_EU_exc)):
            mkt_ng_EU_exc[i].save()
        self.energy_from_ng.save()
        self.mkt_ng_EU_exc = mkt_ng_EU_exc
        
        # 2.2) energy from ng
        energy_from_ng_exc = [exc for exc in self.energy_from_ng.exchanges()]
        energy_from_ng_exc[1]['amount'] = self.TOTAL_ELECTRICITY / self.TOTAL_ENERGY
        energy_from_ng_exc[2]['amount'] = self.HEATPOWER_HEAT * 3.6 / self.TOTAL_ENERGY
        energy_from_ng_exc[3]['amount'] = self.HEAT_INDUSTRY * 3.6 / self.TOTAL_ENERGY
        energy_from_ng_exc[4]['amount'] = self.HEAT_HOUSEHOLDS * 3.6 / self.TOTAL_ENERGY
        for i in range(len(energy_from_ng_exc)):
            energy_from_ng_exc[i].save()
        self.energy_from_ng.save()
        self.energy_from_ng_exc = energy_from_ng_exc

        # 2.3) electricity from ng
        elec_from_ng_exc = [exc for exc in self.elec_from_ng.exchanges()]
        elec_from_ng_exc[1]['amount'] = self.NG_POWER / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[2]['amount'] = self.NG_CHP / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[3]['amount'] = self.COAL_POWER  / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[4]['amount'] = self.COAL_CHP / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[5]['amount'] = self.NUCLEAR_POWER / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[6]['amount'] = self.WIND_POWER / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[7]['amount'] = self.SOLAR_POWER / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[8]['amount'] = self.BIOMASS_CHP / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[9]['amount'] = self.OIL_POWER / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[10]['amount'] = self.OIL_CHP / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[11]['amount'] = self.LIGNITE_POWER / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[12]['amount'] = self.LIGNITE_CHP / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[13]['amount'] = self.HYDRO_POWER / self.TOTAL_ELECTRICITY
        elec_from_ng_exc[14]['amount'] = self.SAVINGS_POWER / self.TOTAL_ELECTRICITY
        for i in range(len(elec_from_ng_exc)):
            elec_from_ng_exc[i].save()
        self.energy_from_ng.save()
        self.elec_from_ng_exc = elec_from_ng_exc

        # 2.4) heat CHP from ng
        heat_from_ng_exc = [exc for exc in self.heat_from_ng.exchanges()]
        heat_from_ng_exc[1]['amount'] = self.NG_CHP_HEAT / self.HEATPOWER_HEAT
        heat_from_ng_exc[2]['amount'] = self.COAL_CHP_HEAT / self.HEATPOWER_HEAT
        heat_from_ng_exc[3]['amount'] = self.BIOMASS_CHP_HEAT / self.HEATPOWER_HEAT
        heat_from_ng_exc[4]['amount'] = self.OIL_CHP_HEAT / self.HEATPOWER_HEAT
        heat_from_ng_exc[5]['amount'] = self.LIGNITE_CHP_HEAT / self.HEATPOWER_HEAT
        for i in range(len(heat_from_ng_exc)):
            heat_from_ng_exc[i].save()
        self.energy_from_ng.save()
        self.heat_from_ng_exc = heat_from_ng_exc

        # 2.5) heat in industry from natural gas
        heat_industry_exc = [exc for exc in self.heat_industry.exchanges()]
        heat_industry_exc[1]['amount'] = self.NG_INDUSTRY / (self.HEAT_INDUSTRY)
        heat_industry_exc[2]['amount'] = self.COAL_INDUSTRY / (self.HEAT_INDUSTRY)
        heat_industry_exc[3]['amount'] = self.OIL_INDUSTRY / (self.HEAT_INDUSTRY)
        heat_industry_exc[4]['amount'] = self.EFFICIENCY_INDUSTRY / (self.HEAT_INDUSTRY)
        heat_industry_exc[5]['amount'] = self.SAVINGS_INDUSTRY / (self.HEAT_INDUSTRY)
        for i in range(len(heat_industry_exc)):
            heat_industry_exc[i].save()
        self.energy_from_ng.save()
        self.heat_industry_exc = heat_industry_exc
        
        # 2.6) heat in industry from natural gas
        heat_households_exc = [exc for exc in self.heat_households.exchanges()]
        heat_households_exc[1]['amount'] = self.NG_HOUSEHOLDS / (self.HEAT_HOUSEHOLDS)
        heat_households_exc[2]['amount'] = self.ELECTRICITY_HOUSEHOLDS / (self.HEAT_HOUSEHOLDS)
        heat_households_exc[3]['amount'] = self.WEATHER_HOUSEHOLDS / (self.HEAT_HOUSEHOLDS)
        heat_households_exc[4]['amount'] = self.EFFICIENCY_HOUSEHOLDS / (self.HEAT_HOUSEHOLDS)
        heat_households_exc[5]['amount'] = self.SAVINGS_HOUSEHOLDS / (self.HEAT_HOUSEHOLDS)
        for i in range(len(heat_households_exc)):
            heat_households_exc[i].save()
        self.energy_from_ng.save()
        self.heat_households_exc = heat_households_exc

        # 3) Set up the functional unit
        self.functional_unit = {self.energy_from_ng: self.TOTAL_ENERGY*1.0e9}

        # Check whether the functional unit considers all the energy flows
        if verbose:
            # Sanity check
            print("Your functional unit considers the following energy flows [TWh]:")
            print(f"\tElectricity:             {sum([exc.amount for exc in elec_from_ng_exc[1::]]) * energy_from_ng_exc[1].amount * self.TOTAL_ENERGY}")
            print(f"\tHeat cogeneration (CHP): {sum([exc.amount for exc in heat_from_ng_exc[1::]]) * energy_from_ng_exc[2].amount * self.TOTAL_ENERGY / 3.6}")
            print(f"\tIndustrial heating:      {sum([exc.amount for exc in heat_industry_exc[1::]]) * energy_from_ng_exc[3].amount * self.TOTAL_ENERGY / 3.6}")
            print(f"\tHousehold heating:       {sum([exc.amount for exc in heat_households_exc[1::]]) * energy_from_ng_exc[4].amount * self.TOTAL_ENERGY / 3.6}")

            elapsed = time.time() - start
            print(f'Finished database setup in {elapsed:.2f} s')

        if setup_database:
            return True

        # 4) LCA calculations
        if verbose:
            start = time.time()
            print('LCA method\t\t\t\t\t\t LCA result\n====================================================================================')
        if len(methods)>0:
            self.lca_results = np.zeros(len(methods), dtype=object)
            lca = bw.LCA(self.functional_unit)
            lca.lci()
            
            # change technosphere matrix to avoid double-counting
            if self.no_double_counting:
                self.modify_technosphere_for_double_counting(lca)
            
            for i, met in enumerate(methods):
                lca.switch_method(met)
                lca.lcia()
                if verbose:
                    print(f'{met}:\t {lca.score/1}')  # kg CO2 eq./year')
                self.lca_results[i] = copy.copy(lca)
        
        if verbose:
            elapsed = time.time() - start
            print(f'Finished LCIA calculations in {elapsed:.2f} s')
                
    def write_categories(self, methods):
        """Write the categories and their units

        Parameters
        ----------
        methods : list of LCIA methods
        """

        if len(methods)>0:
            self.lca_methods = methods
            self.categories = [' '.join(mm) for mm in methods]
            self.categories_unit = [bw.methods[m]['unit'] for m in methods]


    def modify_technosphere_for_double_counting(self, lca):
        '''
        Modify the technosphere matrix to set the entry of given products to 0 for all the activities but 
        those used in the functional unit and the activities producing the product (i.e., matrix diagonal)
        '''
        
        # In order to handle the sparse technosphere matrix, let's map index to activity for all ecoinvent activities
        # Get all the activities that are subject to double counting starting with the 'parent' activity
        activities_target = dict()    
        for energy in self.energy_from_ng.technosphere():
            for fuel in bw.get_activity(energy.input).technosphere():
                if fuel['name'] in ['electricity production, NG',
                                    'electricity production, NG, CHP',
                                    'heat production from NG, CHP',
                                    'heat production from NG, central or small-scale',
                                    'heat production from NG, industry',]:
                    for tech in bw.get_activity(fuel.input).technosphere():
                        tech_index = lca.activity_dict[tech.input.key]
                        activities_target.update({tech_index: tech.input})

        # The activities that should be modified are the activities retrieived above plus the activities used in the scenarios
        activities_no_modify = dict()
        activities_no_modify.update(activities_target)
        scenario_activities = [ds for ds in self.ei_ng_db if ds['name'] in ['electricity production, NG',
                                                                                'electricity production, NG, CHP',
                                                                                'heat production from NG, CHP',
                                                                                'heat production from NG, industry',
                                                                                'heat production from NG, central or small-scale']]
        activities_no_modify.update({lca.activity_dict[ds.key]: ds for ds in scenario_activities})

        # Modify the technosphere matrix
        cx = scipy.sparse.coo_matrix(lca.technosphere_matrix)
        for i,j,v in zip(cx.row, cx.col, cx.data):
            if i in activities_target and j not in activities_no_modify:
                lca.technosphere_matrix[i,j] = 0

        # Redo the LCI
        lca.redo_lci()

    def do_process_contribution(self, methods, verbose=True):

        self.doLCA([], verbose=verbose)
        lca = bw.LCA(self.functional_unit)
        lca.lci()

        # elec_from_ng_exc = [exc for exc in self.elec_from_ng.exchanges()]
        # energy_from_ng_exc = [exc for exc in self.energy_from_ng.exchanges()]
        # heat_from_ng_exc = [exc for exc in self.heat_from_ng.exchanges()]
        exc = self.elec_from_ng_exc[1::] + self.heat_from_ng_exc[1::] + \
            self.heat_industry_exc[1::] + self.heat_households_exc[1::]
        amount_mult = np.concatenate((self.energy_from_ng_exc[1]['amount'] * np.ones(len(self.elec_from_ng_exc[1::])),
                                      self.energy_from_ng_exc[2]['amount'] * np.ones(len(self.heat_from_ng_exc[1::])),
                                      self.energy_from_ng_exc[3]['amount'] * np.ones(len(self.heat_industry_exc[1::])),
                                      self.energy_from_ng_exc[4]['amount'] * np.ones(len(self.heat_households_exc[1::]))), axis=None)
        self.exc_process_contribution = exc

        process_lca = np.zeros(len(methods), dtype=object)
        process_lca_dict = {}

        for j in range(len(exc)):
            act_code = exc[j]['input'][1]
            act_from_code = [act for act in self.ei_ng_db if act['code'] == act_code][0]
            if verbose:
                print(f'{act_from_code}:')  # Gt CO2 eq./year')
            name = act_from_code['name'] + " - " + act_from_code['unit']
            process_lca_dict[f'{name}'] = {}

            functional_unit = {act_from_code: amount_mult[j] * exc[j]['amount']*self.TOTAL_ENERGY*1.0e9}

            if amount_mult[j] * exc[j]['amount']*self.TOTAL_ENERGY > 1.0e-6:
                lca = bw.LCA(functional_unit)
                lca.lci()

                if self.no_double_counting:
                    self.modify_technosphere_for_double_counting(lca)
                    
                for i, met in enumerate(methods):
                    lca.switch_method(met)
                    lca.lcia()

                    if verbose:
                        print(f'{name}:\t {lca.score/1}')  # Gt CO2 eq./year')
                    # process_lca[i] = copy.copy(lca)
                    process_lca_dict[name][f'{met}'] = lca.score
            else:
                for i, met in enumerate(methods):
                    process_lca_dict[name][f'{met}'] = 0.0
        self.process_contribution = process_lca_dict
        
        if len(methods)>0:
            self.write_categories(methods)
        
        
    def multi_lcia_MonteCarlo(self, iterations, lcia_methods, seed=42):
        '''
        This function performs Monte Carlo simulation accross a range of categories

        iterations: e.g., 1000
        lcia_methods: impact assessment methods dictionary, e.g.: {'Climate change': 'IPCC 2013, GWP100, etc."}
        '''
        # Create a MonteCarloLCA object with functional unit but no method. 
        self.doLCA([], setup_database=True)
        MC_LCA = bw.MonteCarloLCA(self.functional_unit, seed=seed)
        
        # Run .lci to build the A and B matrices (and hence fix the indices of our matrices)
        MC_LCA.lci()
        
        # Iterate over the methods and stores the C matrix in a dictionary
        C_matrices = dict()
        for method in lcia_methods:
            MC_LCA.switch_method(lcia_methods[method])
            C_matrices[method] = MC_LCA.characterization_matrix
            
        # Create a dictionary to store the results
        results_MC = np.empty((len(lcia_methods), iterations))

        # Populate results dictionary using a for loop over the number of MonteCarlo iterations required 
        # Include a nested for loop over the methods and multiply the characterization matrix with the inventory. 
        print('Starting Monte Carlo')      
        for iteration in range(iterations):
            next(MC_LCA)

            if self.no_double_counting:
                self.modify_technosphere_for_double_counting(MC_LCA)
            
            if iteration%10==0:
                print(f"iteration = {iteration}")
            
            for method_index, method in enumerate(lcia_methods):
                results_MC[method_index, iteration] = (C_matrices[method] * MC_LCA.inventory).sum()
        
        results_MC_dict = dict()
        i_count = 0
        for method in lcia_methods:
            results_MC_dict[method] = results_MC[i_count]
            i_count += 1

        self.results_MC_dict = results_MC_dict
        
        return results_MC_dict