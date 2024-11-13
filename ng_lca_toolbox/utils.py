import numpy as np
import pandas as pd

import brightway2 as bw

from .ng_scenario_lca import NgScenarioLCA



def excel2dfs(path2excel_scenarios='./data/scenarios/scenarios_data.xlsx'):
    # Scenarios from excel - supply:
    df_ng_supply = pd.read_excel(path2excel_scenarios, sheet_name='supply')
    df_ng_supply.drop([lab for lab in df_ng_supply.columns if 'Unnamed' in lab or 'Supplier' in lab], axis=1, inplace=True)
    df_ng_supply.dropna(axis=0, how='any', inplace=True)
    df_ng_supply.set_index('code', inplace=True)

    # Scenarios from excel - demand:
    df_ng_demand = pd.read_excel(path2excel_scenarios, sheet_name='demand', skiprows=[0])
    df_ng_demand.drop([lab for lab in df_ng_demand.columns if 'Unnamed' in lab or 'source' in lab], axis=1, inplace=True)
    df_ng_demand.set_index('code', inplace=True)

    return df_ng_supply, df_ng_demand    


def excel2ngScenarioLCA(scenario, supply_scenario, ei_ng_db, path2excel_scenarios='./data/scenarios/scenarios_data.xlsx'): 
    df_ng_supply, df_ng_demand = excel2dfs(path2excel_scenarios)
    df_ng_demand.dropna(axis=0, how='any', inplace=True)

    # generating dictionary for scenario data
    valid_index = df_ng_demand.index.dropna()
    scenario_data_dict = df_ng_demand[[f'{scenario}_output']].to_dict()[f'{scenario}_output']
    scenario_data_dict['ELECTRICITY'] = df_ng_demand.loc[[ite for ite in valid_index if 'POWER' in ite]][[f'{scenario}_output']].sum().values[0]
    scenario_data_dict['HEATPOWER'] = df_ng_demand.loc[[ite for ite in valid_index if 'CHP' in ite and '_HEAT_' not in ite]][[f'{scenario}_output']].sum().values[0]
    scenario_data_dict['HEATPOWER_HEAT'] = df_ng_demand.loc[[ite for ite in valid_index if 'CHP' in ite and '_HEAT_' in ite]][[f'{scenario}_output']].sum().values[0]
    scenario_data_dict['HEAT_INDUSTRY'] = df_ng_demand.loc[[ite for ite in valid_index if 'INDUSTRY' in ite]][[f'{scenario}_output']].sum().values[0]
    scenario_data_dict['HEAT_HOUSEHOLDS'] = df_ng_demand.loc[[ite for ite in valid_index if 'HOUSEHOLDS' in ite]][[f'{scenario}_output']].sum().values[0]
    # supply_scenario = "base" if scenario=="base" else 'alternative'
    scenario_data_dict.update(df_ng_supply[[f'{supply_scenario}_share']].to_dict()[f'{supply_scenario}_share'])
    total_share = df_ng_supply.loc["TOT_NG", f'{supply_scenario}_share']
    assert abs(total_share-1.0) < 1e-3
    assert abs(df_ng_supply[[f'{supply_scenario}_share']].replace("-", 0.0).sum().values[0]-1.0 - total_share) < 1e-3

    return NgScenarioLCA(scenario_data_dict, ei_ng_db=ei_ng_db, no_double_counting=True)


def result2df(list_of_NgScenarioLCA, verbose=True):
    # return df_results
    n_scenarios = len(list_of_NgScenarioLCA)

    df_results = np.zeros(
        len(list_of_NgScenarioLCA[0].lca_methods), dtype=object)

    scenario = []
    for j in range(n_scenarios):
        if type(list_of_NgScenarioLCA[j])==list or type(list_of_NgScenarioLCA[j])==np.ndarray:
            print('using surrogate for LCA')
            scenario.append(np.insert(list_of_NgScenarioLCA[j], 0, 0.0))
        else:
            scenario.append(list_of_NgScenarioLCA[j].lca_results)
    for i in range(len(list_of_NgScenarioLCA[0].lca_methods)):
        data_df = {}
        for j in range(n_scenarios):
            if type(scenario[j][0]) == np.float64:
                # scenario[j] = np.insert(scenario[j], 0, 0.0)
                data_df[f'scenario_{j}'] = [scenario[j][i],
                                            100*scenario[j][i] / scenario[0][i].score,
                                            (scenario[j][i] - scenario[0][i].score)]
            else:
                data_df[f'scenario_{j}'] = [scenario[j][i].score/1,
                                            100*scenario[j][i].score /
                                            scenario[0][i].score,
                                            (scenario[j][i].score - scenario[0][i].score)/1]

        met = scenario[0][i].method[0] + ' ' + \
            scenario[0][i].method[1] 
        if len(scenario[0][i].method)>2:
            met += ' ' + scenario[0][i].method[2]
            
        columns_df = [met, 'Difference from BAU [%]',
                      'Difference from BAU']
        df_results[i] = pd.DataFrame.from_dict(
            data_df, orient='index', columns=columns_df)
        df_results[i] = df_results[i].round(
            {columns_df[0]: 0, columns_df[1]: 2, columns_df[2]: 0})
        # df_results[i][met] = df_results[i][met].map('{:,.2e}'.format)
        if verbose:
            print(df_results[i][columns_df[:2]])

    return df_results
