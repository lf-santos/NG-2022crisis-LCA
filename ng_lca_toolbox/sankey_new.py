import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px
import numpy as np
import pandas as pd
import copy

def generate_pickle_sankey(vars, rus_sup='reduced'):
    input_dict = {}

    elec_non_ng = 0
    chp_non_ng = 0
    hi_non_ng = 0 
        
    input_label_elec = ["coal", "oil", "nuclear", "solar", "wind", "lignite"]
    input_label_chp = ["coal", "oil", "biomass", "lignite"]
    input_label_industry = ["coal", "oil"]
    for i, l in enumerate(input_label_elec):
        input_dict[f'{l}_pp_twh'] = vars[i] * 0.5 * 9.77
        input_dict[f'{l}_pp'] = vars[i] * 9.77
        elec_non_ng += input_dict[f'{l}_pp']
    for i, l in enumerate(input_label_chp):
        input_dict[f'{l}_chp_twh'] = vars[i+len(input_label_elec)]*0.918611*9.77
        input_dict[f'{l}_chp'] = vars[i+len(input_label_elec)]*9.77
        chp_non_ng += input_dict[f'{l}_chp']
    for i, l in enumerate(input_label_industry):
        input_dict[f'{l}_hi_twh'] = vars[i+len(input_label_elec)+len((input_label_chp))]*0.918611*9.77
        input_dict[f'{l}_hi'] = vars[i+len(input_label_elec)+len((input_label_chp))]*9.77
        hi_non_ng += input_dict[f'{l}_hi']
    input_dict['savings'] = vars[-2]
    input_dict['heatpump'] = vars[-1]
        
        
    savings=vars[-1]
    heatpump=vars[-2]
    ng_base_twh = [132.3*9.77, 78.6*9.77, 80.9*9.77, 34.1*9.77, 10.2*9.77, 39.5*9.77,]
    ng_base_twh = [132.3*9.77, 78.6*9.77, 80.9*9.77, 34.1*9.77, (10.2+13.6)*9.77, 39.5*9.77,] # with stock change
    # needed_NG = sum([152.4*9.77, 94.3*9.77, 112.9*9.77, 34.1*9.77, 10.2*9.77, 77.5*9.77]) # BP
    needed_NG = sum(ng_base_twh)
    elec_usage = 127.0 * 0.46
    chp_usage = 127.0 * 0.54
    hi_usage = 97.3
    hh_usage = 137.8
    # other_usage_ng = needed_NG - sum([154.5*9.77*0.55, 154.5*9.77*0.45, 110.5 * 9.77, 174.2*9.77])
    other_usage_ng = needed_NG - sum([elec_usage*9.77,chp_usage*9.77,hi_usage*9.77,hh_usage*9.77,])
    print(other_usage_ng)
    if other_usage_ng <= 0:
        other_usage_ng = 0.01
    # ng_comp_values = [(132.3-110.0)*9.77, (78.6+50.0)*9.77, 80.9*9.77, 34.1*9.77, (10.2+13.6+10.0)*9.77, 39.5*9.77] # BP
    ng_comp_values = copy.deepcopy(ng_base_twh)
    ng_comp_values[0] -= 110*9.77
    ng_comp_values[1] += 50*9.77
    ng_comp_values[-2] += 10*9.77

    x=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25,        # 0-8
                0.6, 0.6, 0.6, 0.6,                                 # 9-12
                0.1, 0.1, 0.1, 0.1, 0.1, 0.1,                       # 13-18
                0.8, 0.8, 0.8, 0.8, 0.6, 0.8, 0.6,                      # 19-25
                0.6, 0.8,
                ]
    label=["NG", "Coal", "Oil", "Nuclear", "Solar", "Wind", "Biomass", "Electric heaters", "Savings in households", # 0-8
        'Electricity production', 'Heat&Power cogeneration', 'Heat in industry', 'Heat in households',              # 9-12  
        'Russia', 'LNG', 'Norway', 'Algeria', 'Other sources', 'Endogenous production',                             # 13-18             
        'Electricity', 'Steam/furnace', 'Furnace', 'Ambient/water heat', 'Other NG usage', 'Loss', 'Lignite',       # 19-25
        'dummy1','dummy2',# dummy to fix base
        ]
    COLOR=[
                '#a6cee3',  # NG
                '#b15928',  # COal
                '#ffff99',  # Oil
                '#6a3d9a',  # Nuclear
                '#FF00FF', #'magenta',  # Solar
                '#e31a1c',  # Wind
                '#808000',  # Biomass
                '#ff7f00',  # Electric heaters
                '#cab2d6',  # savings households

                '#fdbf6f',  # electric production
                '#b2df8a',  # CHP
                '#33a02c',  # Heat in industry
                '#fb9a99',  # heat households
                
                '#a6cee3',  # NG-Russia
                '#a6cee3', #'#1f78b4',  # LNG
                '#a6cee3',  # NG-
                '#a6cee3',  # NG-
                '#a6cee3',  # NG-
                '#a6cee3',  # NG-endogenous production
                
                '#D3D3D3',  # final products
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
                '#D3D3D3',
            ]
    COLOR[25] = '#964B00'#'brown'
    
    # Add opacity with color entry as rgba
    # COLOR = [f"rgb{tuple(int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))}" for h in COLOR]
    COLOR = [f"rgba({int(h.lstrip('#')[0:2], 16)}, {int(h.lstrip('#')[2:4], 16)}, {int(h.lstrip('#')[4:6], 16)}, 0.75)" for h in COLOR]
    # print(COLOR)
    
    source=[0, 0, 0, 0, 0,
            1, 1, 1,
            2, 2, 2,
            3, 3, 3,
            4, 4, 4,
            5, 5, 5,
            6, 6, 6,
            25, 25, 25,
            7,
            8,
            13, 14, 15, 16, 17, 18,
            12, 12,
            11, 11, 11, # heat in industry
            9, 9,
            10, 10, 10,
            23, 
            26,# dummy to fix base
            ]
    target=[23, 9, 10, 11, 12,
                    9, 10, 11,
                    9, 10, 11,
                    9, 10, 11,
                    9, 10, 11,
                    9, 10, 11,
                    9, 10, 11,
                    9, 10, 11,
                    12,
                    12,
                    0, 0, 0, 0, 0, 0,
                    22, 24,
                    20, 21, 24,
                    19, 24,
                    19, 20, 24,
                    19,
                    27,# dummy to fix base
                    ]
    value=[
                other_usage_ng,
                elec_usage*9.77-elec_non_ng, 
                chp_usage*9.77-chp_non_ng, hi_usage* 9.77-hi_non_ng, 
                (hh_usage-savings-heatpump)*9.77,
                input_dict['coal_pp'],
                input_dict['coal_chp'],
                input_dict['coal_hi'],
                input_dict['oil_pp'],
                input_dict['oil_chp'],
                input_dict['oil_hi'],
                input_dict['nuclear_pp'],
                0,#input_dict['nuclear_chp'],
                0,#input_dict['nuclear_hi'],
                input_dict['solar_pp'],
                0,#input_dict['solar_chp'],
                0,#input_dict['solar_hi'],
                input_dict['wind_pp'],
                0,#input_dict['wind_chp'],
                0,#input_dict['wind_hi'],
                0,#input_dict['biomass_pp'],
                input_dict['biomass_chp'],
                0,#input_dict['biomass_hi'],
                input_dict['lignite_pp'],
                input_dict['lignite_chp'],
                0,#input_dict['lignite_hi'],
                heatpump*9.77,
                savings*9.77,] + ng_comp_values + [
                hh_usage*9.77*0.917879344, (1-0.917879344)*hh_usage*9.77,
                hi_usage*9.77*(1-0.0*0.376)*0.918611, hi_usage*9.77*0.0*0.376*0.918611, hi_usage*(1-0.918611)*9.77,  # heat in industry
                elec_usage*9.77*0.459,elec_usage*9.77*(1-0.459),
                chp_usage*9.77*0.459, chp_usage*9.77*0.33, chp_usage*9.77*(1-0.459-0.33),
                0.0,
                50*9.77# dummy to fix base
            ]
    input_dict['ng_comp'] = ng_comp_values
    input_dict['x'] = x
    input_dict['label'] = label
    input_dict['color'] = COLOR
    input_dict['source'] = source
    input_dict['target'] = target
    input_dict['value'] = value
    input_dict['needed_NG'] = needed_NG
    input_dict['ng_comp_values'] = ng_comp_values
    input_dict['title_text'] = f"Sankey diagram for NG shortage in supply = {(needed_NG-sum(ng_comp_values))/9.77:0.0f} BCMS.\nCompensated NG = {heatpump+savings+sum([v for v in vars[0:-2]]):0.0f} BCMs"
    input_dict['savings'] = savings
    input_dict['heatpump'] = heatpump
    if rus_sup=='phaseout':
        input_dict['value'][28] = 0.01
        input_dict['value'][29] -= 14.4*9.77
        input_dict['ng_comp_values'][0] = 0.0
        input_dict['ng_comp_values'][1] -= 14.4*9.77
        del input_dict['title_text']
        input_dict['title_text'] = f"Sankey diagram for NG shortage in supply = {(input_dict['needed_NG']-sum(list(input_dict['ng_comp_values'])))/9.77:0.0f} BCM.\nCompensated NG = {input_dict['heatpump']+input_dict['savings']+sum([v for v in input_dict['value'][5:23]])/9.77:0.0f} BCM"
        # input_dict['title_text'] = f"Sankey diagram for NG shortage in supply = {(input_dict['needed_NG']-sum(list(input_dict['ng_comp_values'])))/9.77:0.0f} BCM.\nCompensated NG = {sum([v for k, v in input_dict.items() if ('_pp' in k or '_chp' in k or '_hi' in k or 'savings' in k or 'heatpump' in k)])/9.77:0.0f} BCM"
    elif rus_sup=='base':
        # ng_comp_values = [132.3*9.77, 78.6*9.77, 80.9*9.77, 34.1*9.77, (10.2+13.6)*9.77, 39.5*9.77] # BP
        ng_comp_values = ng_base_twh
        input_dict['value'][28:28+6] = ng_comp_values
        input_dict['ng_comp_values'][0:6] = ng_comp_values
        del input_dict['title_text']
        # input_dict['title_text'] = f"Sankey diagram for NG shortage in supply = {(input_dict['needed_NG']-sum(list(input_dict['ng_comp_values'])))/9.77:0.0f} BCM.\nCompensated NG = {sum([v for k, v in input_dict.items() if ('_pp_twh' in k or '_chp_twh' in k or '_hi_twh' in k or 'savings' in k or 'heatpump' in k)])/9.77:0.0f} BCM"
        input_dict['title_text'] = f"Sankey diagram for NG shortage in supply = {(input_dict['needed_NG']-sum(list(input_dict['ng_comp_values'])))/9.77:0.0f} BCM.\nCompensated NG = {input_dict['heatpump']+input_dict['savings']+sum([v for v in input_dict['value'][5:23]])/9.77:0.0f} BCM"

    # with open('./models/sankey_dict6.pickle', 'wb') as handle:
    #     pickle.dump(input_dict, handle)

    return input_dict


def generate_data_sankey(pickle_path='./models/sankey_dict.pickle', x_data='', ite=0, rus_sup='reduced'):
    
    with open(pickle_path, 'rb') as handle:
        input_dict = pickle.load(handle)
    if x_data=='':
        input_dict2 = input_dict
    else:
        input_dict2 = x_data
        
    error_msg = '' 
    
    xx = np.array([0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30,
                   0.55, 0.55, 0.55, 0.55,
                   0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                #    0.99, 0.99, 0.99, 0.99, 0.55, 0.99, 0.3,
                   0.99, 0.99, 0.99, 0.55, 0.99, 0.3,
                   0.3, 0.55, # dummy to fix base
                   ])
    yy = np.concatenate(
        (
            np.append(np.array([0.15]), np.linspace(0.4, 0.93, 8)),
            np.linspace(0.01, 0.99, 5)[0:-1],
            np.linspace(0.11, 0.99, 6),
            np.linspace(0.01, 0.99, 4)[0:-1], 0.99*np.ones(3),
            [1.1, 1.1], # dummy to fix base
        )
    )
    print(input_dict2['label'])
    df = pd.DataFrame([xx, yy, input_dict2['label'], input_dict2['color']]).T
    df.columns = ['x', 'y', 'label', 'color']
    # df.index = input_dict2['label']
    df.set_index('label', inplace=True)
    df.dropna(inplace=True)
    sum_coal = sum([v for k, v in input_dict2.items() if 'coal' in k])
    if sum_coal == 0:
        df.loc['Coal'].x = 0.123
        df.loc['Coal'].y = 0.123
        # df.drop('Coal', inplace=True)
    sum_oil = sum([v for k, v in input_dict2.items() if 'oil' in k])
    if sum_oil == 0:
        df.loc['Oil'].x = 0.123
        df.loc['Oil'].y = 0.123
        # df.drop('Oil', inplace=True)
    sum_nuclear = sum([v for k, v in input_dict2.items() if 'nuclear' in k])
    if sum_nuclear == 0:
        df.loc['Nuclear'].x = 0.123
        df.loc['Nuclear'].y = 0.123
        # df.drop('Nuclear', inplace=True)
    sum_solar = sum([v for k, v in input_dict2.items() if 'solar' in k])
    if sum_solar == 0:
        df.loc['Solar'].x = 0.123
        df.loc['Solar'].y = 0.123
        # df.drop('Solar', inplace=True)
    sum_wind = sum([v for k, v in input_dict2.items() if 'wind' in k])
    if sum_wind == 0:
        df.loc['Wind'].x = 0.123
        df.loc['Wind'].y = 0.123
        # df.drop('Wind', inplace=True)
    sum_biomass = sum([v for k, v in input_dict2.items() if 'biomass' in k])
    if sum_biomass == 0:
        df.loc['Biomass'].x = 0.123
        df.loc['Biomass'].y = 0.123
        # df.drop('Biomass', inplace=True)
    sum_lignite = sum([v for k, v in input_dict2.items() if 'lignite' in k])
    if sum_lignite == 0:
        df.loc['Lignite'].x = 0.123
        df.loc['Lignite'].y = 0.123
        # df.drop('Biomass', inplace=True)
    sum_heatpump = sum([v for k, v in input_dict2.items() if 'heatpump' in k])
    if sum_heatpump == 0:
        df.loc['Electric heaters'].x = 0.123
        df.loc['Electric heaters'].y = 0.123
        # df.drop('Electric heaters', inplace=True)
    sum_savings = sum([v for k, v in input_dict2.items() if 'savings' in k])
    if sum_savings == 0:
        df.loc['Savings in households'].x = 0.123
        df.loc['Savings in households'].y = 0.123
        # df.drop('Savings in households', inplace=True)

    df_links = pd.DataFrame.from_dict({
        'source': input_dict2['source'],
        'target': input_dict2['target'],
        'value': input_dict2['value']
    })
    # df_links = df_links.loc[(df_links.value>0)]
    if ite == 1:
        x_domain = [0.0, 0.47]
        y_domain = [0.57, 1.0]
    elif ite == 2:
        x_domain = [0.53, 1.0]
        y_domain = [0.57, 1.0]
    elif ite == 3:
        x_domain = [0.0, 0.47]
        y_domain = [0.0, 0.43]
    elif ite == 4:
        x_domain = [0.53, 1.0]
        y_domain = [0.0, 0.43]
    else:
        x_domain = [0.0, 1.0]
        y_domain = [0.0, 1.0]

    # link colors
    # if
    node_label = [d if d != 'Heat&Power cogeneration' else 'CHP' for d in list(df.index)]
    node_label = [d if d != 'Endogenous production' else 'Endogenous' for d in node_label]
    node_label = [d if d != 'Ambient/water heat' else 'Individual heating' for d in node_label]
    node_label = [d if d != 'Other NG usage' else 'Other usage' for d in node_label]
    
    # annotating values inside nodes
    x_pos = [xxx for xxx in df.x.values if xxx != 0.123]
    y_pos = [yyy for yyy in df.y.values if yyy != 0.123]
        
    data =  go.Sankey(
        domain={
            'x': x_domain,
            'y': y_domain,
        },
        valueformat="0.0f",
        valuesuffix="TWh",
        # hovertemplate = "%{label}: <br>Popularity: %{percent} </br> %{text}"
        # node_hoverlabel = 'hi',
        # arrangement='snap',
        # orientation = 'h',
        textfont = dict(color='black',size=16,family='calibri'),
        node={
            'pad': 40,
            'thickness': 10,
            'line': dict(color="black", width=0.3),
            'x': x_pos,
            'y': y_pos,
            # [d if d!='Heat&Power cogeneration' else 'CHP' for d in list(df.index)],# list(df.index),
            'label': node_label,
            'color': list(df.color.values),
        },
        link={
            'source': list(df_links.source.values),
            'target': list(df_links.target.values),
            'value':  list(df_links.value.values),
            'colorscales': [dict(colorscale=[[0, 'rgb(0,0,255)'], [1, 'rgb(255,0,0)']])],
            'color': [df.color.iloc[i] for i in df_links.source.values],
            # 'color': ['#a6cee3', '#a6cee3', '#a6cee3', '#a6cee3', '#a6cee3', '#1f78b4', '#1f78b4', '#1f78b4', '#b2df8a', '#b2df8a', '#b2df8a', '#33a02c', '#33a02c', '#33a02c',
            #           '#fb9a99', '#fb9a99', '#fb9a99', '#e31a1c', '#e31a1c', '#e31a1c', '#fdbf6f', '#fdbf6f', '#fdbf6f', '#ff7f00', '#cab2d6',
            #           '#a6cee3', '#a6cee3', '#a6cee3', '#a6cee3', '#a6cee3',
            #           'magenta', 'magenta', '#ffff99', 'red', 'black', '#ffff99', 'black', '#ffff99', 'red', 'black', 'black',],
        }
    )
    fig = go.Figure(
        data=data,
    )
    # fig.update_traces(
    #     node_hovertemplate = '<br>%{x}'+ '<br>Increase from base: <b>%{y:.2f}%',
    # )
    fig.update_layout(
        title_text = input_dict2['title_text'][0] if isinstance(input_dict2['title_text'], tuple) else input_dict2['title_text'],
        font_size=16,
        font_color='black',
        font_family='calibri',
        paper_bgcolor='white', #'rgba(1,1,1,1)',#'white', #'#f5f2ea',  # 'rgba(220,220,220,0.1)',#'#D3D333',
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode='x',
        xaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False,
        },
        yaxis={
            'showgrid': False,
            'zeroline': False,
            'visible': False,
        },
        margin_t=90,
        margin_b=60,
        margin_l=30,
        margin_r=30,
        title_pad_t=0,
        title_pad_b=0,
        title_pad_l=30,
        title_pad_r=30,
    )
    x_coordinate = [0.0, 0.27, 0.52, 0.95]
    for i, column_name in enumerate(["NG supply", "Energy source", "Process", 'Product']):
        fig.add_annotation(
            x=x_coordinate[i],
            # x=x_coordinate/8,
            y=1.12,
            xref="x",
            yref="paper",
            text=column_name,
            showarrow=False,
            font=dict(
                family="calibri",
                size=16,
                color="black"
            ),
            align="center",
        )
        
    # fig.write_image('sankey1.svg')
    
    return fig

def sankey_saver():
    """This function reads the pickle files that describe the sankey diagrams, plots them, and saves them
    """
    for opt in range(6):
        fig_sankey = generate_data_sankey(f'./models/sankey_dict{opt+1}.pickle', 
                                            rus_sup='reduced' if opt>0 else 'base')
        
        fig_sankey.write_image(f'./sankey_figs/sankey{opt+1}.svg')
        

if __name__=='__main__':
    
    list_of_x0 = [  {'coal_pp': 0, 'oil_pp': 0, 'nuclear_pp': 0, 'solar_pp': 0, 'wind_pp': 0, 'lignite_pp': 0, 'coal_chp': 0, 'oil_chp': 0, 'biomass_chp': 0, 'lignite_chp': 0, 'coal_hi': 0, 'oil_hi': 0, 'heatpump': 0, 'savings': 0},
                    {'coal_pp': 24.0, 'oil_pp': 0, 'nuclear_pp': 7.0, 'solar_pp': 0, 'wind_pp': 0, 'lignite_pp': 0, 'coal_chp': 0, 'oil_chp': 0, 'biomass_chp': 0, 'lignite_chp': 0, 'coal_hi': 0, 'oil_hi': 0, 'heatpump': 9.0, 'savings': 10.0},
                    {'coal_pp': 50.0, 'oil_pp': 0, 'nuclear_pp': 0, 'solar_pp': 0, 'wind_pp': 0, 'lignite_pp': 0, 'coal_chp': 0, 'oil_chp': 0, 'biomass_chp': 0, 'lignite_chp': 0, 'coal_hi': 0, 'oil_hi': 0, 'heatpump': 0, 'savings': 0},
                    {'coal_pp': 0, 'oil_pp': 0, 'nuclear_pp': 0, 'solar_pp': 20.0, 'wind_pp': 20.0, 'lignite_pp': 0, 'coal_chp': 0, 'oil_chp': 0, 'biomass_chp': 0, 'lignite_chp': 0, 'coal_hi': 0, 'oil_hi': 0, 'heatpump': 0, 'savings': 10.0},
                ]
    
    for i, x0 in enumerate(list_of_x0):
        input_dict = generate_pickle_sankey(list(x0.values()), rus_sup='base' if sum(list(x0.values()))==0 else 'reduced')
        with open(f'./models/sankey_dict{i+1}.pickle', 'wb') as handle:
            pickle.dump(input_dict, handle)
        print(input_dict)
        print(len(input_dict['label']))
        print(len(input_dict['color']))
        print(len(input_dict['source']))
        print(len(input_dict['target']))
        print(len(input_dict['value']))
        print(len(input_dict['x']))
        # generate_data_sankey(x_data=input_dict, rus_sup='base' if sum(list(x0.values()))==0 else 'reduced')
        fig_sankey = generate_data_sankey(f'./models/sankey_dict{i+1}.pickle', 
                                            rus_sup='reduced' if i>0 else 'base')
        
        fig_sankey.write_image(f'./sankey_figs/sankey{i+1}.svg')
        
        
    # sankey_saver()