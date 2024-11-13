# Natural gas crisis in europe 2022: environmental implications

## Description

This repository contains code to reproduce the results in the scientific article <i>Santos et al. (2024), Environmental impacts of restructuring the EU’s natural gas supply and consumption: Learnings from the 2022 energy crisis.</i>


## Usage

1. New conda environment:
We recommend starting a new environment for the calculations, here called `ng_lca`. Install [`brightway2`](https://github.com/brightway-lca/brightway2), [`activity-browser`](https://github.com/LCA-ActivityBrowser/activity-browser), and the other dependencies to run the notebooks as follows:

    ```
    conda create -n ng_lca -c conda-forge -c cmutel python=3.9 brightway2=2.4.3 activity-browser=2.9.0 -y
    conda activate ng_lca
    pip install -r requirements.txt
    ```
<br>

2. New brightway2 project:
Start a new project in brightway2 for this analysis with the following code in a python terminal within the `ng_lca` environment. 
Make sure you have ecoinvent 3.9.1 cutoff available in your machine and change the `folder` variable to the path to the ecoinvent v3.9.1 datasets.

    ```python
    import brightway2 as bw2

    BW_PROJECT = 'NG_LCA_EU27'    # <- change the name of the project if desired
    bw2.projects.set_current(BW_PROJECT)   # Creating/accessing the project

    folder = '.../ecoinvent_3.9.1_cutoff/datasets' # <- insert the path to ecoinvent3.9 here
    ei39 = bw2.SingleOutputEcospold2Importer(folder, 'ecoinvent 3.9.1 cutoff')

    bw2.bw2setup()
    ei39.apply_strategies()
    ei39.statistics()
    ei39.write_database()
    ```

    Alternatively, the same brightway project can be created in the `Basic setup` section of the `0_implement_EU27_NG_scenario_in_Ecoinvent.ipynb` notebook.
<br>

3. Creating modified Ecoinvent database:
Then, you can run the `0_implement_EU27_NG_scenario_in_Ecoinvent.ipynb` notebook to generate a modified version of econinvent, where the activities of natural gas as well as its supply mix are changed.
<br>

4. Run LCIA calculations:
Now, you can reproduce the results from the publication with the [`1_ng_crisis_aftermath.ipynb`](1_ng_crisis_aftermath.ipynb), [`2_ng_lca_ClimateChange.ipynb`](2_ng_lca_ClimateChange.ipynb), [`3_ng_lca_EnvFootprint.ipynb`](3_ng_lca_EnvFootprint.ipynb), and [`4_ng_suppliers.ipynb`](4_ng_suppliers.ipynb) notebooks.

## Cite us

DOI: in press

```bibtex
@article{iscience2024nglca,
      title={Environmental impacts of restructuring the EU’s natural gas supply and consumption: Learnings from the 2022 energy crisis}, 
      author={Santos, Lucas F and Istrate, Robert and Mac Dowell, Niall and Guillén-Gosálbez, Gonzalo},
      journal={iScience},
      year={2024},
      volume={in-press},
      publisher={Elsevier}
}
```