# SolProp_ML
Code that combines machine learning and thermodynamics for the prediction of solubility related properties.
To run this code, and to make predictions, you need the ML models.
Code including ML models is available as a [conda package](https://anaconda.org/fhvermei/solprop_ml), 
or through an [online webtool](https://rmg.mit.edu/database/solvation/search/).
The models in SolProp_ML are trained with databases from the [SolProp data collection](https://zenodo.org/record/5970538).


## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
  * [Option 1: Installing as a conda package](#option-1-installing-as-a-conda-package)
  * [Option 2: Installing from source](#option-2-installing-from-source)
- [Supported solvents and solutes](#supported-solvents-and-solutes)
- [Definitions of inputs and outputs](#definitions-of-inputs-and-outputs)
- [Sample Jupyter Notebook files for solubility and property prediction](#sample-jupyter-notebook-files-for-solubility-and-property-prediction)
- [Example of making ML predictions and solubility calculations from a csv input file](#example-of-making-ml-predictions-and-solubility-calculations-from-a-csv-input-file)
  * [Data reading](#data-reading)
  * [Only predict properties with ML](#only-predict-properties-with-ml)
  * [Code for solubility calculations in organic solvents at different temperatures](#code-for-solubility-calculations-in-organic-solvents-at-different-temperatures)
- [Web Interface](#web-interface)
- [How to Cite](#how-to-cite)
- [License Information](#license-information)

## Requirements
SolProp_ML has been so far tested to work on Mac and Linux OS. It may not work on Windows.

## Installation
SolProp_ML can either be installed from conda or from source (i.e. directly from this git repo).

Both options require conda, so first install Miniconda from [https://conda.io/miniconda.html](https://conda.io/miniconda.html).

### Option 1: Installing as a conda package
1. `conda install -c fhvermei solprop_ml`
2. `pip install git+https://github.com/bp-kelley/descriptastorus`

### Option 2: Installing from source
1. `git clone https://github.com/fhvermei/SolProp_ML.git`
2. `cd SolProp_ML` (`SolProp_ML` is the path to where you cloned the git repository)
3. `conda env create -f environment.yml`
4. `conda activate env_solprop`
5. `pip install git+https://github.com/bp-kelley/descriptastorus`
6. `pip install -e .` or add the path to the `SolProp_ML` to the `PYTHONPATH` in your .bash_profile or .bashrc file
7. Download the machine learning model files from [here](https://zenodo.org/record/5970538).
8. Copy the `Gsolv`, `Hsolv`, and `Saq` folders from the `Machine_learning_models` folder of the downloaded file and place them under `SolProp_ML/solvation_predictor/trained_models/`

## Supported solvents and solutes
SolProp_ML currently supports prediction for only electrically neutral solute compounds containing H, B, C, N, O, S, P, 
F, Cl, Br, and I and nonionic liquid solvents. Mixture solvents or solutes are not supported. Predictions for any
out-of-range solvents and solutes won't be reliable.

## Definitions of inputs and outputs
The definitions of prediction inputs and outputs are described in the sample jupyter notebook files.
Please refer to the `.ipynb` files located under `SolProp_ML/sample_files/` and `input.csv` files located under each 
example folder.

Input and output definitions are also listed under [Example of making ML predictions and solubility calculations from a csv input file](#example-of-making-ml-predictions-and-solubility-calculations-from-a-csv-input-file).

## Sample Jupyter Notebook files for solubility and property prediction
Sample Jupyter Notebook (ipython) files for doing solubility and property predictions are located under `SolProp_ML/sample_files/`.
There are two Jupyter notebook files, `example_script.ipynb` and `example_with_csv_inputs.ipynb`.
Refer to those files to learn how to use the SolProp_ML model.

## Example of making ML predictions and solubility calculations from a csv input file

### Data reading
The solute/solvent smiles or inchis are read from (1) a csv file, or (2) a pandas dataframe. The columns names must be 'solvent', 'solute', 'temperature' 'reference_solubility', and 'reference_solvent'.
Only the column 'solute' is required, all other columns are optional depending on the property that you want to predict or calculate.

### Only predict properties with ML
Options:

path --> path to csv input file

df --> pandas dataframe instead of path to csv file

gsolv --> predict the solvation free energy of a 'solute' in a 'solvent' at 298K in [kcal/mol]

hsolv --> predict the solvation enthalpy of a 'solute' in a 'solvent' at 298K in [kcal/mol]

saq --> predict the aqueous solubility of a 'solute' in water at 298K in [log10(mol/L)]

solute_parameters --> predict the Abraham Solute Parameters of a 'solute'

reduced_number --> use a reduced number of models in the model ensemble for fast predictions with a lower accuracy

validate_smiles --> validate the smiles before predictions and convert InChI to smiles (takes time)

export_csv --> path to csv file if you want to export results

logger --> path to log file

from ... import calculate_solubility.predict_property

predictions = predict_property(path='./input.csv',
                               df=None,
                               gsolv=True/False,
                               hsolv=True/False,
                               saq=True/False,
                               solute_parameters=True/False,
                               reduced_number=True/False,
                               validate_smiles=True/False,
                               export_csv='./results.csv',
                               logger='./log.log')
                               
### Code for solubility calculations in organic solvents at different temperatures
This predicts logS, which is in log10 of mol solute per L solution unit (log10(mol/L)). 

Options:

path --> path to csv input file

df --> pandas dataframe instead of path to csv file

reduced_number --> use a reduced number of models in the model ensemble for fast predictions with a lower accuracy

validate_smiles --> validate the smiles before predictions and convert InChI to smiles (takes time)

calculate_aqueous --> use aqueous solubility to calculate solubility in other solvents, even if reference solubilities are available

export_csv --> path to csv file if you want to export results

export_detailed_csv --> path to csv file if you want to export results

logger --> path to log file

from ... import calculate_solubility

results = calculate_solubility(path='./input.csv',
                               df=None,
                               validate_smiles=True/False,
                               calculate_aqueous=True/False,
                               reduced_number=True/False,
                               export_csv='./results.csv',
                               export_detailed_csv='./detailed_results.csv',
                               logger='./log.log')

## Web Interface
A user-friendly online webtool is available on [here](https://rmg.mit.edu/database/solvation/search/) under
the <i>Solid Solubility Prediction</i> tool.

## How to Cite
If you use this software for research, please cite the paper (link to be added soon) as follows:

Vermeire, F. H.; Chung, Y.; Green, W. H. Predicting Solubility Limits of Organic Solutes for a Wide Range of Solvents
and Temperatures. <b>2021</b>, [<i>submitted</i>]

## License Information
SolProp_ML is a free, open-source software package distributed under the 
[Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/legalcode).





