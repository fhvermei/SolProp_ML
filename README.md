# SolProp_ML
Code that combines machine learning and thermodynamics for the prediction of solubility related properties.
To run this code, and to make predictions, you need the ML models.
Code including ML models is available as a conda package (link), or through an online webtool (link).
The models in SolProp_ML are trained with databases from the SolProp data collection (link).

## Requirements


## Make ML predictions and solubility calculations

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
                               
### Code for solubility calculations in organic solvents at different temperatures (logS [log10(mol/L)])
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







