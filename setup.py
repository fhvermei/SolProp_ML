from distutils.core import setup

setup(
    name="solvation_predictor",
    version="1.0",
    packages=[
        "solvation_predictor",
        "solvation_predictor.data",
        "solvation_predictor.train",
        "solvation_predictor.models",
        "solvation_predictor.features",
        "solvation_predictor.solubility",
    ],
    package_data={
        "solvation_predictor": ["trained_models/*/*.pt", "solubility/*.json"]
    },
    url="https://github.com/fhvermei/SolPropML",
    license="MIT",
    author="Florence Vermeire, Yunsie Chung, William Green",
    author_email="florence.vermeire@ugent.be",
    description="Package to make solubility predictions with ML",
)
