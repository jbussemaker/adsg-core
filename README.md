# Architecture Design Space Graph Core

[![Tests](https://github.com/jbussemaker/adsg-core/workflows/Tests/badge.svg)](https://github.com/jbussemaker/adsg-core/actions/workflows/tests.yml?query=workflow%3ATests)
[![PyPI](https://img.shields.io/pypi/v/adsg-core.svg)](https://pypi.org/project/adsg-core)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/adsg-core/badge/?version=latest)](https://adsg-core.readthedocs.io/en/latest/?badge=latest)

[GitHub Repository](https://github.com/jbussemaker/adsg-core) |
[Documentation](https://adsg-core.readthedocs.io/)

The Architecture Design Space Graph (ADSG) allows you to model design spaces using a directed graph that contains three
types of architectural choices:

- Selection choices: selecting among mutually-exclusive options, used for *selecting* which nodes are part of an
  architecture instance
- Connection choices: connecting one or more source nodes to one or more target nodes, subject to connection constraints
  and optional node existence (due to selection choices)
- Additional design variables: continuous or discrete, subject to optional existence (due to selection choices)

Modeling a design space like this allows you to:

- Model hierarchical relationships between choices, for example only activating a choice when another choice has some
  option selected, or restricting the available options for choices based on higher-up choices
- Formulate the design space as an optimization problem that can be solved using numerical optimization algorithms
- Generate architecture instances for a given design vector, automatically correct incorrect design variables, and get
  information about which design variables were active
- Implement an evaluation function (architecture instance --> metrics) and run the optimization problem

## Installation

First, create a conda environment (skip if you already have one):
```
conda create --name adsg python=3.10
conda activate adsg
```

Then install the package:
```
conda install numpy scipy~=1.9
pip install adsg-core
```

Optionally also install optimization algorithms ([SBArchOpt](sbarchopt.readthedocs.io/)):
```
pip install adsg-core[opt]
```

If you want to interact with the ADSG from a [Jupyter notebook](https://jupyter.org/):
```
pip install adsg-core[nb]
jupyter notebook
```

## Documentation

Refer to the [documentation](https://adsg-core.readthedocs.io/) for more background on the ADSG
and how to implement architecture optimization problems.

## Citing

If you use the ADSG in your work, please cite it:

Bussemaker, J.H., Ciampa, P.D., & Nagel, B. (2020). System architecture design space exploration: An approach to
modeling and optimization. In AIAA Aviation 2020 Forum (p. 3172).
DOI: [10.2514/6.2020-3172](https://doi.org/10.2514/6.2020-3172)

## Contributing

The project is coordinated by: Jasper Bussemaker (*jasper.bussemaker at dlr.de*)

If you find a bug or have a feature request, please file an issue using the Github issue tracker.
If you require support for using ADSG Core or want to collaborate, feel free to contact me.

Contributions are appreciated too:
- Fork the repository
- Add your contributions to the fork
  - Update/add documentation
  - Add tests and make sure they pass (tests are run using `pytest`)
- Read and sign the [Contributor License Agreement (CLA)](https://github.com/jbussemaker/adsg-core/blob/main/ADSG%20Core%20DLR%20Individual%20Contributor%20License%20Agreement.docx)
  , and send it to the project coordinator
- Issue a pull request

**NOTE:** Do *NOT* directly contribute to the `adsg_core.optimization.assign_enc` and `.sel_choice_enc` modules!
Their development happens in separate repositories:
- [AssignmentEncoding](https://github.com/jbussemaker/AssignmentEncoding)
- [SelectionChoiceEncoding](https://github.com/jbussemaker/SelectionChoiceEncoding)

Use `update_enc_repos.py` to update the code in this repository.

### Adding Documentation

```
pip install -r requirements-docs.txt
mkdocs serve
```

Refer to [mkdocs](https://www.mkdocs.org/) and [mkdocstrings](https://mkdocstrings.github.io/) documentation
for more information.
