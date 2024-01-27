# Architecture Design Space Graph Core

The Architecture Design Space Graph (ADSG) allows you to model design spaces using a directed graph that contains three
types of architectural choices:

- Selection choices: selecting among mutually-exclusive options, used for *selecting* which nodes are part of an
  architecture instance
- Connection choices: connecting one or more source nodes to one or more target nodes, subject to connection constraints
  and optional node existence (due to selection choices)
- Additional design variables: continuous or discrete, subject to optional existence (due to selection choices)

The library implements:

1. The directed graph, with:
    - Nodes: generic, selection-choice, connector, connection-choice, design variable, metric
    - Edges: derivation, incompatibility, connection, connection-exclusion
    - Choice constraints: linked values, permutations, unordered, unordered non-replacing
    - One or more derivation-start nodes
2. Mechanisms for making choices
    - Influence matrix
    - Graph operations for applying selection- and connection-choices
3. Mechanisms for formulating optimization problems
    - Design variable encoding of selection- and connection-choices
    - Hierarchy analysis and design vector correction
    - Design vector to graph conversion
    - Optimization problem statistics calculation
    - Bridge to the architecture optimization algorithms in [SBArchOpt](https://sbarchopt.readthedocs.io/)

To get started with the ADSG have a look at the [guide](guide.ipynb) or [API reference](api_adsg.md).
For detailed background information refer to the [theory](theory.md).

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

Optionally also install optimization algorithms ([SBArchOpt](https://sbarchopt.readthedocs.io/)):
```
pip install adsg-core[opt]
```

If you want to interact with the ADSG from a [Jupyter notebook](https://jupyter.org/):
```
pip install adsg-core[nb]
jupyter notebook
```

## Citing

If you use the ADSG in your work, please cite it:

Bussemaker, J.H., Ciampa, P.D., & Nagel, B. (2020). System architecture design space exploration: An approach to
modeling and optimization. In AIAA Aviation 2020 Forum (p. 3172).
DOI: [10.2514/6.2020-3172](https://doi.org/10.2514/6.2020-3172)

## Usage

Quick overview. Refer to tutorials.
