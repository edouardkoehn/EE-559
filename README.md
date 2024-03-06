# project-template
Template repostiory for accelerate your project
## Installation
- Clone the repo

```bash
git clone https://github.com/edouardkoehn/WM_Atlas.git
```
- Create your virtual env
```bash
conda create -n wm python=3
conda activate wm
```
- Install poetry
```bash
pip install poetry
```
- install the modul and set up the precommit
```bash
poetry install
poetry run pre-commit install
poetry env info
```

## Managing dependancies
- to add an dependancy with poetry for exemple numpy
```bash
poetry add numpy
```
It would automatically install and udpate the pyproject.toml file. If you didn't specifie it, it would add it to the tool.poetry.dependencies.

- to add an dependancy with poetry to a specific group
```bash
poetry add pytest --group test
```
It would automatically install and udpate the pyproject.toml file. The dependancie would be attached to the tool.poetry.dependcies.group.
[More info about poetry](https://python-poetry.org/docs/managing-dependencies/#adding-a-dependency-to-a-group)

## Pre-commit config
Pre-commit hook are small software that are called when you do a git commit. Those hooks do several test on you code (formating, imports, etc. ). If not all the tests have been passed, the pre-commmit would not allow you to do the commit. You would have to resolve each of the error before beeing able to do the commit.

Good pratrice with pre-commit :
1) Git commit regularly
2) Never git commit when you are in a rush
