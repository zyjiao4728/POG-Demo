# POG-Demo

![sys-vesrion](https://img.shields.io/badge/Ubuntu-20.04-blue) 

Sequential Manipulation Planning on Scene Graph

## 1. Installation

Clone the repository

```bash
git clone --recursive <git-package-url>
```

where `<git-package-url>` the git repo URL of our package.

Install the sdf submodule

```bash
cd POG-Demo/sdf
pip install -e .
```

Install the pog module

```bash
cd ..
pip install -e .
```
## 2. Run Examples
We provid two examples which correspond to exp 1 and 2 on paper.

``` bash
python pog_example/iros_2022_exp/exp1/main.py 10 -viewer
python pog_example/iros_2022_exp/exp2/main.py -viewer
```
## 3. Website
Please checkout the [project website](https://sites.google.com/view/planning-on-graph/) for more details and video demos

