# ResQ
Neural ODE based ResNets on neutral atom quantum computers

# Usage
Training code is written in the Julia programming language, with most of the code being centered around Bloqade.jl, a package for simulating neutral atom systems.
The training code is contained within training.jl, and can be run in the terminal using 

```
julia--project=ResQ_env training.jl <dataset> <lattice shape> <lattice spacing> <classification task>
```
Prior to the first time you run this, you may need to install the relevant packages. To do this, enter Pkg mode in the Julia REPL (]), run `activate  ResQ_env` and then run `instantiate`. After this all the dependencies should be installed.


# Code Licensing
© 2025 Rice University subject to Creative Commons Attribution 4.0 International license (Creative Commons — Attribution 4.0 International — CC BY 4.0)

Contact ptl@rice.edu for permissions.
