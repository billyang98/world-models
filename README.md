# Dyna World Models

This is a pytorch implementation of a Dyna variant of Ha and Schmidhuber's World Models paper.

Rather than generating trajectories from a random policy once and learning a model from only those trajectories
we iteratively generate trajectories from the current best policy, learn a better model with those trajectories,
learn a better policy from that model, and repeat.

# Credits
This is a fork of https://github.com/ctallec/world-models

