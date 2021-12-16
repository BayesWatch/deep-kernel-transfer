Code for reproducing the **periodic functions** experiments reported in the paper.

The code needs some polishing and some hyperparameters must be set by changing variables directly at the code level.

- For the in-range condition set `test_range=(-5.0, 5.0)`
- For the out-of-range condition set `test_range=(-5.0, 10.0)`

For the out-of-range condition you should also change the horizontal axis limit from `plt.ylim(-6.0, 6.0)` to `plt.ylim(-6.0, 11.0)`

- The file [./train_DKT.py] contains the code for training and evaluating a DKT model.
- The file [./train_FT.py] contains the code for training and evaluating the baseline Feature Transfer model.
- The file [./train_MAML.py] contains the code for training the and evaluating the MAML model.


Acknowledgments
---------------

The MAML and Feature Transfer code has been adapted from:

- [https://github.com/vmikulik/maml-pytorch](https://github.com/vmikulik/maml-pytorch)
- [https://github.com/cbfinn/maml](https://github.com/cbfinn/maml)
