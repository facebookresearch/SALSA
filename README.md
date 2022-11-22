This repository contains code to recreate the results from our paper [*SALSA: Attacking Lattice Cryptography with Transformers*](https://arxiv.org/abs/2207.04785), which uses transformers to recover secrets from RLWE samples ($\mathbf{a}$, $b$).

## Quickstart

__Installation__: To get started, clone the repository to a machine that has at least one gpu. Create the necessary conda environment via ```conda create --name lattice_env --file requirements.txt``` and activate your shiny new environment via ```conda activate lattice_env```.

__Your first experiment__: Once you've done this, run ``` python3 train.py ```. This will run an experiment with the following RLWE parameters: $n=30$, $\sigma=3$, $h=3$, $q=251$. The model is an asymmetric gated universal transformer with encoder/decoder embedding $1024/512$, encoder/decoder attention heads $16/4$, encoder/decoder loops $2/8$, and $2$ layers in both the encoder and decoder. 

Our results on this architecture and problem parameters are reported in Figure 3 and Table 2 in the paper. 

__Parameters you can play with__: 
In our paper, we experiment with the following parameters. All others remained fixed with the defaults provided in ```src/train.py``` and ```src/envs/lattice.py```. 
- Model architecture parameters (defined in ```src/train.py```):
  - ```enc_emb_dim```: encoder's embedding dimension
  - ```dec_emb_dim```: decoder's embedding dimension
  - ```n_enc_layers```: number of layers in encoder
  - ```n_dec_layers```: number of layers in decoder
  - ```n_enc_heads```: number of attention heads in encoder
  - ```n_dec_heads```: number of attention heads in decoder
  - ```enc_loops```: number of loops through encoder (Universal Transformer parameter)
  - ```dec_loops```: number of loops through decoder (Universial Transformer parameter)
- Training parameters
  - ```epoch_size```: number of LWE samples per epoch
  - ```batch_size```: how many LWE samples per batch
- LWE problem parameters (defined in ```src/envs/lattice.py```)
  - ```N```: lattice dimension
  - ```Q```: prime modulus for LWE problem
  - ```reuse```: boolean flag determining if you reuse samples
  - ```num_reuse_samples```: number of LWE samples held for reuse at any given time
  - ```times_reused```: number of times you reuse a sample before discarding it
  - ```sigma```: stdev of error distribution used in LWE
  - ```hamming```: Hamming weight of binary LWE secret
  - ```input_int_base```: integer encoding base for transformer inputs
  - ```output_int_base```: integer encoding base for transformer outputs
  - ```percQ_bound```: use if you want to restrict input LWE elements to range of pQ, where 0 < p <= 1. 1 is default.



Example experimental settings can be found in ```./slurm_params/table2_n50.json```. These are the exact settings for the $n=50$ results in Table 2/Figure 3. Furthermore, these are the base settings for most other experiments in our paper. By modifying parameter settings in this file, you can recreate other experiments in our main paper body (e.g. modify  ```percQ_bound``` and ```hamming``` to recreate Table 7, or modify ```sigma``` to recreate Figure 5).

__Running sweeps with slurm__: To run sweeps on our cluster, we use slurm to parse the json files and farm out experiments to machines. If you add additional elements to the lists in the json files (e.g. ```N: [30, 40, 50]``` instead of just ```N: [50]```) and use an appropriate parser, you too can run sweeps locally. 


__Analyzing results__: If you have a large set of experiments you want to analyze, you can use ```./notebooks/analyze_experiment_logs.ipynb```. This will parse log file(s) from a given experiment(s). 

__Provided example__: In ```./checkpoint/example/test_run```, there is an example trained model (logs and parameter .pkl files) for the $N=30$, $Hamming=3$ setting. This model was trained by running ```train.py``` with all default parameters. The analysis scripts in notebook (e.g. ```analyze_experiment_logs.ipynb``` and ```run_distinguisher.py```) load this model by default as an example. 

To use ```run_distinguisher.py```, or load the trained model, you will need the model checkpoint, which can be downloaded [here](https://dl.fbaipublicfiles.com/SALSA/checkpoint.pth), and should be copied in ```./checkpoint/example/test_run```. (You can also train a new model and load it with ```run_distinguisher.py```).

## Citing this repo

Please use the following citation for this repository. 

```
@inproceedings{wenger2022salsa,
  title={SALSA: Attacking Lattice Cryptography with Transformers},
  author={Wenger, Emily and Chen, Mingjie and Charton, Fran{\c{c}}ois and Lauter, Kristin},
  booktitle = {Advances in Neural Information Processing Systems},
  volume={36},
  year={2022}
}
```

## License - Community

SALSA is licensed, as per the license found in the LICENSE file.
See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


