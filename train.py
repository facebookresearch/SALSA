# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import random
import argparse
import numpy as np
import torch
import os
import pickle
import sys

import src
from src.slurm import init_signal_handler, init_distributed_mode
from src.utils import bool_flag, initialize_exp
from src.model import check_model_params, build_modules
from src.envs import ENVS, build_env
from src.trainer import Trainer
from src.evaluator import Evaluator


np.seterr(all='raise')


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="debug",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # model parameters
    parser.add_argument("--enc_emb_dim", type=int, default=1024,
                        help="Encoder embedding layer size")
    parser.add_argument("--dec_emb_dim", type=int, default=512,
                        help="Decoder embedding layer size")
    parser.add_argument("--n_enc_layers", type=int, default=2,
                        help="Number of Transformer layers in the encoder")
    parser.add_argument("--n_dec_layers", type=int, default=2,
                        help="Number of Transformer layers in the decoder")
    parser.add_argument("--n_enc_heads", type=int, default=16,
                        help="Number of Transformer encoder heads")
    parser.add_argument("--n_dec_heads", type=int, default=4,
                        help="Number of Transformer decoder heads")
    parser.add_argument("--n_enc_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer encoder")
    parser.add_argument("--n_dec_hidden_layers", type=int, default=1,
                        help="Number of FFN layers in Transformer decoder")
    parser.add_argument("--xav_init", type=bool_flag, default=False,
                        help="Xavier initialization for transformer parameters")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="GELU initialization in FFN layers (else RELU)")
    
    parser.add_argument("--norm_attention", type=bool_flag, default=False,
                        help="Normalize attention and train temperaturee in Transformer")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=False,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")

    # universal transformer parameters
    parser.add_argument("--enc_loop_idx", type=int, default=1,
                        help="Index of the encoder shared weight layers (-1 for none)")
    parser.add_argument("--dec_loop_idx", type=int, default=1,
                        help="Index of the decoder shared weight layers (-1 for none)")
    parser.add_argument("--enc_loops", type=int, default=2,
                        help="Fixed/max nr of train passes through the encoder loop")
    parser.add_argument("--dec_loops", type=int, default=8,
                        help="Fixed/max nr of train passes through the decoder loop")
    parser.add_argument("--gated", type=bool_flag, default=True,
                        help="Gated loop layers")
    parser.add_argument("--enc_gated", type=bool_flag, default=False,
                        help="All encoder layers gated")
    parser.add_argument("--dec_gated", type=bool_flag, default=False,
                        help="All decoder layers gated")    
    parser.add_argument("--scalar_gate", type=bool_flag, default=False,
                        help="Scalar gates")
    # ACT
    parser.add_argument("--enc_act", type=bool_flag, default=False,
                        help="Encoder looped layer ACT")
    parser.add_argument("--dec_act", type=bool_flag, default=False,
                        help="Decoder looped layer ACT")
    parser.add_argument("--act_threshold", type=float, default=0.01,
                        help="Prob threshold for ACT")
    parser.add_argument("--act_ponder_coupling", type=float, default=0.05,
                        help="Ponder loss coupling for ACT")

    # training parameters
    parser.add_argument("--env_base_seed", type=int, default=0,
                        help="Base seed for environments (-1 to use timestamp seed)")
    parser.add_argument("--max_len", type=int, default=512,
                        help="Maximum sequences length")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Number of sentences per batch")
    parser.add_argument("--batch_size_eval", type=int, default=64,
                        help="Number of sentences per batch during evaluation")
    parser.add_argument("--optimizer", type=str, default="adam_warmup,lr=0.00001,warmup_updates=8000,weight_decay=0.99",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--weighted_loss", type=bool_flag, default=False,
                        help='Weight loss to emphasize higher bits?')
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=300000,
                        help="Epoch size / evaluation frequency")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="_valid_lattice_xe_loss,1,75",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--secret_stop", type=bool_flag, default=True, help="stop when secret is found?")
    parser.add_argument("--validation_metrics", type=str, default="_valid_lattice_xe_loss",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")
    parser.add_argument("--num_workers", type=int, default=10,
                        help="Number of CPU workers for DataLoader")

    # export data / reload it
    parser.add_argument("--export_data", type=bool_flag, default=False,
                        help="Export data and disable training.")
    parser.add_argument("--reload_data", type=str, default="",
                        help="Load dataset from the disk (task1,train_path1,valid_path1,test_path1;task2,train_path2,valid_path2,test_path2)")
    parser.add_argument("--reload_size", type=int, default=-1,
                        help="Reloaded training set size (-1 for everything)")
    parser.add_argument("--batch_load", type=bool_flag, default=False,
                        help="Load training set by batches (of size reload_size).")

    # environment parameters
    parser.add_argument("--env_name", type=str, default="lattice",
                        help="Environment name")
    ENVS[parser.parse_known_args()[0].env_name].register_args(parser)

    # tasks
    parser.add_argument("--tasks", type=str, default="lattice",
                        help="Tasks")

    # beam search configuration
    parser.add_argument("--beam_eval", type=bool_flag, default=True,
                        help="Evaluate with beam search decoding.")
    parser.add_argument("--beam_eval_train", type=int, default=0,
                        help="At training time, number of validation equations to test the model on using beam search (-1 for everything, 0 to disable)")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--beam_length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--beam_early_stopping", type=bool_flag, default=True,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # reload pretrained model / checkpoint
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")
    parser.add_argument("--freeze_embeddings", type=bool_flag, default="False",
                        help="Freeze embeddings for retraining?")

    # evaluation
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")
    parser.add_argument("--eval_from_exp", type=str, default="",
                        help="Path of experiment to use")
    parser.add_argument("--eval_data", type=str, default="",
                        help="Path of data to eval")
    parser.add_argument("--eval_verbose", type=int, default=0,
                        help="Export evaluation details")
    parser.add_argument("--eval_verbose_print", type=bool_flag, default=False,
                        help="Print evaluation details")

    # debug
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # CPU / multi-gpu / multi-node
    parser.add_argument("--cpu", type=bool_flag, default=False,
                        help="Run on CPU")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--windows", type=bool_flag, default=False,
                        help="Windows version (no multiprocessing for eval)")
    parser.add_argument("--nvidia_apex", type=bool_flag, default=False,
                        help="NVIDIA version of apex")

    return parser


def main(params):

    # Initialize the dump path. 
    if params.export_data:
        e = '_error' if params.error else ''
        params.dump_path = params.dump_path
        params.exp_name = 'data'
        params.exp_id = f'N={params.N}_Q={params.Q}{e}/'

    # initialize the multi-GPU / multi-node training
    # initialize experiment / SLURM signal handler for time limit / pre-emption
    init_distributed_mode(params)
    logger = initialize_exp(params)
    if params.is_slurm_job:
        init_signal_handler()

    # CPU / CUDA
    if params.cpu:
        assert not params.multi_gpu
    else:
        assert torch.cuda.is_available()
    src.utils.CUDA = not params.cpu

    # build environment / modules / trainer / evaluator
    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)
    
    # evaluation
    if params.eval_only:
        scores = evaluator.run_all_evals()
        for k, v in scores.items():
            logger.info("%s -> %.6f" % (k, v))
        logger.info("__log__:%s" % json.dumps(scores))
        exit()

    # training
    for _ in range(params.max_epoch):

        logger.info("============ Starting epoch %i ... ============" % trainer.epoch)

        trainer.n_equations = 0

        while trainer.n_equations < trainer.epoch_size:

            # training steps
            for task_id in np.random.permutation(len(params.tasks)):
                task = params.tasks[task_id]
                if params.export_data:
                    trainer.export_data(task)
                else:
                    trainer.enc_dec_step(task)
                trainer.iter()

        logger.info("============ End of epoch %i ============" % trainer.epoch)

        # evaluate perplexity
        scores = evaluator.run_all_evals()

        # print / JSON log
        if params.is_master:
            logger.info("__log__:%s" % json.dumps(scores))

        # end of epoch
        trainer.save_best_model(scores)
        trainer.save_periodic()
        trainer.end_epoch(scores)


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # Eval-only mode.
    if params.eval_only and params.eval_from_exp != "":
        # read params from pickle
        pickle_file = params.eval_from_exp + "/params.pkl"
        assert os.path.isfile(pickle_file)
        pk = pickle.load(open(pickle_file, 'rb'))
        pickled_args = pk.__dict__
        del pickled_args['exp_id']
        for p in params.__dict__:
            if p in pickled_args:
                params.__dict__[p] = pickled_args[p]

        params.eval_only = True
        params.reload_model = params.eval_from_exp + '/best-' + params.validation_metrics + '.pth'
        params.eval_size = None
        params.reload_data = params.tasks + ',' + params.eval_data + ',' + params.eval_data + ',' + params.eval_data
        params.is_slurm_job = False
        params.local_rank = -1

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        if params.exp_id == '':
            params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True

    # check parameters
    check_model_params(params)

    # run experiment
    main(params)
