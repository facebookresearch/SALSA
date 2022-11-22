# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
import os
import torch
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pickle
import ast
import re
from scipy import stats
import time

sys.path.append('../')
import src
from train import *
from src.slurm import init_signal_handler, init_distributed_mode
#torch.cuda.set_device(0)
from src.utils import to_cuda, initialize_exp
from src.envs import ENVS, build_env
from src.model import check_model_params, build_modules
from src.trainer import Trainer
from src.evaluator import Evaluator
import ast
import numpy as np

from concurrent.futures import ProcessPoolExecutor

def get_params(path, dump_path='./checkpoints'):
    params = pickle.load(open(path + 'params.pkl', 'rb'))
    params.reload_checkpoint = path + 'checkpoint.pth' 
    params.local_rank = -1
    params.master_port = -1
    params.eval_only = True
    params.deterministic = False #True
    params.dump_path = dump_path
    params.is_slurm_job = False
    params.batch_size = 5

    if params.n_enc_layers <= params.enc_loop_idx:
        params.enc_loop_idx = 0
        params.enc_loops = 0
    if params.n_dec_layers <= params.dec_loop_idx:
        params.dec_loop_idx = 0
        params.dec_loops = 0
        params.gated = False

    params.enc_act = False
    params.dec_act = False
    params.reuse = False
    params.freeze_embeddings = False
    params.num_workers = 1
    params.eval_only = True
    params.env_base_seed = int(time.time()) # you can change this if you want. 
    return params


def get_samples(i, num_samples, random, rng, params, env, g=0, sec_idx=0):
    ''' 
    Given an advantage and a "random" flag, output samples for use in distinguisher. 
    Also will encode the samples for use in model. 
    '''
    
    if not random: 
        A_s = []
        A_s_orig = []
        B_s = []
        while len(A_s) < num_samples:
            A, B = env.generator.generate(rng, sec_idx, params.N)
            c = np.random.randint(0, params.Q, size=params.N)
            # EJW 11/4/22
            A_orig = A.copy() # so we can compare performance against model prediction on ORIGINAL A, rather than b. 
            A[:,i] += c # add C to the ith coordinate of A
            updatedA = A % params.Q
            updated_A_orig = A_orig % params.Q   # EJW 11/4/22
            updatedB = (B + g*c) % params.Q
            A_s.extend([list(a) for a in updatedA])
            A_s_orig.extend([list(a) for a in updated_A_orig])
            B_s.extend(list(updatedB))
    else:
        A_s = []
        B_s = []
        while len(A_s) < num_samples:
            A, B = np.random.randint(0, params.Q, size=(params.N, params.N)), np.random.randint(0, params.Q, size=params.N)
            A_s.extend([list(a) for a in A])
            B_s.extend(list(B))
        A_s_orig = A_s.copy() # EJW 11/4/22 dummy variable
            
    # Make data usable. 
    A_s = A_s[:num_samples]
    A_s_orig = A_s_orig[:num_samples] # EJW 11/4/22 This is all the A samples but without c transformation. 
    B_s = B_s[:num_samples]
    updatedA = np.array(A_s)
    updatedB = np.array(B_s)
    updatedA_orig = np.array(A_s_orig) # EJW 11/4/22

    # Encode data
    encA, lenA, encB, lenB, y = encode_data(updatedA,updatedB,params,env)
    if not random: 
        # Then encode/return the other data
        encA_orig, lenA_orig, encB_orig, lenB_orig, y_orig = encode_data(updatedA_orig, updatedB, params, env)
    else: 
        encA_orig, lenA_orig, encB_orig, lenB_orig, y_orig = None, None, None, None, None


    return (encA, lenA, encB, lenB, y), (encA_orig, lenA_orig, encB_orig, lenB_orig, y_orig)

def idx_to_infix(env, idx, input=True):
    """
    Convert an indexed prefix expression to SymPy.
    """
    prefix = [env.id2word[wid] for wid in idx]
    infix = env.input_to_infix(prefix) if input else env.output_to_infix(prefix)
    return infix

def check_hypothesis(eq):
    """
    Check a hypothesis for a given equation and its solution.
    """
    env = Evaluator.ENV

    src = [env.id2word[wid] for wid in eq["src"]]
    tgt = [env.id2word[wid] for wid in eq["tgt"]]
    hyp = [env.id2word[wid] for wid in eq["hyp"]]

    # update hypothesis
    eq["src"] = env.input_to_infix(src)
    eq["tgt"] = tgt
    eq["hyp"] = hyp
    try:
        m, diff, bw = env.check_prediction(src, tgt, hyp)
    except Exception:
        m = -1
    eq["is_valid"] = m
    return eq

def encode_data(A, B, params, env):
    # Get the encoded version
    prefix = []
    x = np.array([prefix + env.input_encoder.encode(el) for el in A])
    y = np.array([env.output_encoder.write_int(el) for el in B])

    # Make them tensors
    int_len =  len(env.input_encoder.write_int(0))
    sep = 0 if env.input_encoder.no_separator else 1
    nb_eqs = [env.code_class(xi, yi, int_len+sep) for xi, yi in zip(x, y)]
    x = [torch.LongTensor([env.word2id[w] for w in seq]) for seq in x]
    y = [torch.LongTensor([env.word2id[w] for w in seq]) for seq in y]
    x, x_len = env.batch_sequences(x)
    y, y_len = env.batch_sequences(y)

    x1, len1, x2, len2, nb_ops = x, x_len, y, y_len, nb_eqs
    # Set things up. 
    alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
    pred_mask = (alen[:, None] < len2[None] - 1)  # do not predict anything given the last target word
    y = x2[1:].masked_select(pred_mask[:-1])

    x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
    return x1_, len1_, x2, len2, y

def load_model(params, args):
    # get the model

    init_distributed_mode(params)
    logger = initialize_exp(params)

    if params.cpu:
        assert not params.multi_gpu
    else:
        assert torch.cuda.is_available()
    src.utils.CUDA = not params.cpu

    env = build_env(params)
    modules = build_modules(env, params)
    trainer = Trainer(modules, env, params)
    evaluator = Evaluator(trainer)

    # Get the models. 
    encoder = (
        modules["encoder"].module
        if params.multi_gpu
        else modules["encoder"]
    )
    decoder = (
        modules["decoder"].module
        if params.multi_gpu
        else modules["decoder"]
    )
    encoder.eval()
    decoder.eval()
    try:
        # Load from args.basepath since this is where checkpoint is stored -- may not match where original model was stored. 
        trainer.env.generator.secrets= parse_secret(args.basepath + "/train.log")
    except: 
        print('getting deterministic secret from log') #, params.dump_path)
        trainer.env.generator.secrets = parse_secret(args.basepath + "/train.log")

    # get a dataloader. 
    trainer.dataloader = {'lattice': iter(env.create_train_iterator('lattice', trainer.data_path, params))}
    
    return encoder, decoder, env, modules, trainer, evaluator


def parse_secret(logfile):
    log = open(logfile, 'r').readlines()
    sec = [(i, l) for (i, l) in enumerate(log) if ' secrets: ' in l][0]
    secline, sec = sec[0], sec[1].split(' secrets: ')[1].rstrip()
    i = 1
    while not sec.endswith('])]'):
        sec += log[secline+i].rstrip()
        i += 1
    sec = re.sub('  +', ' ', sec)
    mod_sec = ast.literal_eval(sec.replace('array(', '').replace(')', ''))
    return mod_sec


def run_beam_generation(x1, x2, len1, len2, params, trainer, encoder, decoder):
    # Run evaluation on these. 
    encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
    _, _, generations = decoder.generate_beam(
        encoded.transpose(0, 1),
        len1,
        beam_size=params.beam_size,
        length_penalty=params.beam_length_penalty,
        early_stopping=params.beam_early_stopping,
        max_len=4096,
    )

    inputs = []
    for i in range(len(generations)):
        #if valid[i] and params.eval_verbose < 2:
        #    continue
        for j, (score, hyp) in enumerate(
            sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)
        ):
            inputs.append(
                {
                    "i": i,
                    "j": j,
                    "score": score,
                    "src": x1[1 : len1[i] - 1, i].tolist(),
                    "tgt": x2[1 : len2[i] - 1, i].tolist(),
                    "hyp": hyp[1:].tolist(),
                    "task": 'lattice',
                }
            )

    bs = len(len1)

    outputs = []
    if params.windows is True:
        for inp in inputs:
            outputs.append(check_hypothesis(inp))
    else:
        with ProcessPoolExecutor(max_workers=20) as executor:
            for output in executor.map(check_hypothesis, inputs, chunksize=1):
                outputs.append(output)

    # save evaluation details
    beam_log = {}
    for i in range(len(len1)):
        sr = idx_to_infix(trainer.env, x1[1 : len1[i] - 1, i].tolist(), True)
        tgt = idx_to_infix(trainer.env, x2[1 : len2[i] - 1, i].tolist(), False)
        #if valid[i]:
        beam_log[i] = {"src": sr, "tgt": tgt, "hyps": [(tgt, None, True)]}


    valid = torch.zeros(params.N)
    n_correct = 0
    for i in range(bs):

        # select hypotheses associated to current equation
        gens = sorted([o for o in outputs if o["i"] == i], key=lambda x: x["j"])
        #assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (
         #   i in beam_log) == valid[i]
        if len(gens) == 0:
            continue

        # source / target
        sr = gens[0]["src"]
        tgt = gens[0]["tgt"]
        beam_log[i] = {"src": sr, "tgt": tgt, "hyps": []}

        # for each hypothesis
        for j, gen in enumerate(gens):

            # sanity check
            assert (
                gen["src"] == sr
                and gen["tgt"] == tgt
                and gen["i"] == i
                and gen["j"] == j
            )

            # if hypothesis is correct, and we did not find a correct one before
            is_valid = gen["is_valid"]
            is_b_valid = is_valid >= 0.0 and is_valid == 1

            # update beam log
            beam_log[i]["hyps"].append((gen["hyp"], gen["score"], is_b_valid))
    return beam_log
