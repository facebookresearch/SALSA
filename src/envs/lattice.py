# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from logging import getLogger
import os
import io
import sys
import time

# import math
import numpy as np
import src.envs.encoders as encoders
import src.envs.generators as generators


import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from ..utils import bool_flag


SPECIAL_WORDS = ["<s>", "</s>", "<pad>"]
logger = getLogger()

class InvalidPrefixExpression(Exception):
    def __init__(self, data):
        self.data = data

    def __str__(self):
        return repr(self.data)


class LatticeEnvironment(object):

    TRAINING_TASKS = {"lattice"}

    def __init__(self, params):
        
        self.max_len = params.max_len
        self.operation = params.operation
        self.error = params.error
        self.secret = params.secret.split(',')
        for i in range(len(self.secret)): self.secret[i] = int(self.secret[i])
        
        if self.operation == "modular_multiply":
            self.generator = generators.ModularMultiply(params, self.secret)
        elif self.operation == "circ_rlwe":
            # Random secret
            self.generator = generators.RLWE(params, np.random.default_rng([int(params.env_base_seed)**2, int(time.time())]))
        else:
            logger.error(f"Unknown operation {self.operation}")

        self.output_encoder = encoders.Encoder(params, True, output=True)
        self.input_encoder = encoders.Encoder(params, False, output=False)
        self.max_output_length = params.max_output_len
        self.export_data = params.export_data
        
        # vocabulary
        self.common_symbols = ['|', '+'] # added ab separator
        self.words = SPECIAL_WORDS + self.common_symbols + sorted(list(
            set(self.output_encoder.symbols + self.input_encoder.symbols) # + [str(el) for el in params.secret_sizes])
        ))
        self.id2word = {i: s for i, s in enumerate(self.words)}
        self.word2id = {s: i for i, s in self.id2word.items()}
        assert len(self.words) == len(set(self.words))

        # number of words / indices
        self.n_words = params.n_words = len(self.words)
        self.eos_index = params.eos_index = 0
        self.pad_index = params.pad_index = 1
        logger.info(f"vocabulary: {len(self.word2id)} words")
        if len(self.word2id) < 1000:
            logger.info(f"words: {self.word2id}")

        # matrix counter for sample prefix
        self.mat_count = 0 


    # def update_curriculum_params(self):
    #     if self.hamming_curriculum and (self.curr_hamming < self.max_hamming):
    #         self.curr_hamming += 1 # TODO make this a parameter
    #         self.generator.updateSecretKey(self.curr_hamming)
    #     elif (self.generator.percQ_bound < 1) and self.percQ_increase > 0:
    #         self.generator.percQ_bound += self.percQ_increase
    #         logger.info(f'Updated percQ bound to {self.generator.percQ_bound}')
    #         if self.generator.percQ_bound > 1:
    #             self.generator.percQ_bound = 1
    
    def batch_sequences(self, sequences):
        """
        Take as input a list of n sequences (torch.LongTensor vectors) and return
        a tensor of size (slen, n) where slen is the length of the longest
        sentence, and a vector lengths containing the length of each sentence.
        """
        lengths = torch.LongTensor([len(s) + 2 for s in sequences])
        sent = torch.LongTensor(lengths.max().item(), lengths.size(0)).fill_(
            self.pad_index
        )
        assert lengths.min().item() > 2

        sent[0] = self.eos_index
        for i, s in enumerate(sequences):
            sent[1 : lengths[i] - 1, i].copy_(s)
            sent[lengths[i] - 1, i] = self.eos_index

        return sent, lengths

    def input_to_infix(self, lst):
        m = self.input_encoder.decode(lst)
        if m is None:
            return "Invalid"
        return str(m)

    def output_to_infix(self, lst):
        m = self.output_encoder.decode(lst)
        if m is None:
            return "Invalid"
        return str(m)

    def gen_expr(self, data_type=None):
        """
        Generate pairs of matrices and inverses
        Encode this as a prefix sentence
        """
        idx = 0

        currN = self.generator.N

        # Generate the data. 
        gen = self.generator.generate(self.rng, idx, currN)

        if gen is None:
            return None
        x_data, y_data  = gen

        # If you are exporting the data, don't encode. 
        if self.export_data:
            return x_data, y_data

        if self.operation == 'circ_rlwe':
            prefix = []
            x = [prefix + self.input_encoder.encode(el) for el in x_data]
            y = [self.output_encoder.write_int(el) for el in y_data] 
        else:
            # encode input
            x = self.input_encoder.encode(x_data)
            # encode output
            y = self.output_encoder.encode(y_data)
        if self.max_len > 0 and (len(x) >= self.max_len or len(y) >= self.max_len):
            return None
        return x, y

    def decode_class(self, i):
        if self.input_encoder.balanced or len(self.generator.secrets)==1:
            return "1"
        else:
            return i 

    def code_class(self, xi, yi, int_len):
        return xi.shape[0] // int_len

    def check_prediction(self, src, tgt, hyp):
        if len(hyp) == 0 or len(tgt) == 0:
            return -1, [0 for _ in range(len(tgt))], self.generator.Q+1
        val_hyp = self.output_encoder.decode(hyp[::-1])
        if val_hyp is None:
            return -1, [0 for _ in range(len(tgt))], self.generator.Q+1
        val_tgt = self.output_encoder.decode(tgt[::-1])
        if len(val_hyp) != len(val_tgt):
            return -1, [0 for _ in range(len(tgt))], self.generator.Q+1
        val_src = self.input_encoder.decode(src)
        return self.generator.evaluate(val_src, val_tgt, val_hyp), self.generator.evaluate_bitwise(tgt, hyp), self.generator.get_difference(val_tgt, val_hyp)
        
    def create_train_iterator(self, task, data_path, params):
        """
        Create a dataset for this environment.
        """
        logger.info(f"Creating train iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=True,
            params=params,
            path=(None if data_path is None else data_path[task][0]),
        )
        return DataLoader(
            dataset,
            timeout=(0 if params.num_workers == 0 else 1800),
            batch_size=(max(int(params.batch_size // params.N)+1,1) if params.reload_data == "" else params.batch_size), # EJW add an extra element for generation so we can be sure to get the right batch size.
            num_workers=(
                params.num_workers
                if data_path is None or params.num_workers == 0
                else 1
            ),
            shuffle=False,
            collate_fn=(dataset.collate_fn if params.export_data is False else dataset.collate_fn_export),
        )

    def create_test_iterator(
        self, data_type, task, data_path, batch_size, params, size
    ):
        """
        Create a dataset for this environment.
        """
        assert data_type in ["valid", "test"]
        logger.info(f"Creating {data_type} iterator for {task} ...")

        dataset = EnvDataset(
            self,
            task,
            train=False,
            params=params,
            path=(
                None
                if data_path is None
                else data_path[task][1 if data_type == "valid" else 2]
            ),
            size=size,
            type=data_type,
        )
        return DataLoader(
            dataset,
            timeout=0,
            batch_size=(max(int(batch_size // params.N)+1,1) if params.reload_data == "" else batch_size),
            num_workers=1,
            shuffle=False,
            collate_fn=(dataset.collate_fn if params.export_data is False else dataset.collate_fn_export),
        )

    @staticmethod
    def register_args(parser):
        """
        Register environment parameters.
        """
        parser.add_argument(
            "--eval_size",
            type=int,
            default=1000,
            help="Size of valid and test samples",
        )
        parser.add_argument(
            "--operation", type=str, default="circ_rlwe", help="Operation performed"
        )
        parser.add_argument(
            "--generator", type=str, default="uniform", help="Random generation of coefficients"
        )
        parser.add_argument(
            "--N", type=int, default=30, help="dimension of matrix"
        )
        parser.add_argument(
            "--Q", type=int, default=251, help="modulo"
        )

        # Reuse samples
        parser.add_argument(
            "--reuse", type=bool_flag, default=True, help='reuse samples during training?'
        )
        parser.add_argument(
            "--num_reuse_samples", type=int, default=10000, help='number of samples to choose from during one reuse batch'
        )
        parser.add_argument(
            "--times_reused", type=int, default=10, help='how many times to reuse a sample before discarding it?'
        )
        parser.add_argument(
            "--K", type=int, default=1, help="if K > 1, will combine multiple reused samples"
        )

        # Restricted A values
        parser.add_argument(
            "--percQ_bound", type=float, default=1.0, help='what percent of Q is the max of A -- initial value if percQ_increase > 0 else the whole time value'
        )
        parser.add_argument(
            "--maxQ_prob", type=float, default=0, help='probability that we will let the maxQ=Q vs maxQ=percQ_bound*Q'
        )


        # Error
        parser.add_argument(
            "--error", type=bool_flag, default=True, help='add error to generation?'
        )
        parser.add_argument(
            "--sigma", type=float, default=3, help='sigma for gaussian error'
        )

        parser.add_argument(
            "--correctQ", type=bool_flag, default=False, help='flip the Q range to be within -Q/2 and Q/2?'
        )
    
        parser.add_argument(
            "--secret", type=str, default="71", help="secret"
        )
        parser.add_argument(
            "--secrettype", type=str, default="b", help="binary, gaussian, integer secret?"
        )
        parser.add_argument(
            "--sparsity", type=float, default=0.5, help="what's the sparsity of the binary secret"
        )
        parser.add_argument(
            "--density", type=float, default=0, help="density of secret"
        )
        parser.add_argument(
            "--hamming", type=int, default=3, help="if >0, will set exactly this many bits to 1"
        )


        # Bases
        parser.add_argument(
            "--balanced_base", type=bool_flag, default=False, help="use balanced base?"
        )
        parser.add_argument(
            "--input_int_base", type=int, default=81, help="base of the input encoder"
        )
        parser.add_argument(
            "--output_int_base", type=int, default=81, help="base of the output encoder"
        )
        parser.add_argument(
            "--max_output_len", type=int, default=512, help="max length of output, beam max size"
        )
        
        
        # representation parameters
        parser.add_argument(
            "--no_separator",
            type=bool_flag,
            default=True,
            help="No separator between numbers",
        )
        

class EnvDataset(Dataset):
    def __init__(self, env, task, train, params, path, size=None, type=None):
        super(EnvDataset).__init__()
        self.env = env
        self.train = train
        self.task = task
        self.batch_size = params.batch_size
        self.env_base_seed = params.env_base_seed
        self.path = path
        self.global_rank = params.global_rank
        self.count = 0
        self.type = type
        assert task in LatticeEnvironment.TRAINING_TASKS
        assert size is None or not self.train
        assert not params.batch_load or params.reload_size > 0

        # batching
        self.num_workers = params.num_workers
        self.batch_size = params.batch_size

        self.batch_load = params.batch_load
        self.reload_size = params.reload_size
        self.reload_data = False
        self.local_rank = params.local_rank
        self.n_gpu_per_node = params.n_gpu_per_node

        self.basepos = 0
        self.nextpos = 0
        self.seekpos = 0

        # generation, or reloading from file
        if path is not None:
            self.reload_data = True
            assert os.path.isfile(path)
            if params.batch_load and self.train:
                self.load_chunk()
            else:
                logger.info(f"Loading data from {path} ...")
                with io.open(path, mode="r", encoding="utf-8") as f:
                    # either reload the entire file, or the first N lines
                    # (for the training set)
                    if not train:
                        lines = [line.rstrip().split("|") for line in f]
                    else:
                        lines = []
                        for i, line in enumerate(f):
                            if i == params.reload_size:
                                break
                            if i % params.n_gpu_per_node == params.local_rank:
                                lines.append(line.rstrip()) #.split("|"))
                self.data = [xy.split("\t") for xy in lines]
                self.data = [xy for xy in self.data if len(xy) == 2]
                if params.error:
                    self.init_rng()
                    self.error = np.int64(self.env.rng.normal(0, params.sigma, size = len(self.data)).round())
                else:
                    self.error = None
                logger.info(f"Loaded {len(self.data)} equations from the disk.")

        # dataset size: infinite iterator for train, finite for valid / test
        # (default of 10000 if no file provided)
        if self.train:
            self.size = 1 << 60
        elif size is None:
            self.size = 10000 if path is None else len(self.data)
        else:
            assert size > 0
            self.size = size

    def load_chunk(self):
        self.basepos = self.nextpos
        logger.info(
            f"Loading data from {self.path} ... seekpos {self.seekpos}, "
            f"basepos {self.basepos}"
        )
        endfile = False
        with io.open(self.path, mode="r", encoding="utf-8") as f:
            f.seek(self.seekpos, 0)
            lines = []
            for i in range(self.reload_size):
                line = f.readline()
                if not line:
                    endfile = True
                    break
                if i % self.n_gpu_per_node == self.local_rank:
                    lines.append(line.rstrip().split("|"))
            self.seekpos = 0 if endfile else f.tell()

        self.data = [xy.split("\t") for _, xy in lines]
        self.data = [xy for xy in self.data if len(xy) == 2]
        self.nextpos = self.basepos + len(self.data)
        logger.info(
            f"Loaded {len(self.data)} equations from the disk. seekpos {self.seekpos}, "
            f"nextpos {self.nextpos}"
        )
        if len(self.data) == 0:
            self.load_chunk()

    def collate_fn_export(self, elements):
        x, y = zip(*elements)
        if self.env.operation == 'circ_rlwe':
            x_ar = []
            for el in x:
                for e in el:
                    x_ar.append(np.array(e) % self.env.generator.Q) # get rid of the negative numbers.
            y_ar = [np.array(el) for el in y]
            y = np.concatenate(y_ar)
            x = x_ar

        # Fix the length to be the actual batch size.
        x = x[:self.batch_size]
        y = y[:self.batch_size]

        x = [list(seq) for seq in x]
        y = list(y)
        return x,y
        

    def collate_fn(self, elements):
        """
        Collate samples into a batch.
        """
        # Reset the matrix counter
        self.env.mat_count = 0 

        x, y = zip(*elements)
        if self.env.operation == 'circ_rlwe':
            x_ar = []
            y_ar = []
            for e1, e2 in zip(x,y):
                if self.reload_data: # line by line
                    x_ar.append(np.array(e1))
                else:
                    for ee1, ee2 in zip(e1,e2): # matrix by matrix
                        x_ar.append(np.array(ee1))
                        y_ar.append(np.array(ee2))
            y = y_ar
            x = x_ar

        # Fix the length to be the actual batch size.
        x = x[:self.batch_size]
        y = y[:self.batch_size]

        # Distinguish the different equations
        int_len =  len(self.env.input_encoder.write_int(0))
        sep = 0 if self.env.input_encoder.no_separator else 1
        nb_eqs = [self.env.code_class(xi, yi, int_len+sep) for xi, yi in zip(x, y)]

        # Pad 
        x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        x, x_len = self.env.batch_sequences(x)
        y, y_len = self.env.batch_sequences(y)
        return (x, x_len), (y, y_len), torch.LongTensor(nb_eqs)

    def init_rng(self):
        """
        Initialize random generator for training.
        """
        if hasattr(self.env, "rng"):
            return
        if self.train:
            worker_id = self.get_worker_id()
            self.env.worker_id = worker_id
            self.env.rng = np.random.RandomState(
                [worker_id, self.global_rank, self.env_base_seed]
            )
            logger.info(
                f"Initialized random generator for worker {worker_id}, with seed "
                f"{[worker_id, self.global_rank, self.env_base_seed]} "
                f"(base seed={self.env_base_seed})."
            )
        else:
            self.env.rng = np.random.RandomState(None if self.type == "valid" else 0)

    def get_worker_id(self):
        """
        Get worker ID.
        """
        if not self.train:
            return 0
        worker_info = torch.utils.data.get_worker_info()
        assert (worker_info is None) == (self.num_workers == 0)
        return 0 if worker_info is None else worker_info.id

    def __len__(self):
        """
        Return dataset size.
        """
        return self.size

    def __getitem__(self, index):
        """
        Return a training sample.
        Either generate it, or read it from file.
        """
        self.init_rng()
        if self.path is None:
            return self.generate_sample()
        else:
            return self.read_sample(index)

    def read_sample(self, index):
        """
        Read a sample.
        """
        print('loading chunk here')
        idx = index
        if self.train:
            if self.batch_load:
                if index >= self.nextpos:
                    self.load_chunk()
                idx = index - self.basepos
            else:
                index = self.env.rng.randint(len(self.data))
                idx = index
        x, y = self.data[idx]
        x = np.array(x.split()).astype(int)
        y = np.array(y.split()[0]).astype(int)

        if self.task == 'lattice':
            pass
        else:
            raise Exception(f"Must use {self.task} when reloading data")

        x = self.env.input_encoder.encode(x)
        y = self.env.output_encoder.write_int(y)
        assert len(x) >= 1 and len(y) >= 1
        return x, y

    def generate_sample(self):
        """
        Generate a sample.
        """
        while True:
            try:
                if self.task == "lattice":
                    xy = self.env.gen_expr(self.type)
                else:
                    raise Exception(f"Unknown data type: {self.task}")
                if xy is None:
                    continue
                x, y = xy
                break
            except Exception as e:
                logger.error(
                    'An unknown exception of type {0} occurred for worker {4} in line {1} for expression "{2}". Arguments:{3!r}.'.format(
                        type(e).__name__,
                        sys.exc_info()[-1].tb_lineno,
                        "F",
                        e.args,
                        self.get_worker_id(),
                    )
                )
                continue
        self.count += 1
        self.env.mat_count += 1

        return x, y
