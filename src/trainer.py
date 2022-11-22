# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import io
import sys
import ast
import time
from logging import getLogger
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from .optim import get_optimizer
from .utils import to_cuda

# if torch.cuda.is_available():
has_apex = True
try:
    import apex
except:
    has_apex - False


logger = getLogger()


class Trainer(object):
    def __init__(self, modules, env, params):
        """
        Initialize trainer.
        """
        # if torch.cuda.is_available() and params.nvidia_apex is True:
        #     import apex

        # modules / params
        self.modules = modules
        self.params = params
        self.env = env

        # epoch / iteration size
        self.epoch_size = params.epoch_size
        self.total_samples = 0
        if self.epoch_size == -1:
            self.epoch_size = self.data
            assert self.epoch_size > 0

        # data iterators
        self.iterators = {}

        # set parameters
        self.set_parameters()

        # float16 / distributed (no AMP)
        assert params.amp >= 1 or not params.fp16
        assert params.amp >= 0 or params.accumulate_gradients == 1
        # assert not params.multi_gpu or params.amp == -1 or params.nvidia_apex
        assert not params.nvidia_apex or has_apex
        if params.multi_gpu: # and params.amp == -1:
            logger.info("Using nn.parallel.DistributedDataParallel ...")
            for k in self.modules.keys():
                self.modules[k] = nn.parallel.DistributedDataParallel(
                    self.modules[k],
                    device_ids=[params.local_rank],
                    output_device=params.local_rank,
                    broadcast_buffers=True,
                )

        # set optimizer
        self.set_optimizer()

        # float16 / distributed (AMP)
        self.scaler = None
        if params.amp >= 0:
            self.init_amp()
        #    if params.multi_gpu:
        #        logger.info("Using apex.parallel.DistributedDataParallel ...")
        #        for k in self.modules.keys():
        #            self.modules[k] = apex.parallel.DistributedDataParallel(
        #                self.modules[k], delay_allreduce=True
        #            )

        # stopping criterion used for early stopping
        if params.stopping_criterion != "":
            split = params.stopping_criterion.split(",")
            assert (len(split) == 2 and split[1].isdigit()) or (len(split)==3 and split[1].isdigit() and split[2].isdigit())
            self.decrease_counts_max = int(split[1]) if (len(split) == 2) else int(split[2])
            self.stopping_level = int(split[1]) if len(split)==3 else -1
            if self.stopping_level > 0 and self.stopping_level < 1:
                # stopping level is expressed as decimal rather than percent
                self.stopping_level = self.stopping_level * 100
                logger.info(f'Converted stopping criterion {self.stopping_level / 100} into a percent {self.stopping_level}%')
            self.decrease_counts = 0
            if split[0][0] == "_":
                self.stopping_criterion = (split[0][1:], False)
            else:
                self.stopping_criterion = (split[0], True)
            self.best_stopping_criterion = -1e12 if self.stopping_criterion[1] else 1e12
        else:
            self.stopping_criterion = None
            self.best_stopping_criterion = None

        # Secret matching criterion -- stop if True
        self.secret_match = np.array([False]) # for _ in range(self.env.num_secrets_per_size)])
        self.secret_stop = params.secret_stop

        # validation metrics
        self.metrics = []
        metrics = [m for m in params.validation_metrics.split(",") if m != ""]
        for m in metrics:
            m = (m[1:], False) if m[0] == "_" else (m, True)
            self.metrics.append(m)
        self.best_metrics = {
            metric: (-1e12 if biggest else 1e12) for (metric, biggest) in self.metrics
        }
        self.best_10percQ = 0

        # training statistics
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.stats = OrderedDict(
            [("processed_e", 0)]
            + [("processed_w", 0)]
            + sum(
                [[(x, []), (f"{x}-AVG-STOP-PROBS", [])] for x in env.TRAINING_TASKS], []
            )
        )
        self.last_time = time.time()

        # reload potential checkpoints
        self.reload_checkpoint()

        # file handler to export data
        if params.export_data:
            assert params.reload_data == ""
            params.export_path_prefix = []
            self.file_handler_prefix = []

            # Make separate files for train/test/valid data.
            for target in ['train', 'test', 'valid']:
                export_path_prefix = os.path.join(params.dump_path, f"{target}_data.prefix")
                file_handler_prefix = io.open(
                    export_path_prefix, mode="a", encoding="utf-8"
                )
                logger.info(
                    f"Data will be stored in prefix in: {export_path_prefix} ..."
                )
                params.export_path_prefix.append(export_path_prefix)
                self.file_handler_prefix.append(file_handler_prefix)
                
        # reload exported data
        if params.reload_data != "":
            assert params.num_workers in [0, 1]
            assert params.export_data is False
            if params.reload_data == 'lattice_default':
                base_path = f'/private/home/ewenger/lattice_ml/transformer_code/data/N={params.N}_Q={params.Q}/'
                try:
                    assert os.path.exists(base_path) # make sure you've generated the code. 
                except AssertionError:
                    logger.info('Pregenerated data not found at this path -- please generate the data and try again.')
                train_path, valid_path, test_path = base_path + 'train_data.prefix', base_path + 'valid_data.prefix', base_path + 'test_data.prefix'
                self.data_path = {'lattice': (train_path, valid_path, test_path)}  
            else:
                s = [x.split(",") for x in params.reload_data.split(";") if len(x) > 0]
                assert (
                    len(s) >= 1
                    and all(len(x) == 4 for x in s)
                    and len(s) == len(set([x[0] for x in s]))
                ) 
                self.data_path = {
                    task: (train_path, valid_path, test_path)
                    for task, train_path, valid_path, test_path in s
                }

            # Make sure all the paths exist. 
            assert all(
                all(os.path.isfile(path) for path in paths)
                for paths in self.data_path.values()
            )
            for task in self.env.TRAINING_TASKS:
                assert (task in self.data_path) == (task in params.tasks)
        else:
            self.data_path = None

        # create data loaders
        if not params.eval_only:
            if params.env_base_seed < 0:
                params.env_base_seed = np.random.randint(1_000_000_000)
            self.dataloader = {
                task: iter(self.env.create_train_iterator(task, self.data_path, params))
                for task in params.tasks
            }

    def set_parameters(self):
        """
        Set parameters.
        """
        self.parameters = {}
        named_params = []
        for v in self.modules.values():
            named_params.extend(
                [(k, p) for k, p in v.named_parameters() if p.requires_grad]
            )
        self.parameters["model"] = [p for k, p in named_params]
        for k, v in self.parameters.items():
            logger.info("Found %i parameters in %s." % (len(v), k))
            assert len(v) >= 1

    def set_optimizer(self):
        """
        Set optimizer.
        """
        params = self.params
        self.optimizer = get_optimizer(
            self.parameters["model"], params.optimizer
        )
        logger.info("Optimizer: %s" % type(self.optimizer))

    def init_amp(self):
        """
        Initialize AMP optimizer.
        """
        params = self.params
        assert (
            params.amp == 0
            and params.fp16 is False
            or params.amp in [1, 2, 3]
            and params.fp16 is True
        )
        mod_names = sorted(self.modules.keys())
        if params.nvidia_apex is True:
            modules, optimizer = apex.amp.initialize(
                [self.modules[k] for k in mod_names],
                self.optimizer,
                opt_level=("O%i" % params.amp),
            )
            self.modules = {k: module for k, module in zip(mod_names, modules)}
            self.optimizer = optimizer
        else:
            self.scaler = torch.cuda.amp.GradScaler()

    def optimize(self, loss):
        """
        Optimize.
        """
        # check NaN
        if (loss != loss).data.any():
            logger.warning("NaN detected")
            # exit()

        params = self.params

        # optimizer
        optimizer = self.optimizer

        # regular optimization
        if params.amp == -1:
            optimizer.zero_grad()
            loss.backward()
            if params.clip_grad_norm > 0:
                clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
            optimizer.step()

        # AMP optimization
        elif params.nvidia_apex is True:
            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if params.clip_grad_norm > 0:
                    clip_grad_norm_(apex.amp.master_params(self.optimizer), params.clip_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            else:
                with apex.amp.scale_loss(loss, optimizer, delay_unscale=True) as scaled_loss:
                    scaled_loss.backward()

        else:
            if params.accumulate_gradients > 1:
                loss = loss / params.accumulate_gradients
            self.scaler.scale(loss).backward()

            if (self.n_iter + 1) % params.accumulate_gradients == 0:
                if params.clip_grad_norm > 0:
                    self.scaler.unscale_(optimizer)
                    clip_grad_norm_(self.parameters["model"], params.clip_grad_norm)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

    def iter(self):
        """
        End of iteration.
        """
        self.n_iter += 1
        self.n_total_iter += 1
        self.print_stats()

    def print_stats(self):
        """
        Print statistics about the training.
        """
        if self.n_total_iter % 200 != 0:
            return

        s_iter = "%7i - " % self.n_total_iter
        s_stat = " || ".join(
            [
                "{}: {:7.4f}".format(k.upper().replace("_", "-"), np.mean(v))
                for k, v in self.stats.items()
                if type(v) is list and len(v) > 0
            ]
        )
        for k in self.stats.keys():
            if type(self.stats[k]) is list:
                del self.stats[k][:]

        # learning rates
        s_lr = (
            (" - LR: ")
            + " / ".join("{:.4e}".format(group["lr"]) for group in self.optimizer.param_groups)
        )

        # processing speed
        new_time = time.time()
        diff = new_time - self.last_time
        s_speed = "{:7.2f} equations/s - {:8.2f} words/s - ".format(
            self.stats["processed_e"] * 1.0 / diff,
            self.stats["processed_w"] * 1.0 / diff,
        )
        self.stats["processed_e"] = 0
        self.stats["processed_w"] = 0
        self.last_time = new_time

        # log speed + stats + learning rate
        logger.info(s_iter + s_speed + s_stat + s_lr)

    def save_checkpoint(self, name, include_optimizer=True):
        """
        Save the model / checkpoints.
        """
        if not self.params.is_master:
            return

        path = os.path.join(self.params.dump_path, "%s.pth" % name)
        logger.info("Saving %s to %s ..." % (name, path))

        data = {
            "epoch": self.epoch,
            "n_total_iter": self.n_total_iter,
            "best_metrics": self.best_metrics,
            "best_stopping_criterion": self.best_stopping_criterion,
            "params": {k: v for k, v in self.params.__dict__.items()},
        }

        for k, v in self.modules.items():
            logger.warning(f"Saving {k} parameters ...")
            data[k] = v.state_dict()

        if include_optimizer:
            logger.warning("Saving optimizer ...")
            data["optimizer"] = self.optimizer.state_dict()
            if self.scaler is not None:
                data["scaler"] = self.scaler.state_dict()

        torch.save(data, path)

    def reload_checkpoint(self):
        """
        Reload a checkpoint if we find one.
        """
        checkpoint_path = os.path.join(self.params.dump_path, "checkpoint.pth")
        if not os.path.isfile(checkpoint_path):
            if self.params.reload_checkpoint == "":
                return
            else:
                checkpoint_path = self.params.reload_checkpoint
                assert os.path.isfile(checkpoint_path)
        
        logger.warning(f"Reloading checkpoint from {checkpoint_path} ...")
        data = torch.load(checkpoint_path, map_location="cpu")

        # reload model parameters
        for k, v in self.modules.items():
            assert k in data
            try:
                v.load_state_dict(data[k])
            except:
                print('removing module prefix')
                if all([k2.startswith("module.") for k2 in data[k].keys()]):
                    data[k] = {
                        k2[len("module.") :]: v2 for k2, v2 in data[k].items()
                    }
                v.load_state_dict(data[k])


        # reload optimizer
        # AMP checkpoint reloading is buggy, we cannot reload optimizer
        # instead, we only reload current iterations / learning rates
        if self.params.amp == -1 or not self.params.nvidia_apex:
            logger.warning("Reloading checkpoint optimizer ...")
            self.optimizer.load_state_dict(data["optimizer"])
        else:
            logger.warning("Not reloading checkpoint optimizer.")
            for group_id, param_group in enumerate(
                self.optimizer.param_groups
            ):
                if "num_updates" not in param_group:
                    logger.warning("No 'num_updates' for optimizer.")
                    continue
                logger.warning(
                    "Reloading 'num_updates' and 'lr' for optimizer."
                )
                param_group["num_updates"] = data["optimizer"][
                    "param_groups"
                ][group_id]["num_updates"]
                param_group["lr"] = self.optimizer.get_lr_for_step(
                    param_group["num_updates"]
                )

        if self.params.fp16 and not self.params.nvidia_apex:
            logger.warning("Reloading gradient scaler ...")
            self.scaler.load_state_dict(data["scaler"])
        else:
            print(self.scaler is None, "scaler" in data)
            assert (self.scaler is None) and ("scaler" not in data)

        # reload main metrics
        self.epoch = data["epoch"] + 1
        self.n_total_iter = data["n_total_iter"]
        self.best_metrics = data["best_metrics"]
        self.best_stopping_criterion = data["best_stopping_criterion"]
        logger.warning(
            f"Checkpoint reloaded. Resuming at epoch {self.epoch} / iteration {self.n_total_iter} ..."
        )

    def save_periodic(self):
        """
        Save the models periodically.
        """
        if not self.params.is_master:
            return
        if (
            self.params.save_periodic > 0
            and self.epoch % self.params.save_periodic == 0
        ):
            self.save_checkpoint("periodic-%i" % self.epoch)

    def save_best_model(self, scores):
        """
        Save best models according to given validation metrics.
        """
        if not self.params.is_master:
            return
        for metric, biggest in self.metrics:
            if metric not in scores:
                logger.warning('Metric "%s" not found in scores!' % metric)
                continue
            factor = 1 if biggest else -1
            if factor * scores[metric] > factor * self.best_metrics[metric]:
                self.best_metrics[metric] = scores[metric]
                logger.info("New best score for %s: %.6f" % (metric, scores[metric]))
                self.save_checkpoint("best-%s" % metric)

        # Manually adding percs diff model saving. 
        currPercQ = float(ast.literal_eval(scores["valid_lattice_percs_diff"])[0])
        if currPercQ > self.best_10percQ:
            self.best_10percQ = currPercQ
            logger.info("New best score for toleranc=10percQ acc: %.6f" % (currPercQ))
            self.save_checkpoint("best-10percQ")
        

    def end_epoch(self, scores):
        """
        End the epoch AND update curriculum info if using. 
        """
        # Check to see if secret_match is True -- this means the model has learned the secret based on eval tests.
        if np.all(self.secret_match) and self.secret_stop: # and (not self.env.hamming_curriculum or (self.env.curr_hamming == self.env.max_hamming)):
            logger.info('Found secret match - ending experiment.')
            self.save_checkpoint("checkpoint")
            if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                os.system("scancel " + os.environ["SLURM_JOB_ID"])
            exit()

        # stop if the stopping criterion has not improved after a certain number of epochs
        if self.stopping_criterion is not None and (
            self.params.is_master or not self.stopping_criterion[0].endswith("_mt_bleu")
        ) and ('bitwise'  not in self.stopping_criterion[0]):
            metric, biggest = self.stopping_criterion
            assert metric in scores, metric       

            factor = 1 if biggest else -1

            if factor * scores[metric] > factor * self.best_stopping_criterion:
                self.best_stopping_criterion = scores[metric]
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
            # EJW: check to see if your accuracy criterion exceeded the stopping level
            if (self.stopping_level > 0) and (self.best_stopping_criterion*factor >= self.stopping_level*factor):
                logger.info(
                    "Exceeded preset max stopping criterion value. Ended the experiment."
                )     
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()

        elif (self.stopping_criterion is not None) and ('bitwise' in self.stopping_criterion[0]):
            # there are 3 options available 
            assert self.stopping_criterion[0] in ['bitwise_mean', 'bitwise_top', 'bitwise_bottom']

            bw = ast.literal_eval(scores['valid_lattice_bitwise_acc']) # TODO what if it isn't? 

            if self.stopping_criterion[0] == 'bitwise_mean':
                score = np.mean(bw)
            elif self.stopping_criterion[0] == 'bitwise_top':
                score = np.mean(bw[:int(len(bw)//2)])
            elif self.stopping_criterion[0] == 'bitwise_bottom':
                score = np.mean(bw[int(len(bw)//2):])

            if  score > self.best_stopping_criterion:
                self.best_stopping_criterion = score
                logger.info(
                    "New best validation score: %f" % self.best_stopping_criterion
                )
                self.decrease_counts = 0
            else:
                logger.info(
                    "Not a better validation score (%i / %i)."
                    % (self.decrease_counts, self.decrease_counts_max)
                )
                self.decrease_counts += 1
            if self.decrease_counts > self.decrease_counts_max:
                logger.info(
                    "Stopping criterion has been below its best value for more "
                    "than %i epochs. Ending the experiment..."
                    % self.decrease_counts_max
                )
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
            # EJW: check to see if your accuracy criterion exceeded the stopping level
            if (self.stopping_level > 0) and (self.best_stopping_criterion >= self.stopping_level):
                logger.info(
                    "Exceeded preset max accuracy value. Ended the experiment."
                )     
                if self.params.multi_gpu and "SLURM_JOB_ID" in os.environ:
                    os.system("scancel " + os.environ["SLURM_JOB_ID"])
                exit()
        self.save_checkpoint("checkpoint")
        self.epoch += 1

    def get_batch(self, task):
        """
        Return a training batch for a specific task.
        """
        try:
            batch = next(self.dataloader[task])
        except Exception as e:
            logger.error(
                "An unknown exception of type {0} occurred in line {1} when fetching batch. "
                "Arguments:{2!r}. Restarting ...".format(
                    type(e).__name__, sys.exc_info()[-1].tb_lineno, e.args
                )
            )
            if self.params.is_slurm_job:
                if int(os.environ["SLURM_PROCID"]) == 0:
                    logger.warning("Requeuing job " + os.environ["SLURM_JOB_ID"])
                    os.system("scontrol requeue " + os.environ["SLURM_JOB_ID"])
                else:
                    logger.warning("Not the master process, no need to requeue.")
            raise

        return batch

    def export_data(self, task):
        """
        Export data to the disk.
        """
        env = self.env
        for fh in self.file_handler_prefix:
            #(x1, len1), (x2, len2), _ = self.get_batch(task)
            x1, x2 = self.get_batch(task)
            for i in range(len(x1)):
                prefix1 = [str(wid) for wid in x1[i]]
                prefix2 = [str(x2[i])]
                # save
                prefix1_str = " ".join(prefix1)
                prefix2_str = " ".join(prefix2)
                fh.write(f"{prefix1_str}\t{prefix2_str}\n")
                fh.flush()
          
        # number of processed sequences / words
        self.n_equations += self.params.batch_size
        self.total_samples += self.params.batch_size 
        self.stats["processed_e"] += len(x1)
        self.stats["processed_w"] += (len(x1)*len(x1[0]
        ) + len(x2) - 2)#.sum().item()

    def enc_dec_step(self, task):
        """
        Encoding / decoding step.
        """
        params = self.params
        encoder, decoder = self.modules["encoder"], self.modules["decoder"]
        encoder.train()
        decoder.train()

        # batch
        (x1, len1), (x2, len2), _ = self.get_batch(task)

        # target words to predict
        alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
        pred_mask = (
            alen[:, None] < len2[None] - 1
        )  # do not predict anything given the last target word
        y = x2[1:].masked_select(pred_mask[:-1])
        assert len(y) == (len2 - 1).sum().item()

        # cuda
        x1, len1, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

        # forward / loss
        if params.amp == -1 or params.nvidia_apex:
            encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1,
            )
            _, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=False, weighted= self.params.weighted_loss,  # EJW -- set as param.
            )
        else:
            with torch.cuda.amp.autocast():
                encoded = encoder("fwd", x=x1, lengths=len1, causal=False)
                decoded = decoder(
                    "fwd",
                    x=x2,
                    lengths=len2,
                    causal=True,
                    src_enc=encoded.transpose(0, 1),
                    src_len=len1,
                )
                _, loss = decoder(
                    "predict",
                    tensor=decoded,
                    pred_mask=pred_mask,
                    y=y,
                    get_scores=False,
                    weighted= self.params.weighted_loss, # default should be FALSE
                )
        
        self.stats[task].append(loss.item())

        # optimize
        self.optimize(loss)

        # number of processed sequences / words
        self.n_equations += params.batch_size
        self.stats["processed_e"] += len1.size(0)
        self.stats["processed_w"] += (len1 + len2 - 2).sum().item()
