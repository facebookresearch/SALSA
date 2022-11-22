# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from logging import getLogger
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
import os
import torch
import numpy as np
from scipy import stats
import time
import ast

from .utils import to_cuda



logger = getLogger()


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
        m, bw, diff = env.check_prediction(src, tgt, hyp)
    except Exception as e:
        print('exception', e)
        m, bw, diff = -1, [0 for _ in range(len(tgt))], env.generator.Q+1
    eq["is_valid"] = m
    eq['bitwise'] = bw
    eq['difference'] = abs(diff)
    return eq


class Evaluator(object):

    ENV = None

    def __init__(self, trainer):
        """
        Initialize evaluator.
        """
        self.trainer = trainer
        self.modules = trainer.modules
        self.params = trainer.params
        self.env = trainer.env
        self.softmax = torch.nn.Softmax(dim=0)
        self.rng = np.random.RandomState(int(time.time()))
        Evaluator.ENV = trainer.env

    def run_all_evals(self):
        """
        Run all evaluations.

        """
        params = self.params
        scores = OrderedDict({"epoch": self.trainer.epoch})

        # save statistics about generated data
        if params.export_data:
            scores["total"] = self.trainer.total_samples
            return scores

        with torch.no_grad():
            for data_type in ["valid"]:
                for task in params.tasks:
                    if params.beam_eval:
                        self.enc_dec_step_beam(data_type, task, scores)
                    else:
                        self.enc_dec_step(data_type, task, scores)
        return scores

    def encode_data(self, sec_idx, A, B):
        ''' Encodes data in format expected by model. ''' 
        if len(self.env.generator.secrets) > 1:
            idx = sec_idx % self.env.generator.num_secrets_per_size
            prefix = [idx, '|']
        else:
            prefix = []

        x = np.array([prefix + self.env.input_encoder.encode(el) for el in A])
        y = np.array([self.env.output_encoder.write_int(el) for el in B], dtype=object)

        # Make them tensors
        int_len =  len(self.env.input_encoder.write_int(0))
        sep = 0 if self.env.input_encoder.no_separator else 1
        nb_eqs = [self.env.code_class(xi, yi, int_len+sep) for xi, yi in zip(x, y)]
        x = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in x]
        y = [torch.LongTensor([self.env.word2id[w] for w in seq]) for seq in y]
        return x,y,nb_eqs

    def batch_data(self, A, B, nb_eqs):
        ''' Batches data in preparation for inference. '''
        x1, x_len = self.env.batch_sequences(A)
        y1, y_len = self.env.batch_sequences(B)
        x1, len1, x2, len2, _ = x1, x_len, y1, y_len, nb_eqs
        alen = torch.arange(y_len.max(), dtype=torch.long, device=len2.device)
        pred_mask = (alen[:, None] < len2[None] - 1)  # do not predict anything given the last target word
        _y = x2[1:].masked_select(pred_mask[:-1])
        return x1, len1, x2, len2, _y

    def run_beam_generation(self, x1, x1_, x2, len1, len1_, len2, encoder, decoder):
        """ Master function to run this rather than having it in like 5 different places. """
        # Run beam generation to get output. 
        encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
        _, _, generations= decoder.generate_beam(encoded.transpose(0, 1), len1_,
                                                    beam_size=self.params.beam_size,
                                                    length_penalty=self.params.beam_length_penalty,
                                                    early_stopping=self.params.beam_early_stopping,
                                                    max_len=4096)
        # Analyze beam output
        inputs = []
        for i in range(len(generations)):
            for j, (score, hyp) in enumerate(
                    sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)
            ):
                inputs.append({ "i": i, "j": j, "score": score,
                                "src": x1[1 : len1[i] - 1, i].tolist(),
                                "tgt": x2[1 : len2[i] - 1, i].tolist(),
                                "hyp": hyp[1:].tolist(),
                                "task": 'lattice'})       
                
        bs = len(len1)
        outputs = []
        if self.params.windows is True:
            for inp in inputs:
                outputs.append(check_hypothesis(inp))
        else:
            with ProcessPoolExecutor(max_workers=20) as executor:
                for output in executor.map(check_hypothesis, inputs, chunksize=1):
                    outputs.append(output)

        # Get beam outputs
        beam_log = {}
        for i in range(len(len1)):
            sr = idx_to_infix(self.env, x1[1 : len1[i] - 1, i].tolist(), True)
            tgt = idx_to_infix(self.env, x2[1 : len2[i] - 1, i].tolist(), False)
            beam_log[i] = {"src": sr, "tgt": tgt, "hyps": [(tgt, None, True)]}

        for i in range(bs):
            # select hypotheses associated to current equation
            gens = sorted([o for o in outputs if o["i"] == i], key=lambda x: x["j"])
            if len(gens) == 0:
                continue
            # source / target
            sr = gens[0]["src"]
            tgt = gens[0]["tgt"]
            beam_log[i] = {"src": sr, "tgt": tgt, "hyps": []}

            logger.info(f"hyp={gens[0]['hyp']}\ntgt={tgt}")
            #assert False == True, 'done'

            # for each hypothesis
            for j, gen in enumerate(gens):
                # sanity check
                assert (
                    gen["src"] == sr
                    and gen["tgt"] == tgt
                    and gen["i"] == i
                    and gen["j"] == j
                )
                is_valid = gen["is_valid"]
                is_b_valid = is_valid >= 0.0 and is_valid == 1
                # update beam log
                beam_log[i]["hyps"].append((gen["hyp"], gen["score"], is_b_valid))
        return beam_log

    def eval_secret(self, sec_idx, K, encoder, decoder):
        ''' Function to do special secret guessing. '''
    
        # do the secret value
        N = len(self.env.generator.secrets[sec_idx]) # sec_idx will always be 0. 
        specialA = np.identity(N, dtype=np.int64) * K
        specialB = np.inner(specialA, self.env.generator.secrets[sec_idx])

        # Get the encoded version of the data
        x, y, nb_eqs = self.encode_data(sec_idx, specialA, specialB)
       
        # Predict output
        pred_final = []

        # Iterate through via batch size because otherwise you can overwhelm
        for k in range(0, len(x), self.params.batch_size):
            x1, len1, x2, len2, _y = self.batch_data(x[k:k+self.params.batch_size], 
                                                            y[k:k+self.params.batch_size], 
                                                            nb_eqs[k:k+self.params.batch_size])
            x1_, len1_, x2, len2, y_ = to_cuda(x1, len1, x2, len2, _y)
            beam_log = self.run_beam_generation(x1, x1_, x2, len1, len1_, len2, encoder, decoder)

            for b in beam_log:
                try:
                    pred_final.append(self.env.output_encoder.decode(beam_log[b]['hyps'][0][0][::-1])[0]) 
                except Exception as e:
                    pred_final.append(-1)
        try:
            pred_softmax = self.softmax(torch.Tensor(pred_final)).detach().cpu().numpy()
        except:
            logger.info('Error in softmax prediction, secret decoding failed.')
            return

        # Inversion vector
        invert = np.vectorize(lambda x: 1 - x)

        # 3 methods of testing for matching: mean, mode, and softmax mean
        pred_bin1 = np.vectorize(lambda x: 0 if x > np.mean(pred_final) else 1)(pred_final) 
        pred_bin2 = np.vectorize(lambda x: 0 if x != stats.mode(pred_final)[0][0] else 1)(pred_final)
        pred_bin3 = np.vectorize(lambda x: 0 if x > np.mean(pred_softmax) else 1)(pred_softmax)

        bin1_match = sum((pred_bin1 == self.env.generator.secrets[sec_idx]).astype(int))
        bin2_match = sum((pred_bin2 == self.env.generator.secrets[sec_idx]).astype(int))
        bin3_match = sum((pred_bin3 == self.env.generator.secrets[sec_idx]).astype(int))
        
        # Match list
        match_counts = np.array([bin1_match, self.params.N-bin1_match, bin2_match, self.params.N-bin2_match, bin3_match, self.params.N-bin3_match])
        match_vecs = np.array([pred_bin1, invert(pred_bin1), pred_bin2, invert(pred_bin2), pred_bin3, invert(pred_bin3)])

        # Report results. 
        if np.any(match_counts == self.params.N):
            match_idx = np.argwhere(match_counts == self.params.N)[0][0] # just get the first ouptut
            logger.info(f'All bits in secret {sec_idx} have been recovered!  K={K}')
            logger.info(f'Predicted secret {sec_idx}: {str(list(match_vecs[match_idx]))}, real secret {sec_idx}: {str(list(self.env.generator.secrets[sec_idx]))}')
            self.trainer.secret_match[sec_idx] = True
        else:
            argmax = np.argmax(match_counts)
            logger.info(f'Secret matching: {match_counts[argmax]}/{self.params.N} bits matched for secret {sec_idx}, K={K}')
            logger.info(f'Best secret guess for {sec_idx}: {str(list(match_vecs[argmax]))}')
            if self.trainer.secret_match[sec_idx] != True:
                self.trainer.secret_match[sec_idx] = False
            
    ##### CODE TO RUN THE DISTINGUISHER
    def get_distinguisher_samples(self, i, g, num_samples, random, sec_idx):
        if not random: 
            A_s = []
            A_s_orig = []
            B_s = []
            while len(A_s) < num_samples:
                A, B = self.env.generator.generate(self.rng, sec_idx, self.params.N)
                c = np.random.randint(0, self.params.Q, size=self.params.N)
                A_orig = A.copy() # so we can compare performance against model prediction on ORIGINAL A, rather than b. 
                A[:,i] += c # add C to the ith coordinate of A
                updatedA = A % self.params.Q
                updated_A_orig = A_orig % self.params.Q   # EJW 11/4/22
                updatedB = (B + g*c) % self.params.Q
                A_s.extend([list(a) for a in updatedA])
                A_s_orig.extend([list(a) for a in updated_A_orig])
                B_s.extend(list(updatedB))
        else:
            A_s = []
            B_s = []
            while len(A_s) < num_samples:
                A, B = np.random.randint(0, self.params.Q, size=(self.params.N, self.params.N)), np.random.randint(0, self.params.Q, size=self.params.N)
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
        x, y, nb_eqs = self.encode_data(sec_idx, updatedA, updatedB)
        if not random:
            x_orig, y_orig, nb_eqs_orig = self.encode_data(sec_idx, updatedA_orig, updatedB)
        else:
            x_orig, y_orig, nb_eqs_orig = None, None, None
        return (x,y, nb_eqs), (x_orig, y_orig, nb_eqs_orig)

    def predict_distinguisher_outputs(self, data, encoder, decoder):
        ''' Runs beam generation + decoding. '''
        x,y,nb_eqs= data
        preds = []
        bs = []
        # Generate predictions 
        bsize = 32
        for k in range(0, len(x), bsize):
            x1, len1, x2, len2, _y = self.batch_data(x[k:k+bsize], y[k:k+bsize], nb_eqs[k:k+bsize])
            x1_, len1_, x2, len2, y_ = to_cuda(x1, len1, x2, len2, _y)
            beam_log = self.run_beam_generation(x1, x1_, x2, len1, len1_, len2, encoder, decoder)

            # Decode output
            rl = []
            bs_ = []
            for b in beam_log:
                try:
                    rl.append(self.env.output_encoder.decode(beam_log[b]['hyps'][0][0][::-1])[0]) 
                except Exception as e:
                    rl.append(-1)
                try:
                    bs_.append(self.env.output_encoder.decode(beam_log[b]['tgt'][::-1])[0])
                except:
                    bs_.append(-1)
            preds.extend(rl)
            bs.extend(bs_)
        return np.array(preds), np.array(bs)
  
    def run_distinguisher(self, sec_idx, scores, encoder, decoder):
        ''' Code to run the actual distinguisher for secret recovery. ''' 
        # Get parameters. 
        curr10PercQ = float(ast.literal_eval(scores["valid_lattice_percs_diff"])[0])/100
        tolerance = 0.1 # Fixed tolerance at 10% of Q. 
        bound = tolerance * self.params.Q
        advantage = (curr10PercQ-2*tolerance) # This is current advantage of model as distinguisher. 
        num_samples = max(int(2 // ((advantage ** 2))),50) # Max ever is 400 when curr10PercQ=30. 
        # Now run it 
        secret = np.ones(self.params.N)
        guess = 0 # fix secret bit guess at 0
        for i in range(self.params.N):
            # Get the data. 
            real_data, orig_lwe_data = self.get_distinguisher_samples(i, guess, num_samples, False, sec_idx)
            unif_data, _ = self.get_distinguisher_samples(i, guess, num_samples, True, sec_idx)
        
            # Run through the model. 
            lwe_pred, _ = self.predict_distinguisher_outputs(real_data, encoder, decoder)
            unif_pred, unif_real = self.predict_distinguisher_outputs(unif_data, encoder, decoder)
            lwe_orig_pred, _ = self.predict_distinguisher_outputs(orig_lwe_data, encoder, decoder)

            # Get the differences. 
            #lwe_diff = abs(lwe_pred - lwe_real) # EJW 11/5/22
            lwe_diff = abs(lwe_orig_pred - lwe_pred)
            unif_diff = abs(unif_pred - unif_real)
        
            # Count the number less than bound. 
            lwe_count = np.sum((lwe_diff < bound).astype(int))
            unif_count = np.sum((unif_diff < bound).astype(int))

            # TODO better method for selecting if bit is 1/0. 
            secret[i] = 0 if ((lwe_count - unif_count) > (advantage * num_samples)/2) else 1 #(advantage-2*tolerance)*num_samples) else 1

        # Predict results.
        invert = np.vectorize(lambda x: 1 - x)
        secret = secret.astype(int)
        match_counts = sum((secret == self.env.generator.secrets[sec_idx]).astype(int))
        match_counts2 = sum((invert(secret) == self.env.generator.secrets[sec_idx]).astype(int))
        # Report results. 
        if (match_counts == self.params.N) or (match_counts2 == self.params.N):
            logger.info(f'Distinguisher Method - all bits in secret {sec_idx} have been recovered!')
            logger.info(f'Distinguisher Method -Predicted secret {sec_idx}: {str(list(secret)) if match_counts > match_counts2 else str(list(invert(secret)))}, real secret {sec_idx}: {str(list(self.env.generator.secrets[sec_idx]))}')
            self.trainer.secret_match[sec_idx] = True
        else:
            logger.info(f'Distinguisher Method - Secret matching: {max(match_counts, match_counts2)}/{self.params.N} bits matched for secret {sec_idx}')
            logger.info(f'Distinguisher Method - Best secret guess for {sec_idx}: {str(list(secret)) if match_counts > match_counts2 else str(list(invert(secret)))}')
            if self.trainer.secret_match[sec_idx] != True:
                self.trainer.secret_match[sec_idx] = False



    def enc_dec_step(self, data_type, task, scores):
        """
        Encoding / decoding step.
        """
        params = self.params
        env = self.env
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ["lattice"]

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, dtype=torch.long)
        n_bitwise_valid = None
        n_total = torch.zeros(10000, dtype=torch.long)

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        # iterator
        iterator = self.env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=params.batch_size_eval,
            params=params,
            size=int(params.eval_size // params.N), 
        )

        for (x1, len1), (x2, len2), nb_ops in iterator:
            if n_bitwise_valid is None:
                bitlen = len2[0].cpu().int()-2 # Get rid of EOS characters at beginning and end.
                n_bitwise_valid = torch.zeros(size=(10000, bitlen), dtype=torch.long) # vector for bitwise accuracy. 

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()

            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)

            # forward / loss
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1_,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()

            # Get correct bits for each N level
            unique_ops = torch.unique(nb_ops)
            valid_bitwise = torch.zeros(size=(len(unique_ops), bitlen))
            for i, n in enumerate(unique_ops):
                    test = t[:, nb_ops == n].sum(1).long()  
                    valid_bitwise[i,:] = torch.flip(test, [0])[1:-1] # flip so high bit on the left

            # export evaluation details
            if params.eval_verbose:
                for i in range(len(len1)):
                    src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                    tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)
                    s = f"Equation {n_total.sum().item() + i} "
                    s += f"({'Valid' if valid[i] else 'Invalid'})\n"
                    s += f"src={src}\ntgt={tgt}\n"
                    if params.eval_verbose_print:
                        logger.info(s)
                    f_export.write(s + "\n")
                    f_export.flush()

            # stats
            xe_loss += loss.item() * len(y)
            n_valid.index_add_(-1, nb_ops, valid)
            n_bitwise_valid.index_add_(0, unique_ops.int(), valid_bitwise.long())
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))
            #print(n_total)



        # evaluation details
        if params.eval_verbose:
            f_export.close()

        # log
        _n_valid = n_valid.sum().item()
        _n_bitwise_valid = n_bitwise_valid.sum(0) 
        _n_total = n_total.sum().item()

        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            f"equations were evaluated correctly."
        )
        

        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_acc"] = 100.0 * _n_valid / _n_total

        # bitwise accuracy
        total_bitwise = np.around((100. * _n_bitwise_valid / _n_total).numpy(), decimals=1).tolist()

        scores[f"{data_type}_{task}_bitwise_acc"] = str(total_bitwise)

        # evaluate secret accuracy
        logger.info('evaluating secret accuracy')
        for i in range(len(self.env.generator.secrets)):
            self.eval_secret(i, self.env.generator.Q-1, encoder, decoder)

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            e = env.decode_class(i)
            scores[f"{data_type}_{task}_acc_{e}"] = (
                100.0 * n_valid[i].item() / max(n_total[i].item(), 1)
            )
            scores[f"{data_type}_{task}_bitwise_acc_{e}"] = (
                (100 * n_bitwise_valid[i] / max(n_total[i].item(), 1)).tolist()
            )
            if n_valid[i].item() > 0:
                logger.info(
                    f"{e}: {n_valid[i].item()} / {n_total[i].item()} "
                    f"({100. * n_valid[i].item() / max(n_total[i].item(), 1)}%)"
                )

    def enc_dec_step_beam(self, data_type, task, scores, size=None):
        """
        Encoding / decoding step with beam generation and SymPy check.
        """
        params = self.params
        env = self.env
        max_beam_length = self.env.max_output_length
        encoder = (
            self.modules["encoder"].module
            if params.multi_gpu
            else self.modules["encoder"]
        )
        decoder = (
            self.modules["decoder"].module
            if params.multi_gpu
            else self.modules["decoder"]
        )
        encoder.eval()
        decoder.eval()
        assert params.eval_verbose in [0, 1, 2]
        assert params.eval_verbose_print is False or params.eval_verbose > 0
        assert task in ["lattice"]

        # evaluation details
        if params.eval_verbose:
            eval_path = os.path.join(
                params.dump_path, f"eval.beam.{data_type}.{task}.{scores['epoch']}"
            )
            f_export = open(eval_path, "w")
            logger.info(f"Writing evaluation results in {eval_path} ...")

        def display_logs(logs, offset):
            """
            Display detailed results about success / fails.
            """
            if params.eval_verbose == 0:
                return
            for i, res in sorted(logs.items()):
                n_valid = sum([int(v) for _, _, v in res["hyps"]])
                s = f"Equation {offset + i} ({n_valid}/{len(res['hyps'])})\n"
                s += f"src={res['src']}\ntgt={res['tgt']}\n"
                for hyp, score, valid in res["hyps"]:
                    if score is None:
                        s += f"{int(valid)} {hyp}\n"
                    else:
                        s += f"{int(valid)} {score :.3e} {hyp}\n"
                if params.eval_verbose_print:
                    logger.info(s)
                f_export.write(s + "\n")
                f_export.flush()

        # stats
        xe_loss = 0
        n_valid = torch.zeros(10000, params.beam_size, dtype=torch.long)
        n_total = torch.zeros(10000, dtype=torch.long)

        # iterator
        iterator = env.create_test_iterator(
            data_type,
            task,
            data_path=self.trainer.data_path,
            batch_size=(max(int(params.batch_size // params.N)+1,1) if params.reload_data == "" else params.batch_size),
            params=params,
            size=int(params.eval_size // params.N),
        )
        eval_size = len(iterator.dataset)
        n_perfect_match = 0
        n_correct = 0

        for (x1, len1), (x2, len2), nb_ops in iterator:

            # target words to predict
            alen = torch.arange(len2.max(), dtype=torch.long, device=len2.device)
            pred_mask = (
                alen[:, None] < len2[None] - 1
            )  # do not predict anything given the last target word
            y = x2[1:].masked_select(pred_mask[:-1])
            assert len(y) == (len2 - 1).sum().item()
            # cuda
            x1_, len1_, x2, len2, y = to_cuda(x1, len1, x2, len2, y)
            bs = len(len1)

            # forward
            encoded = encoder("fwd", x=x1_, lengths=len1_, causal=False)
            decoded = decoder(
                "fwd",
                x=x2,
                lengths=len2,
                causal=True,
                src_enc=encoded.transpose(0, 1),
                src_len=len1_,
            )
            word_scores, loss = decoder(
                "predict", tensor=decoded, pred_mask=pred_mask, y=y, get_scores=True
            )

            # correct outputs per sequence / valid top-1 predictions
            t = torch.zeros_like(pred_mask, device=y.device)
            t[pred_mask] += word_scores.max(1)[1] == y
            valid = (t.sum(0) == len2 - 1).cpu().long()
            n_perfect_match += valid.sum().item()


            # save evaluation details
            beam_log = {}
            for i in range(len(len1)):
                src = idx_to_infix(env, x1[1 : len1[i] - 1, i].tolist(), True)
                tgt = idx_to_infix(env, x2[1 : len2[i] - 1, i].tolist(), False)

                if valid[i]:
                    beam_log[i] = {"src": src, "tgt": tgt, "hyps": [(tgt, None, True)]}

            # stats
            xe_loss += loss.item() * len(y)
            n_valid[:, 0].index_add_(-1, nb_ops, valid)
            n_total.index_add_(-1, nb_ops, torch.ones_like(nb_ops))

            # continue if everything is correct. if eval_verbose, perform
            # a full beam search, even on correct greedy generations
            if valid.sum() == len(valid) and params.eval_verbose < 2:
                display_logs(beam_log, offset=n_total.sum().item() - bs)
                continue

            # invalid top-1 predictions - check if there is a solution in the beam
            invalid_idx = (1 - valid).nonzero().view(-1)
            logger.info(
                f"({n_total.sum().item()}/{len(valid)}) Found "
                f"{bs - len(invalid_idx)}/{bs} valid top-1 predictions. "
                f"Generating solutions ..."
            )

            # generate
            _, _, generations = decoder.generate_beam(
                encoded.transpose(0, 1),
                len1_,
                beam_size=params.beam_size,
                length_penalty=params.beam_length_penalty,
                early_stopping=params.beam_early_stopping,
                max_len=max_beam_length,
            )

            # prepare inputs / hypotheses to check
            # if eval_verbose < 2, no beam search on equations solved greedily
            inputs = []
            for i in range(len(generations)):
                if valid[i] and params.eval_verbose < 2:
                    continue
                for j, (score, hyp) in enumerate(
                    sorted(generations[i].hyp, key=lambda x: x[0], reverse=True)
                ):
                    # DO BITWISE TEST HERE.
                    inputs.append(
                        {
                            "i": i,
                            "j": j,
                            "score": score,
                            "src": x1[1 : len1[i] - 1, i].tolist(),
                            "tgt": x2[1 : len2[i] - 1, i].tolist(),
                            "hyp": hyp[1:].tolist(),
                            "task": task,
                        }
                    )

            # check hypotheses with multiprocessing
            outputs = []
            if params.windows is True:
                for inp in inputs:
                    outputs.append(check_hypothesis(inp))
            else:
                with ProcessPoolExecutor(max_workers=20) as executor:
                    for output in executor.map(check_hypothesis, inputs, chunksize=1):
                        outputs.append(output)

            # read results
            bitwise_acc = []
            diffs = []
            for i in range(bs):

                # select hypotheses associated to current equation
                gens = sorted([o for o in outputs if o["i"] == i], key=lambda x: x["j"])
                assert (len(gens) == 0) == (valid[i] and params.eval_verbose < 2) and (
                    i in beam_log
                ) == valid[i]
                if len(gens) == 0:
                    continue

                # source / target
                src = gens[0]["src"]
                tgt = gens[0]["tgt"]
                beam_log[i] = {"src": src, "tgt": tgt, "hyps": []}

                # for each hypothesis
                for j, gen in enumerate(gens):

                    # sanity check
                    assert (
                        gen["src"] == src
                        and gen["tgt"] == tgt
                        and gen["i"] == i
                        and gen["j"] == j
                    )

                    # if hypothesis is correct, and we did not find a correct one before
                    is_valid = gen["is_valid"]
                    is_b_valid = is_valid >= 0.0 and is_valid == 1
                    if is_valid >= 0.0 and not valid[i]:
                        #print('beam found something valid that was not found before')
                        n_correct += 1
                        if is_valid == 1:
                            n_valid[nb_ops[i], j] += 1
                            valid[i] = 1

                    bitwise_acc.append(gen['bitwise'])
                    diffs.append(gen['difference'])
                    # update beam log
                    beam_log[i]["hyps"].append((gen["hyp"], gen["score"], is_b_valid))

            # valid solutions found with beam search
            logger.info(
                f"    Found {valid.sum().item()}/{bs} solutions in beam hypotheses."
            )

            # export evaluation details
            if params.eval_verbose:
                assert len(beam_log) == bs
                display_logs(beam_log, offset=n_total.sum().item() - bs)

        # evaluation details
        if params.eval_verbose:
            f_export.close()
            logger.info(f"Evaluation results written in {eval_path}")

        # log
        _n_valid = n_valid.sum().item()
        _n_total = n_total.sum().item()
        _n_bitwise_valid = np.sum(np.array(bitwise_acc),axis=0)
        
        logger.info(
            f"{_n_valid}/{_n_total} ({100. * _n_valid / _n_total}%) "
            f"equations were evaluated correctly."
        )

        # compute perplexity and prediction accuracy
        #assert _n_total == eval_size
        scores[f"{data_type}_{task}_xe_loss"] = xe_loss / _n_total
        scores[f"{data_type}_{task}_beam_acc"] = 100.0 * _n_valid / _n_total
        scores[f"{data_type}_{task}_perfect"] = 100.0 * n_perfect_match / _n_total
        scores[f"{data_type}_{task}_correct"] = (
            100.0 * (n_perfect_match + n_correct) / _n_total
        )
        # bitwise accuracy
        try:
            total_bitwise = np.around((100 * _n_bitwise_valid / len(bitwise_acc)), decimals=1).tolist()[::-1]
        except Exception as e:
            print('bitwise accuracy calc failed')
            total_bitwise = [0,0]
        scores[f"{data_type}_{task}_bitwise_acc"] = str(total_bitwise)

        # average difference
        percs_diff = np.array(diffs) / params.Q
        percs = [0.1*i for i in range(11)]
        percentiles = []
        for p in percs[1:]:
            pval = np.around(len(percs_diff[(percs_diff <= p)]) / len(percs_diff), decimals=2) * 100
            percentiles.append(pval)
        # Get the > 100% values
        percentiles.append(100) # Everything else is above 1.
        percQ10 = percentiles[0]
        scores[f"{data_type}_{task}_percs_diff"] = str(percentiles) # can use this as a stopping critirion if you want. 
        logger.info(f'percentiles of %Q difference between tgt and hyp: {str(list(percentiles))}')

        # evaluate secret accuracy
        fixedK = [239145, 42899, params.Q, 3*params.Q+7, 42900]
        randomK = np.random.randint(params.Q, 10*params.Q, 5)
        fixedK.extend(randomK)
        smallK = [71, 92, 101,193, 241]
        fixedK.extend(smallK)
        logger.info('evaluating secret accuracy')
        for i in range(len(self.env.generator.secrets)):
            for K in fixedK:
                self.eval_secret(i, K, encoder, decoder)

        # per class perplexity and prediction accuracy
        for i in range(len(n_total)):
            if n_total[i].item() == 0:
                continue
            e = env.decode_class(i)
            logger.info(
                f"{e}: {n_valid[i].sum().item()} / {n_total[i].item()} "
                f"({100. * n_valid[i].sum().item() / max(n_total[i].item(), 1)}%)"
            )
            scores[f"{data_type}_{task}_beam_acc_{e}"] = (
                100.0 * n_valid[i].sum().item() / max(n_total[i].item(), 1)
            )

        # Run distinguisher. 
        if percQ10 >= 25: # Not worth it running before this, numSamples will be too large. 
            for i in range(len(self.env.generator.secrets)):
                self.run_distinguisher(i, scores, encoder, decoder)

