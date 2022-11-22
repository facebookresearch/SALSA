# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Loads model and runs distinguisher. Helpful for testing.

Will load by default the provided checkpoint in ./checkpoint/example/test_run
'''

from distinguisher_helper import *
import argparse
import os
import sys
sys.path.append('../')
from src.logger import create_logger

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to run on?')
    # Experiment base path
    parser.add_argument('--basepath', type=str, default='',
                        help='path from which to load the model')
    # Advantages to try
    parser.add_argument('--acc_within_tolerance', type=str, default='0.3', help='advantages to try - default is 0.3')
    # percQ to try
    parser.add_argument('--tolerance', type=str, default='0.1', help='percQs to try')
    # Dump path
    parser.add_argument('--dumppath', type=str, default='', 
                        help='path to dump results in')
    return parser

def get_logger(args):
    # create a logger
    logger = create_logger(os.path.join(args.dumppath, 'distinguisher.log'), rank=0) #getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("")
    return logger


def predict_outputs(data, params, trainer, encoder, decoder):
    '''
    Predicts beam outputs in batches of 16 to avoid oom issues. 
    '''
    batch_size = 32
    encA, lenA, encB, lenB, y = data
    
    preds = []
    bs = []
    # Generate predictions 
    for i in range(0, encA.shape[1], batch_size):
        # Get the outpu
        beam_log_real = run_beam_generation(encA[:, i:i+batch_size], encB[:, i:i+batch_size], lenA[i:i+batch_size], lenB[i:i+batch_size], params, trainer, encoder, decoder)
        # Decode output
        rl = [trainer.env.output_encoder.decode(beam_log_real[b]['hyps'][0][0][::-1])[0] for b in beam_log_real]
        b = [trainer.env.output_encoder.decode(beam_log_real[b]['tgt'][::-1])[0] for b in beam_log_real]
        preds.extend(rl)
        bs.extend(b)
    return np.array(preds), np.array(bs)

def run_distinguisher(tolerance, acc_within_tolerance, params, trainer, encoder, decoder, logpath):
    # Set up file handle.
    fhandle = open(logpath + '/distinguisher.log', 'a+')
    fhandle.write(f'Running distinguisher with acc_within_tol={acc_within_tolerance} and tolerance={tolerance}\n')
    # Set up rng for sample generation.
    rng = np.random.RandomState([5])
    
    # Set up the bound
    bound = tolerance * params.Q
    
    # Set up an empty vector to hold secret. 
    secret = np.ones(params.N)
    advantage = (acc_within_tolerance-2*tolerance)

    # Compute samples used
    samples_used = max(int(2 // ( advantage ** 2)),50)

    results = {}
    
    for i in range(params.N):
        # Get the data. 
        real_data, orig_lwe_data = get_samples(i, samples_used, False, rng, params, trainer.env, g=0, sec_idx=0)
        unif_data, _ = get_samples(i, samples_used, True, rng, params, trainer.env, g=0, sec_idx=0)
        
        # Run through the model. 
        lwe_pred, _ = predict_outputs(real_data, params, trainer, encoder, decoder)
        unif_pred, unif_real = predict_outputs(unif_data, params, trainer, encoder, decoder)

        # EJW 11/4/22
        lwe_orig_pred, _ = predict_outputs(orig_lwe_data, params, trainer, encoder, decoder)

        # Get the differences. 
        #lwe_diff = abs(lwe_real - lwe_pred)
        # EJW 11/4/22
        lwe_diff = abs(lwe_orig_pred - lwe_pred) # Compare the new prediction to the ORIGINAL LWE PREDICTION
        unif_diff = abs(unif_pred - unif_real)
        results[i] = {}
        results[i]['lwe'] = np.array(lwe_diff)
        results[i]['unif'] = np.array(unif_diff)
        #fhandle.write(f"lwe_orig diff {lwe_diff}\n unif_diff {unif_diff}\n")

        # Count the number less than bound. 
        lwe_count = np.sum((lwe_diff < bound).astype(int))
        unif_count = np.sum((unif_diff < bound).astype(int))
        #fhandle.write(f"lwe_count {lwe_count}\n unif_count {unif_count}\n")
        
        # Predict secret bit. 
        fhandle.write(f'difference is {(lwe_count - unif_count)}/{np.round(advantage*samples_used/2, decimals=2)}\n')
        secret[i] = 0 if ((lwe_count - unif_count) > (advantage*samples_used/2)) else 1
        fhandle.write(f'secret bit {i} = {int(secret[i])}, real bit = {trainer.env.generator.secrets[0][i]}\n')
        
    if np.all(secret == trainer.env.generator.secrets[0]):
        fhandle.write("SECRET RECOVERED")
    fhandle.write("\n")
    results['bound'] = bound
    with open("diff_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    return secret

def main(args):
    # Get the percQ and advantages out -- will run tests with all combos of these.
    advs = [float(el) for el in args.acc_within_tolerance.split(' ')]
    percQ = [float(el) for el in args.tolerance.split(' ')]

    # Make sure the acc_within_tol and tolerance are between 0 and 1
    for el,p in zip(advs, percQ):
        assert el > 0 and el <= 1
        assert p > 0 and p <= 1
        
    # Load the model.
    params = get_params(args.basepath)
    encoder, decoder, env, modules, trainer, evaluator = load_model(params, args)

    # Run the distinguisher for all percQ, advantage combos. 
    for ad in advs:
        for p in percQ:
            _ = run_distinguisher(p, ad, params, trainer, encoder, decoder, args.dumppath)

    #logger.info('Finished!')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    # Make sure the basepath is real. 
    assert os.path.exists(args.basepath)
    # Make the dump path the basepath if not otherwise specified. 
    if (args.dumppath == '') or not os.path.exists(args.dumppath):
        args.dumppath = args.basepath

    # Run the parser
    main(args)
