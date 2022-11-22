# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import os
import pickle
import glob
import numpy as np
import ast
from datetime import datetime
import re

#from torch._C import R


def main_parser(path, xp_list, has_beam, xp_filter, xp_selector, verbose):
    ''' 
    Log parser. 
    '''
    # Find the paths.
    var_args, all_args, xps = load_params_and_args(path, xp_list, xp_filter, xp_selector, verbose)

    # read experiments
    data = []
    indics = []
    indicator='valid_lattice'
    indics = ["beam_acc" if has_beam is True else "acc","xe_loss", 'bitwise_acc', 'percs_diff']
    if has_beam:
        indics.extend(["correct", "perfect", "beam_acc_d1", "beam_acc_d2", "beam_acc_nb", "additional_1","additional_2","additional_3"])
    
    for (env, xp) in xps:
        res = parse(env, xp, indics, path, indicator)
        res.update(vars_from_env_xp(path, var_args, env, xp))
        data.append(res)

    # print stuff if verbose.
    if verbose:
        print(len(data), "experiments read")
        print(len([d for d in data if d["stderr"] is False]),"stderr not found")
        print(len([d for d in data if d["error"] is True]),"runtime errors")
        print(len([d for d in data if "oom" in d and d["oom"] is True]),"oom errors")
        print(len([d for d in data if "terminated" in d and d["terminated"] is True]),"exit code 1")
        print(len([d for d in data if "forced" in d and d["forced"] is True]),"Force Terminated")
        print(len([d for d in data if "last_epoch" in d and d["last_epoch"] >= 0]),"started XP")
        print(len([d for d in data if "ended" in d and d["ended"] is True]),"ended XP")
        print(len([d for d in data if "best_acc" in d and float(d["best_acc"]) > 0.0]),"began predicting")

    # make it into a pandas database. 
    return pd.DataFrame(data), var_args, all_args


################## HELPER FUNCTIONS #################


def load_params_and_args(path, xp_list, xp_filter, xp_selector, verbose):
    ''' Search for experiments at designated path. '''
    var_args = set()
    all_args = {}

    unwanted_args = ['dump_path', 'master_addr']
    xps = [(env, xp) for env in xp_list for xp in os.listdir(path+'/'+env) if (len(xp_selector)==0 or xp in xp_selector) and (len(xp_filter)==0 or not xp in xp_filter)]
    print(xps)
    names = [path + env + '/' + xp for (env, xp) in xps]
    print(names)

    if verbose:
        print(names, len(names),"experiments found")

    # read all args
    pickled_xp = 0
    for name in names:
        pa = name+'/params.pkl'
        if not os.path.exists(pa):
            if verbose:
                print("Unpickled experiment: ", name)
            continue
        pk = pickle.load(open(pa,'rb'))
        all_args.update(pk.__dict__)
        pickled_xp += 1

    # find variable args
    for name in names:
        pa = name+'/params.pkl'
        if not os.path.exists(pa):
            continue
        pk = pickle.load(open(pa,'rb'))
        for key,value in all_args.items():
            if key in pk.__dict__ and value == pk.__dict__[key]:
                continue
            if key not in unwanted_args:
                var_args.add(key)
    
    # print helpful things if verbose.
    if verbose:
        print(pickled_xp, "pickled experiments found")
        print()
        print("common args")
        for key in all_args:
            if key not in unwanted_args and key not in var_args:
                print(key,"=", all_args[key])
        print()
                    
        print(len(var_args)," variables params out of", len(all_args))
        print(var_args)

    return var_args, all_args, xps

def vars_from_env_xp(path, var_args, env, xp):
    res = {}
    pa = path+env+'/'+xp+'/params.pkl'
    if not os.path.exists(pa):
        print("pickle", pa, "not found")
        return res
    pk = pickle.load(open(pa,'rb'))
    for key in var_args:
        if key in pk.__dict__: 
            res[key] = pk.__dict__[key]
        else:
            res[key] = None
    return res

def get_start_time(line):
    parsed_line = line.split(" ")
    dt = datetime.strptime(parsed_line[2]+' '+parsed_line[3],"%m/%d/%y %H:%M:%S")
    try:
        idx = parsed_line.index("epoch")
        curr_epoch = int(parsed_line[idx+1])
    except ValueError:
        curr_epoch = ""
    return dt, curr_epoch



def parse(env, xp, indics, path, indicator="valid_lattice"):
    #print(f'Starting {xp}')
    res = {"env":env, "xp": xp, "stderr":False, "log":False, "error":False}
    stderr_file = os.path.join(os.path.expanduser("~"), 'workdir/'+env+'/*/'+xp+'.stderr')
    nb_stderr =len(glob.glob(stderr_file))
    secret = False
    recover_methods = []
    MAXLEN = 500000
    if nb_stderr > 1:
        print("duplicate stderr", env, xp)
        return res
    for name in glob.glob(stderr_file):
        with open(name, 'rt') as f:
            res["stderr"]=True
            errlines = []
            cuda = False
            terminated = False
            forced = False
            for i, line in enumerate(f):
                if line.find("RuntimeError:") >= 0:
                    errlines.append(line)
                if line.find("CUDA out of memory") >= 0:
                    cuda = True
                if line.find("Exited with exit code 1") >=0:
                    # print(stderr_file)
                    terminated = True
                
                if line.find("Force Terminated") >=0:
                    # print(stderr_file)
                    forced = True
                if (line.find(' bits in secret 0 have been recovered!')>=0) or (line.find('Found secret match - ending experiment.') >= 0) or (line.find('Distinguisher Method - all bits in secret {sec_idx} have been recovered!')>=0): # Bc of weird bug where sometimes secret is only reported in err file. 
                    #print('here', xp)
                    secret = True
                    if line.find('Distinguisher') >=0:
                        recover_methods.append('d')
                    elif line.find('K') >=0:
                        kval = int(line.split('=')[-1].rstrip())
                        recover_methods.append(kval)
                #if i > MAXLEN:
                #    break
            res["forced"] = forced
            res["terminated"] = terminated
            if len(errlines) > 0:    
                res["error"] = True
                res["runtime_errors"] = errlines
                res["oom"] = cuda 
                if not cuda:
                    print(stderr_file,"runtime error no oom")
                
               
    pa = path+env+'/'+xp+'/train.log'
    if not os.path.exists(pa):
        return res
    res["log"] = True
    with open(pa, 'rt') as f:
        series = []
        train_loss=[]
        bitwise = []
        q_abs = []
        q_perc = []
        secret_guesses = [[]]
        secret_real = []
        for ind in indics:
            series.append([])
        best_val = -1.0
        best_xel = 999999999.0
        best_epoch = -1
        epoch = -1
        val = -1
        ended = False
        nanfound = False
        midsecret = False
        res["curr_epoch"]=-1
        res["train_time"]=0
        res["eval_time"]=0
        nb_sig10 = 0
        nb_sig15 = 0
        prop_ones = []
        prop_zeros = []
        total_prop = []
        curr_prop_ones = []
        curr_prop_zeros = []
        curr_total_prop = []
        #idx_prop_ones = 0
        for i, line in enumerate(f):
            try:
                if line.find("Signal handler called with signal 10") >= 0:
                    nb_sig10 += 1
                if line.find("Signal handler called with signal 15") >= 0:
                    nb_sig15 += 1
                if line.find("Stopping criterion has been below its best value for more than") >=0:
                    ended = True
                elif line.find(" Starting epoch") >=0:
                    dt, curr_epoch = get_start_time(line)
                    res["start_time"] = dt
                    if curr_epoch >0 and curr_epoch == res["curr_epoch"]+1:
                        res["eval_time"] += (dt - res["end_time"]).total_seconds()
                    res["curr_epoch"] = curr_epoch
                    if curr_epoch > 0:
                        #print(curr_prop_ones, np.max(curr_prop_ones), curr_total_prop, np.max(curr_total_prop))
                        #print('here')
                        prop_idx = np.argmax(curr_prop_ones)
                        prop_ones.append(np.round(curr_prop_ones[prop_idx],2))
                        prop_zeros.append(np.round(curr_prop_zeros[prop_idx],2))
                        total_prop.append(np.round(curr_total_prop[prop_idx],2))
                        curr_prop_ones = []
                        curr_total_prop = []
                        curr_prop_zeros = []
                elif line.find("============ End of epoch") >=0:
                    dt, curr_epoch = get_start_time(line)
                    if curr_epoch != res["curr_epoch"]:
                        pass
                        #print("epoch mismatch", curr_epoch,"in", env,",", xp)
                    else:
                        res["end_time"] = dt
                        res["train_time"] += (dt-res["start_time"]).total_seconds()
                elif line.find("- model LR:") >=0:
                    loss = line.split(" ")[-5].strip()
                    train_loss.append(None if loss == 'nan' else float(loss)) 
                elif line.find("- LR:") >=0:
                    loss = line.split(" ")[-4].strip()
                    train_loss.append(None if loss == 'nan' else float(loss)) 
                # save off the secret
                elif (midsecret==True) or (line.find('- secrets: ') > 0):
                    if not midsecret:
                        sec = line.split('secrets: ')[1].rstrip()
                        midsecret = True
                    else:
                        line = line.rstrip()
                        sec += line
                        if line.endswith('])]'):
                            sec = re.sub('  +', ' ', sec)
                            mod_sec = ast.literal_eval(sec.replace('array(', '').replace(')', ''))
                            secret_real = mod_sec
                            midsecret = False

                # check for Q difference
                elif line.find('average difference') >= 0:
                    t = line.split('hyp: ')[1].split(', ')
                    q_abs.append(float(t[0]))
                    q_perc.append(float(t[1].split('% ')[0]))
                elif line.find('Best secret guess') >= 0:
                    idx, sec = line.split(': ')
                    idx = int(idx.split(' ')[-1])
                    sec = ast.literal_eval(sec)
                    while len(secret_guesses) < idx+1:
                        secret_guesses.append([])
                    secret_guesses[idx].append(sec)
                    sec = np.array(sec)
                    rsec = np.array(secret_real[0]) # TODO update when you have > 1 secret.
                    #print(sec, rsec)
                    for i in [0,1]:
                        r = np.sum(np.equal(sec[np.where(rsec==i)], rsec[np.where(rsec==i)]).astype(int))/len(rsec[np.where(rsec==i)])
                        if i == 0:
                            curr_prop_zeros.append(r)
                        else:
                            curr_prop_ones.append(r)
                    
                elif line.find('Predicted secret') >0:
                    sec1, sec2 = line.split(', real')
                    idx, sec1 = sec1.split(': ')
                    #print(idx, idx.split(' '))
                    idx = int(idx.split(' ')[-1])
                    if len(secret_guesses) != (idx+1):
                        secret_guesses.append([])
                    sec1 = ast.literal_eval(sec1)
                    secret_guesses[idx].append(sec1)
                elif line.find("Secret matching:") >0:
                    matched = line.split('Secret matching: ')[1].split(' ')[0]
                    m1, m2 = matched.split('/')
                    ham =  sum(secret_real[0])
                    num_found = int(m2) - int(m1)
                    #prop_ones_new = 1 - (num_found / ham)
                    #curr_prop_ones.append(prop_ones_new)
                    #print(int(m1)/int(m2), m1, m2)
                    curr_total_prop.append(np.round(int(m1)/int(m2), 2))
                    #rint(num_found, sum(secret_real[0]), len(secret_real[0]), prop_ones_new)
                    #if (num_found <= ham): # and (prop_ones_new > prop_ones):
                    #    prop_ones.append(prop_ones_new)
                    #else:
                    #    prop_ones.append(0)
                        #print(prop_ones)
                    #    prop_ones = prop_ones_new
                    #    idx_prop_ones = curr_epoch
                    
                elif line.find(" bits in secret 0 have been recovered!") >=0:
                    # Figure out which method found it. 
                    if line.find('Distinguisher') >=0:
                        if 'd' not in recover_methods:
                            recover_methods.append('d')
                    elif line.find('K') >=0:
                        kval = int(line.split('=')[-1].rstrip())
                        if kval not in recover_methods:
                            recover_methods.append(kval)
                    
                # see if secret was found:
                elif (line.find('Found secret match - ending experiment') > 0) or (line.find('secret match')>0):
                    secret=True
                    prop_ones.append(1)
                    prop_zeros.append(1)
                    total_prop.append(1)
                    break

                elif line.find('bitwise acc') >=0:
                    try:
                        bw = line.split('bitwise acc: ')[1]
                        bitwise.append(ast.literal_eval(bw))
                    except:
                        print('wrong bitwise acc line')
                elif line.find('An unknown exception of type') >=0:
                    pass
                    #print('error in log file')
                    #break
                else:                 
                    pos = line.find('__log__:')
                    if pos >=0:
                        if line[pos+8:].find(': NaN,') >= 0:
                            nanfound = True
                            line = line.replace(': NaN,',': -1.0,')
                        dic = ast.literal_eval(line[pos+8:])
                        epoch = dic["epoch"]
                        if not indicator+"_"+indics[0] in dic: 
                            continue
                        if not indicator+"_"+indics[1] in dic: 
                            continue
                        # handle the fact that sometimes you won't use beam acc or you might have mixed eval
                        if "valid_lattice_beam_acc" in dic.keys():
                            val = dic["valid_lattice_beam_acc"]# indics[0]]
                        else:
                            val = dic[indicator+"_"+indics[0]]
                        xel = dic[indicator+"_"+indics[1]]
                        if xel < best_xel:
                            best_xel= xel
                        if val > best_val:
                            best_val = val
                            best_epoch = epoch 
                        for j, indic in enumerate(indics):
                            if indicator+"_"+indic in dic:
                                if ('bitwise' in indic) or ('percs_diff' in indic):
                                    series[j].append(ast.literal_eval(dic[indicator+"_"+indic]))
                                else:
                                    series[j].append(dic[indicator+"_"+indic])
            except Exception as e:
                print(e, "exception in", env, xp)
                continue
            #if i > MAXLEN:
            #    break
                
        if secret==True and total_prop[-1] != 1:
            prop_ones.append(1)
            prop_zeros.append(1)
            total_prop.append(1)
        res["nans"] = nanfound
        res["ended"] = (ended or (nb_sig15 > nb_sig10))
        res["secret_found"] = secret
        res["secret"] = secret_real
        res["secret_guesses"] = secret_guesses
        res["last_epoch"] = epoch
        res["last_acc"] = "{:.2f}".format(val)
        res["best_epoch"] = best_epoch
        res["best_acc"] = float("{:.2f}".format(best_val))
        res["best_xeloss"] = "{:.2f}".format(best_xel)
        res["train_loss"]=train_loss
        res["recover_method"] = recover_methods
        #print(prop_ones)
        res["prop_ones"] = prop_ones
        res["prop_zeros"] = prop_zeros
        res["total_prop"] = total_prop
        res["best_prop_ones"] = np.round(np.max(prop_ones),2) if len(prop_ones) > 0 else 0
        res["best_prop_zeros"] = np.round(np.max(prop_zeros),2) if len(prop_zeros) > 0 else 0
        res["best_total_prop"] = np.round(np.max(total_prop),2) if len(total_prop)>0 else 0
        res["idx_best_total_prop"] = np.argmax(total_prop) if len(total_prop) >0 else -1
        res["idx_best_prop_ones"] = np.argmax(prop_ones) if len(prop_ones) > 0 else -1
        #res["diff_best_prop_ones"] = idx_prop_ones# , series[indics.index('percs_diff')][idx_prop_ones])
        if epoch >=0:
            res["train_time"] /= (epoch+1)
            res["eval_time"] /= (epoch+1)
        res["train_time"] = int(res["train_time"]+0.5) 
        res["eval_time"] = int(res["eval_time"]+0.5) 
        res['diff_abs'] = q_abs
        res['diff_percQ'] = q_perc
            
        #print(series, indics)
        for i,indic in enumerate(indics):
            if ('bitwise' not in indic) and ('percs_diff' not in indic):
                res["last_"+indic] = "{:.2f}".format(series[i][-1]) if len(series[i])>0 else '0'
                res["best_"+indic] = "{:.2f}".format(max(series[i])) if len(series[i])>0 else '0'
                res[indic] = series[i]
            else:
                res["last_"+indic] = "{}".format(series[i][-1]) if len(series[i])>0 else '0'
                res["best_"+indic] = "{}".format(max(series[i])) if len(series[i])>0 else '0'
                res[indic] = series[i]

        if len(bitwise) > 0:
            res["bitwise_acc"] = bitwise

            #if len(series[i])!= epoch + 1:
            #    print("mismatch in nr of epochs",env, xp, epoch+1, len(series[i]), indic)      
    return res
