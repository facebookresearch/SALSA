# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

'''
Class to parse log files from SALSA experiments AND visualize them. 
'''

from re import S
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import seaborn as sns
from tabulate import tabulate
from scipy import stats

from parsers import main_parser 

sns.set_style('white') # plot style

class ExperimentDatabase():
    def __init__(self, xp_list, path='/checkpoint/ewenger/dumped', has_beam=False, xp_filter=[], xp_selector=[], verbose=False):
        self.path = path
        self.xp_list = xp_list
        self.has_beam = has_beam
        self.db, self.var_args, self.all_args = main_parser(self.path, self.xp_list, self.has_beam, xp_filter, xp_selector, verbose)

    def filter_exp(self, filters):
        """ 
        Will return a subset of the dataframe (also a dataframe) which only contains experiments with params matching the filter.
        Example filter: filter={"N": [20, 30]}.
        """ 
        assert len(filters) > 0 # Make sure you have some filters to work with. 
        idx_list = [self.db[f].isin(filters[f]) for f in filters]
        idx_combined = np.all(np.array(idx_list), axis=0) # Check if all idx are true. 
        return self.db[idx_combined] # Separate from central db. 

    def get_data(self, db):
        if (db is not None) and (type(db) != pd.core.frame.DataFrame):
            print('Must either provide a DataFrame input (hint: maybe from filter_exp function?) OR provide no input so function can act on its internal DataFrame.')
            return
        if db is not None:
            data = db
        else:
            data = self.db
        return data

    def print_table(self, db=None, exclude=['nans', 'early_stop_criterion', 'idx', 'stop_criterion', 'mse_weights']):
        data = self.get_data(db)
        cols = [c for c in data.columns if c not in exclude]
        print(tabulate(data[cols], headers='keys', tablefmt="pretty"))

    def training_curve(self, var, db=None, beg=0, legend=False):
        '''
        Prints training curves for either self.db OR 
        '''
        data=self.get_data(db)
        # Plot it!  
        sublist = data[var]
        for line in sublist:
            plt.plot(line[beg:])
        #        sns.lineplot(data=data, x="epochs", y=var, hue="xp") # make the experiments into the legend. 

    def plot_relationship(self, x_var, y_var, filters, plottype='scatter', acc_thresh=0.5, maxval=None, minval=None, custom_xticks=False, legend_outside=True):
        """
        Plots a relationship between 2 variables 
        """
        # Get the data.
        if len(filters) > 0:
            df = self.filter_exp(filters)
        else:
            df = self.db
        df = df[df['best_acc'] > acc_thresh]

        # Plot the data.
        if plottype == 'scatter':
            b = sns.scatterplot(x=x_var, y=y_var, data=df) #ys, xs)
        elif plottype == 'boxplot':
            b = sns.boxplot(x_var,y_var, data=df, width=0.6)
        elif plottype == 'bar':
            b = sns.barplot(x_var, y_var, data=df, color='skyblue')
        elif plottype == 'violin':
            b = sns.violinplot(x_var, y_var, data=df)
        elif plottype == 'acc_cdf':
            b = sns.ecdfplot(data=df, x='best_acc', hue=x_var, palette='Set2')
            sorted_x = np.unique(df['x_var']) # Get elements for legend.
            x_var = y_var
            y_var = 'Proportion'
        elif plottype == 'work':
            # Make sure you're working with the right variables. 
            if y_var != 'Q' and x_var != 'best_epoch':
                print('To plot work, use best_epoch and Q as x_var and y_var, respectively.')
                return
            else: 
                if 'num_train_batches' in df:
                    batch_count = df['num_train_batches'][0]
                else:
                    batch_count = self.all_args['num_train_batches']
                df['log(Q)'] = df['Q'].apply(lambda x: np.ceil(np.log2(x)))
                df['log(NumSamples)'] = df['best_epoch'].apply(lambda x: np.ceil(np.log2((x) * batch_count)))
                slope, intercept, _,_,_ = stats.linregress(df['log(Q)'],df['log(NumSamples)'])
                # use line_kws to set line label for legend
                b = sns.regplot(x="log(Q)", y="log(NumSamples)", data=df, 
                line_kws={'label':"y={0:.1f}x+{1:.1f}".format(slope,intercept)})
                # plot legend
                b.legend(fontsize=12)
                x_var = 'log(Q)'
                y_var = 'log(NumSamples)'
        else:
            print('Valid plot options are: boxplot, bar, violin, acc_cdf, and work.')

        if plottype == 'acc_cdf' and legend_outside:
            box = b.get_position()
            b.set_position([box.x0, box.y0, box.width * 0.85, box.height]) # resize position
            _,l = b.get_legend_handles_labels()
            l = sorted_x
            b.legend(l, title=x_var, loc='center right', bbox_to_anchor=(1.2, 0.5), ncol=1, fontsize=10)


        b.set_ylim(minval,maxval)
        b.set_title(f'{y_var} VS. {x_var}\n filters={filters}' + f', acc_thresh={acc_thresh}', fontsize=15) # if plottype=='work' else ''))
        b.set_xlabel(x_var, fontsize=18)
        b.set_ylabel(y_var, fontsize=18)
        plt.rcParams['figure.figsize'] = [10, 5]
        plt.show()
        
        



    