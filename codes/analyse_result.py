"""
*
*     SOFTWARE NAME
*
*        File:  analyse_result.py
*
*     Authors: Deleted for purposes of anonymity
*
*     Proprietor: Deleted for purposes of anonymity --- PROPRIETARY INFORMATION
*
* The software and its source code contain valuable trade secrets and shall be maintained in
* confidence and treated as confidential information. The software may only be used for
* evaluation and/or testing purposes, unless otherwise explicitly stated in the terms of a
* license agreement or nondisclosure agreement with the proprietor of the software.
* Any unauthorized publication, transfer to third parties, or duplication of the object or
* source code---either totally or in part---is strictly prohibited.
*
*     Copyright (c) 2022 Proprietor: Deleted for purposes of anonymity
*     All Rights Reserved.
*
* THE PROPRIETOR DISCLAIMS ALL WARRANTIES, EITHER EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO IMPLIED WARRANTIES OF MERCHANTABILITY
* AND FITNESS FOR A PARTICULAR PURPOSE AND THE WARRANTY AGAINST LATENT
* DEFECTS, WITH RESPECT TO THE PROGRAM AND ANY ACCOMPANYING DOCUMENTATION.
*
* NO LIABILITY FOR CONSEQUENTIAL DAMAGES:
* IN NO EVENT SHALL THE PROPRIETOR OR ANY OF ITS SUBSIDIARIES BE
* LIABLE FOR ANY DAMAGES WHATSOEVER (INCLUDING, WITHOUT LIMITATION, DAMAGES
* FOR LOSS OF BUSINESS PROFITS, BUSINESS INTERRUPTION, LOSS OF INFORMATION, OR
* OTHER PECUNIARY LOSS AND INDIRECT, CONSEQUENTIAL, INCIDENTAL,
* ECONOMIC OR PUNITIVE DAMAGES) ARISING OUT OF THE USE OF OR INABILITY
* TO USE THIS PROGRAM, EVEN IF the proprietor HAS BEEN ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGES.
*
* For purposes of anonymity, the identity of the proprietor is not given herewith.
* The identity of the proprietor will be given once the review of the
* conference submission is completed.
*
* THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
*
"""


import pandas as pd
import numpy as np
import glob
import _pickle as cPickle
import matplotlib.pyplot as plt

def main():
    # get results
    files = glob.glob('../results/*pickle')
    files.sort()

    # metric names
    var_names = ['MRR', 'MR', 'HITS@1', 'HITS@3', 'HITS@10']

    # define index
    index1, index2, index3, index4 = [], [], [], []
    for j, fl in enumerate(files):
        with open(fl, 'rb') as f:
            title = fl.split('/')[-1].split('_')
            index1.append(title[0])
            index2.append(title[1])
            if title[3] == 'Mutup':
                index3.append(title[2] + '_' + title[3])
            else:
                index3.append(title[2])
            index4.append(title[-1][0])
    indexes = [index1, index2, index3, index4]

    # create dataframe
    data = np.zeros([len(files), 5])
    for j, fl in enumerate(files):
        with open(fl, 'rb') as f:
            test = cPickle.load(f)
            for i, var in enumerate(var_names):
                data[j, i] = test[var]
    
    index = pd.MultiIndex.from_arrays(indexes, names=('Model', 'Data', 'Method', 'trials'))
    data = pd.DataFrame(data, columns=var_names, index=index)
    data.to_hdf('Results.hdf5', key='df')
    
    #pdes = index.get_level_values(0).drop_duplicates()
    #num_pdes = len(pdes)
    #models = index.get_level_values(2).drop_duplicates()
    #num_models = len(models)
    #x = np.arange(num_pdes)
    #width = 0.5/(num_models-1)
    
    #fig, ax = plt.subplots(figsize=(8,6))
    #for i in range(num_models):
    #    pos = x-0.3 + 0.5/(num_models-1)*i
    #    ax.bar(pos, data[data.index.isin([models[i]],level=2)]['MSE'], width)
    
    #ax.set_xticks(x)
    #ax.set_xticklabels(pdes,fontsize=16)
    #ax.tick_params(axis='y',labelsize=16)
    #ax.set_yscale('log')
    #ax.set_xlabel('PDEs',fontsize=16)
    #ax.set_ylabel('MSE',fontsize=16)
    #fig.legend(models,loc=8,ncol=num_models,fontsize=16)
    #plt.tight_layout(rect=[0,0.1,1,1])
    #plt.savefig('Results.pdf')
    

if __name__ == "__main__":
    main()
    print("Done.")
