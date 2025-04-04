"""
*          NAME OF THE PROGRAM THIS FILE BELONGS TO
*
*   file: dataloader.py
*
* Authors: Makoto Takamoto (makoto.takamoto@neclab.eu)
*          Daniel Onoro Rubio (Daniel.onoro@neclab.eu)

NEC Laboratories Europe GmbH, Copyright (c) 2025, All rights reserved.
*     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
*
*          PROPRIETARY INFORMATION ---

SOFTWARE LICENSE AGREEMENT
ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.

This is a license agreement ("Agreement") between your academic institution or non-profit organization or self (called "Licensee" or "You" in this Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this Agreement).  All rights not specifically granted to you in this Agreement are reserved for Licensor.
RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive ownership of any copy of the Software (as defined below) licensed under this Agreement and hereby grants to Licensee a personal, non-exclusive, non-transferable license to use the Software for noncommercial research purposes, without the right to sublicense, pursuant to the terms and conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF LICENSOR’S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this Agreement, the term "Software" means (i) the actual copy of all or any portion of code for program routines made accessible to Licensee by Licensor pursuant to this Agreement, inclusive of backups, updates, and/or merged copies permitted hereunder or subsequently supplied by Licensor,  including all or any file structures, programming instructions, user interfaces and screen formats and sequences as well as any and all documentation and instructions related to it, and (ii) all or any derivatives and/or modifications created or made by You to any of the items specified in (i).
CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is proprietary to Licensor, and as such, Licensee agrees to receive all such materials and to use the Software only in accordance with the terms of this Agreement.  Licensee agrees to use reasonable effort to protect the Software from unauthorized use, reproduction, distribution, or publication. All publication materials mentioning features or use of this software must explicitly include an acknowledgement the software was developed by NEC Laboratories Europe GmbH.
COPYRIGHT: The Software is owned by Licensor.
PERMITTED USES:  The Software may be used for your own noncommercial internal research purposes. You understand and agree that Licensor is not obligated to implement any suggestions and/or feedback you might provide regarding the Software, but to the extent Licensor does so, you are not entitled to any compensation related thereto.
DERIVATIVES: You may create derivatives of or make modifications to the Software, however, You agree that all and any such derivatives and modifications will be owned by Licensor and become a part of the Software licensed to You under this Agreement.  You may only use such derivatives and modifications for your own noncommercial internal research purposes, and you may not otherwise use, distribute or copy such derivatives and modifications in violation of this Agreement.
BACKUPS:  If Licensee is an organization, it may make that number of copies of the Software necessary for internal noncommercial use at a single site within its organization provided that all information appearing in or on the original labels, including the copyright and trademark notices are copied onto the labels of the copies.
USES NOT PERMITTED:  You may not distribute, copy or use the Software except as explicitly permitted herein. Licensee has not been granted any trademark license as part of this Agreement. Neither the name of NEC Laboratories Europe GmbH nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission.
You may not sell, rent, lease, sublicense, lend, time-share or transfer, in whole or in part, or provide third parties access to prior or present versions (or any parts thereof) of the Software.
ASSIGNMENT: You may not assign this Agreement or your rights hereunder without the prior written consent of Licensor. Any attempted assignment without such consent shall be null and void.
TERM: The term of the license granted by this Agreement is from Licensee's acceptance of this Agreement by downloading the Software or by using the Software until terminated as provided below.
The Agreement automatically terminates without notice if you fail to comply with any provision of this Agreement.  Licensee may terminate this Agreement by ceasing using the Software.  Upon any termination of this Agreement, Licensee will delete any and all copies of the Software. You agree that all provisions which operate to protect the proprietary rights of Licensor shall remain in force should breach occur and that the obligation of confidentiality described in this Agreement is binding in perpetuity and, as such, survives the term of the Agreement.
FEE: Provided Licensee abides completely by the terms and conditions of this Agreement, there is no fee due to Licensor for Licensee's use of the Software in accordance with this Agreement.
DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON-INFRINGEMENT.  LICENSEE BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND RELATED MATERIALS.
SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is provided as part of this Agreement.
EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.
EXPORT REGULATION: Licensee agrees to comply with any and all applicable export control laws, regulations, and/or other laws related to embargoes and sanction programs administered by law.
SEVERABILITY: If any provision(s) of this Agreement shall be held to be invalid, illegal, or unenforceable by a court or other tribunal of competent jurisdiction, the validity, legality and enforceability of the remaining provisions shall not in any way be affected or impaired thereby.
NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right or remedy under this Agreement shall be construed as a waiver of any future or other exercise of such right or remedy by Licensor.
GOVERNING LAW: This Agreement shall be construed and enforced in accordance with the laws of Germany without reference to conflict of laws principles.  You consent to the personal jurisdiction of the courts of this country and waive their rights to venue outside of Germany.
ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and entire agreement between Licensee and Licensor as to the matter set forth herein and supersedes any previous agreements, understandings, and arrangements between the parties relating hereto.
*     THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
import logging
import os
import time

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


def time_it(fn):
    def wrapper(*args, **kwargs):
        start = time.time()
        ret = fn(*args, **kwargs)
        end = time.time()
        logging.info(f'Time: {end - start}')
        return ret

    return wrapper


class TrainDataset(Dataset):

    def _get_adj_mat(self):
        a_mat = sparse.dok_matrix((self.nentity, self.nentity))
        for (h, _, t) in self.triples:
            a_mat[t, h] = 1
            a_mat[h, t] = 1

        a_mat = a_mat.tocsr()
        return a_mat

    @time_it
    def build_k_hop(self, k_hop, dataset_name):
        if k_hop == 0:
            return None

        save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw0.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat

        _a_mat = self._get_adj_mat()
        _k_mat = _a_mat ** (k_hop - 1)
        k_mat = _k_mat * _a_mat + _k_mat

        sparse.save_npz(save_path, k_mat)

        return k_mat

    @time_it
    def build_k_rw(self, n_rw, k_hop, dataset_name):
        """
        Returns:
            k_mat: sparse |V| * |V| adjacency matrix
        """
        if n_rw == 0 or k_hop == 0:
            return None

        save_path = f'cached_matrices/matrix_{dataset_name}_k{k_hop}_nrw{n_rw}.npz'

        if os.path.exists(save_path):
            logging.info(f'Using cached matrix: {save_path}')
            k_mat = sparse.load_npz(save_path)
            return k_mat

        a_mat = self._get_adj_mat()
        k_mat = sparse.dok_matrix((self.nentity, self.nentity))

        randomly_sampled = 0

        for i in range(0, self.nentity):
            if i%100==0:
                print('now i = ', i)
            neighbors = a_mat[i]
            if len(neighbors.indices) == 0:
                randomly_sampled += 1
                walker = np.random.randint(self.nentity, size=n_rw)
                k_mat[i, walker] = 1
            else:
                for _ in range(0, n_rw):
                    walker = i
                    for _ in range(0, k_hop):
                        idx = np.random.randint(len(neighbors.indices))
                        walker = neighbors.indices[idx]
                        neighbors = a_mat[walker]
                    k_mat[i, walker] += 1
        logging.info(f'randomly_sampled: {randomly_sampled}')
        k_mat = k_mat.tocsr()

        sparse.save_npz(save_path, k_mat)

        return k_mat

    def __init__(self, triples, nentity, nrelation, negative_sample_size, mode, k_hop, n_rw, dsn,
                 if_adv_neg=None, best_model=None):
        self.len = len(triples)
        self.triples = triples
        self.triple_set = set(triples)
        self.nentity = nentity
        self.nrelation = nrelation

        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.count = self.count_frequency(triples)
        self.true_head, self.true_tail = self.get_true_head_and_tail(self.triples)
        self.dsn = dsn.split('/')[1]  # dataset name

        self.if_adv_neg = if_adv_neg

        if n_rw == 0:
            self.k_neighbors = self.build_k_hop(k_hop, dataset_name=self.dsn)
        else:
            self.k_neighbors = self.build_k_rw(n_rw=n_rw, k_hop=k_hop, dataset_name=self.dsn)

        if if_adv_neg is not None and mode == 'tail-batch':  # original negative sampling via best model
            __batch = 1000
            self.adv_neg_samples = np.zeros([self.len, self.negative_sample_size], dtype=np.int64)
            best_model.eval()
            with torch.no_grad():
                idx_batch = 0
                while idx_batch < self.len:
                    _batch = min(__batch, self.len - idx_batch)
                    pos_samples = triples[idx_batch:idx_batch + _batch]
                    pos_samples = torch.LongTensor(pos_samples).cuda()
                    score = best_model((pos_samples, pos_samples), mode, if_CE=3)
                    argsort = torch.argsort(score, dim=1, descending=True)
                    positive_arg = pos_samples[:, 2]
                    _adv_neg_samples = []
                    for i in range(_batch):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1
                        if ranking.size(0) - 1 > self.negative_sample_size:
                            self.adv_neg_samples[idx_batch + i] = \
                                argsort[i, :self.negative_sample_size].cpu().numpy()
                        else:
                            _adv_neg_samples = argsort[i, :ranking.size(0) - 1].cpu().numpy()
                            __adv_neg_samples = np.random.randint(self.nentity,
                                                                  size=self.negative_sample_size-ranking.size(0)+1)
                            _adv_neg_samples = np.concatenate([_adv_neg_samples, __adv_neg_samples], axis=-1)
                            self.adv_neg_samples[idx_batch + i] = _adv_neg_samples

                    idx_batch += _batch

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        positive_sample = self.triples[idx]

        head, relation, tail = positive_sample

        subsampling_weight = self.count[(head, relation)] + self.count[(tail, -relation - 1)]
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        negative_sample_list = []
        negative_sample_size = 0

        k_hop_flag = True
        while negative_sample_size < self.negative_sample_size:
            if self.k_neighbors is not None and k_hop_flag:
                if self.mode == 'head-batch':
                    khop = self.k_neighbors[tail].indices
                elif self.mode == 'tail-batch':
                    khop = self.k_neighbors[head].indices
                else:
                    raise ValueError('Training batch mode %s not supported' % self.mode)
                negative_sample = khop[np.random.randint(len(khop), size=self.negative_sample_size * 2)].astype(
                    np.int64)
            else:
                negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size * 2)
            if self.mode == 'head-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_head[(relation, tail)],
                    assume_unique=True,
                    invert=True
                )
            elif self.mode == 'tail-batch':
                mask = np.in1d(
                    negative_sample,
                    self.true_tail[(head, relation)],
                    assume_unique=True,
                    invert=True
                )
            else:
                raise ValueError('Training batch mode %s not supported' % self.mode)
            negative_sample = negative_sample[mask]
            negative_sample_list.append(negative_sample)
            if negative_sample.size == 0:
                k_hop_flag = False
            negative_sample_size += negative_sample.size

        negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]

        if self.if_adv_neg is not None and self.mode == 'tail-batch':  # original negative sampling via best model
            negative_sample = torch.from_numpy(self.adv_neg_samples[idx])
        else:
            negative_sample = torch.from_numpy(negative_sample)

        positive_sample = torch.LongTensor(positive_sample)

        return positive_sample, negative_sample, subsampling_weight, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, subsample_weight, mode

    @staticmethod
    def count_frequency(triples, start=4):
        '''
        Get frequency of a partial triple like (head, relation) or (relation, tail)
        The frequency will be used for subsampling like word2vec
        '''
        count = {}
        for head, relation, tail in triples:
            if (head, relation) not in count:
                count[(head, relation)] = start
            else:
                count[(head, relation)] += 1

            if (tail, -relation - 1) not in count:
                count[(tail, -relation - 1)] = start
            else:
                count[(tail, -relation - 1)] += 1
        return count

    @staticmethod
    def get_true_head_and_tail(triples):
        '''
        Build a dictionary of true triples that will
        be used to filter these true triples for negative sampling
        '''

        true_head = {}
        true_tail = {}

        for head, relation, tail in triples:
            if (head, relation) not in true_tail:
                true_tail[(head, relation)] = []
            true_tail[(head, relation)].append(tail)
            if (relation, tail) not in true_head:
                true_head[(relation, tail)] = []
            true_head[(relation, tail)].append(head)

        for relation, tail in true_head:
            true_head[(relation, tail)] = np.array(list(set(true_head[(relation, tail)])))
        for head, relation in true_tail:
            true_tail[(head, relation)] = np.array(list(set(true_tail[(head, relation)])))

        return true_head, true_tail


class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, nentity, nrelation, mode):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.nentity = nentity
        self.nrelation = nrelation
        self.mode = mode

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        if self.mode == 'head-batch':
            tmp = [(0, rand_head) if (rand_head, relation, tail) not in self.triple_set
                   else (-1, head) for rand_head in range(self.nentity)]
            tmp[head] = (0, head)
        elif self.mode == 'tail-batch':
            tmp = [(0, rand_tail) if (head, relation, rand_tail) not in self.triple_set
                   else (-1, tail) for rand_tail in range(self.nentity)]
            tmp[tail] = (0, tail)
        else:
            raise ValueError('negative batch mode %s not supported' % self.mode)

        tmp = torch.LongTensor(tmp)
        filter_bias = tmp[:, 0].float()
        negative_sample = tmp[:, 1]

        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, filter_bias, self.mode

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        filter_bias = torch.stack([_[2] for _ in data], dim=0)
        mode = data[0][3]
        return positive_sample, negative_sample, filter_bias, mode


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
