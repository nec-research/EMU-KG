"""
*
*     SOFTWARE NAME
*
*        File:  model.py
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

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import TestDataset


rng = np.random.default_rng(2022)

class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma,
                 double_entity_embedding=False, double_relation_embedding=False):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        self.entity_dim = hidden_dim * 2 if double_entity_embedding else hidden_dim
        self.relation_dim = hidden_dim * 2 if double_relation_embedding else hidden_dim

        self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
        nn.init.uniform_(
            tensor=self.entity_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        self.relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
        nn.init.uniform_(
            tensor=self.relation_embedding,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if model_name == 'TransD':
            self.proj_entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(
                tensor=self.proj_entity_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

            self.proj_relation_embedding = nn.Parameter(torch.zeros(nrelation, self.relation_dim))
            nn.init.uniform_(
                tensor=self.proj_relation_embedding,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        if model_name == 'pRotatE':
            self.modulus = nn.Parameter(torch.Tensor([[0.5 * self.embedding_range.item()]]))

        # Do not forget to modify this line when you add a new model in the "forward" function
        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'pRotatE', 'TransD']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE' and (not double_entity_embedding or double_relation_embedding):
            raise ValueError('RotatE should use --double_entity_embedding')

        if model_name == 'ComplEx' and (not double_entity_embedding or not double_relation_embedding):
            raise ValueError('ComplEx should use --double_entity_embedding and --double_relation_embedding')


    def forward(self, sample, mode='single', if_CE=False, if_Mutup=False, if_Mixup=False, if_OnlyLS=False):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements 
        in their triple ((head, relation) or (relation, tail)).
        '''

        mix_label = None

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=sample[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=sample[:, 2]
            ).unsqueeze(1)

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                head_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=sample[:, 0]
                ).view(batch_size, negative_sample_size, -1)

                relation_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=sample[:, 1]
                ).unsqueeze(1)

                tail_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=sample[:, 2]
                ).unsqueeze(1)
            else:
                head_t = None
                relation_t = None
                tail_t = None

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)

            if if_CE:
                if if_CE==3:  # 1VsAll
                    head = self.entity_embedding
                else:
                    if if_CE==1:  # uniform sampling
                        _index = np.random.randint(0, self.nentity, size=[batch_size * negative_sample_size])
                        _index = torch.from_numpy(_index).cuda()
                    else:  # negative sampling
                        _index = head_part.view(-1)
                    # negative samples
                    head = torch.index_select(
                        self.entity_embedding,
                        dim=0,
                        index=_index
                    ).view(batch_size, negative_sample_size, -1)
                    # true sample
                    head_t = torch.index_select(
                        self.entity_embedding,
                        dim=0,
                        index=tail_part[:, 0]
                    ).view(batch_size, 1, -1)
                    head = torch.cat((head_t, head), dim=1) # batch, 1+neg, ch
                    if if_Mutup:
                        if if_Mixup:  # Mixup
                            _mut_ratio = rng.beta(2, 1, size=(head.size(0), head.size(1), 1)).astype(np.float32)
                            mix_label = _mut_ratio
                        else:  # Mutup
                            _mut_ratio = rng.choice([0, 1], size=head.size(),
                                                    p=[1. - if_Mutup, if_Mutup]).astype(np.float32)
                        _mut_ratio[:, 0] = 1.  # positive sample
                        _mut_ratio = torch.from_numpy(_mut_ratio).cuda()
                        if not if_OnlyLS:
                            head = head * _mut_ratio + head[:, 0, :][:, None, :] * (1. - _mut_ratio)
            else:
                # negative samples
                _index = head_part.view(-1)
                head = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=_index
                ).view(batch_size, negative_sample_size, -1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=tail_part[:, 1]
            ).unsqueeze(1)

            tail = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=tail_part[:, 2]
            ).unsqueeze(1)

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                head_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=head_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)

                relation_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=tail_part[:, 1]
                ).unsqueeze(1)

                tail_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=tail_part[:, 2]
                ).unsqueeze(1)
            else:
                head_t = None
                relation_t = None
                tail_t = None

        elif mode == 'tail-batch':
            head_part, tail_part = sample  # positive, negative
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)

            head = torch.index_select(
                self.entity_embedding,
                dim=0,
                index=head_part[:, 0]
            ).unsqueeze(1)

            relation = torch.index_select(
                self.relation_embedding,
                dim=0,
                index=head_part[:, 1]
            ).unsqueeze(1)

            if if_CE:
                if if_CE==3:  # 1VsAll
                    tail = self.entity_embedding
                else:
                    if if_CE==1:  # uniform sampling
                        _index = np.random.randint(0, self.nentity, size=[batch_size * negative_sample_size])
                        _index = torch.from_numpy(_index).cuda()
                    else:  # negative sampling
                        _index = tail_part.view(-1)
                    # negative samples
                    tail = torch.index_select(
                        self.entity_embedding,
                        dim=0,
                        index=_index
                    ).view(batch_size, negative_sample_size, -1)
                    # true samples
                    tail_t = torch.index_select(
                        self.entity_embedding,
                        dim=0,
                        index=head_part[:, 2]
                    ).view(batch_size, 1, -1)
                    tail = torch.cat((tail_t, tail), dim=1) # batch, 1+neg, ch
                    if if_Mutup:
                        if if_Mixup:  # Mixup
                            _mut_ratio = rng.beta(2, 1, size=(tail.size(0), tail.size(1), 1)).astype(np.float32)
                            mix_label = _mut_ratio
                        else:
                            _mut_ratio = rng.choice([0, 1], size=tail.size(),
                                                    p=[1. - if_Mutup, if_Mutup]).astype(np.float32)
                        _mut_ratio[:, 0] = 1.  # positive sample
                        _mut_ratio = torch.from_numpy(_mut_ratio).cuda()
                        if not if_OnlyLS:
                            tail = tail * _mut_ratio + tail[:, 0, :][:, None, :] * (1. - _mut_ratio)
            else:
                # negative samples
                _index = tail_part.view(-1)
                tail = torch.index_select(
                    self.entity_embedding,
                    dim=0,
                    index=_index
                ).view(batch_size, negative_sample_size, -1)

            if hasattr(self, 'proj_entity_embedding') and hasattr(self, 'proj_relation_embedding'):
                head_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=head_part[:, 0]
                ).unsqueeze(1)

                relation_t = torch.index_select(
                    self.proj_relation_embedding,
                    dim=0,
                    index=head_part[:, 1]
                ).unsqueeze(1)

                tail_t = torch.index_select(
                    self.proj_entity_embedding,
                    dim=0,
                    index=tail_part.view(-1)
                ).view(batch_size, negative_sample_size, -1)
            else:
                head_t = None
                relation_t = None
                tail_t = None
        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'TransD': self.TransD,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'pRotatE': self.pRotatE
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, head_t, tail_t, relation_t, mode, if_CE)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score, mix_label

    def TransE(self, head, relation, tail, head_t, tail_t, relation_t, mode, if_CE):
        if if_CE==3:
            raise NotImplementedError
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def _transfer(self, e, e_t, r_t):
        return F.normalize(e + (e * e_t).sum(dim=1, keepdim=True) * r_t, 2, -1)

    def TransD(self, head, relation, tail, head_t, tail_t, relation_t, mode, if_CE):
        head_proj = self._transfer(head, head_t, relation_t)
        tail_proj = self._transfer(tail, tail_t, relation_t)

        if mode == 'head-batch':
            score = head_proj + (relation - tail_proj)
        else:
            score = (head_proj + relation) - tail_proj

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, head_t, tail_t, relation_t, mode, if_CE):
        if if_CE == 1 or if_CE==2:  # pure CE for uniform/SAN negative sampling
            if mode == 'head-batch':
                RT = relation * tail
                # score = torch.squeeze(head.transpose(2, 1) @ RT)  # (b,neg,c) @ (b,c,neg)
                score = torch.einsum('bnc,bnc->bn', head, RT)
            else:
                HT = head * relation
                # score = torch.squeeze(HT @ tail.transpose(2, 1))  # (b,neg,c) @ (b,c,neg)
                score = torch.einsum('bnc,bnc->bn', HT, tail)
        elif if_CE == 3:  # 1vsAll
            if mode == 'head-batch':
                RT = torch.squeeze(relation * tail)
                # score = torch.squeeze(head.transpose(2, 1) @ RT)  # (b,neg,c) @ (b,c,neg)
                score = torch.squeeze(torch.einsum('dc,bc->bd', head, RT))
            else:
                HT = torch.squeeze(head * relation)
                # score = torch.squeeze(HT @ tail.transpose(2, 1))  # (b,neg,c) @ (b,c,neg)
                score = torch.squeeze(torch.einsum('bc,dc->bd', HT, tail))
        else:
            if mode == 'head-batch':
                score = head * (relation * tail)
            else:
                score = (head * relation) * tail

            score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, head_t, tail_t, relation_t, mode, if_CE):
        if if_CE==3 and len(head.size())==2:
            re_head, im_head = torch.chunk(head.unsqueeze(1), 2, dim=2)
            re_head, im_head = torch.squeeze(re_head), torch.squeeze(im_head)
        else:
            re_head, im_head = torch.chunk(head, 2, dim=2)
        if if_CE==3 and len(relation.size())==2:
            re_relation, im_relation = torch.chunk(relation.unsqueeze(1), 2, dim=2)
            re_relation, im_relation = torch.squeeze(re_relation), torch.squeeze(im_relation)
        else:
            re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        if if_CE==3 and len(tail.size())==2:
            re_tail, im_tail = torch.chunk(tail.unsqueeze(1), 2, dim=2)
            re_tail, im_tail = torch.squeeze(re_tail), torch.squeeze(im_tail)
        else:
            re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if if_CE == 1 or if_CE==2:  # pure CE for uniform/SAN negative sampling
            if mode == 'head-batch':
                RT_r = re_relation * re_tail + im_relation * im_tail
                RT_i = re_relation * im_tail - im_relation * re_tail
                # score = torch.squeeze(head.transpose(2, 1) @ RT)  # (b,neg,c) @ (b,c,neg)
                re_score = torch.einsum('bnc,bnc->bn', re_head, RT_r)
                im_score = torch.einsum('bnc,bnc->bn', im_head, RT_i)
            else:
                HR_r = re_head * re_relation - im_head * im_relation
                HR_i = re_head * im_relation + im_head * re_relation
                re_score = torch.einsum('bnc,bnc->bn', HR_r, re_tail)
                im_score = torch.einsum('bnc,bnc->bn', HR_i, im_tail)

            score = re_score + im_score
        elif if_CE == 3:  # 1vsAll
            if mode == 'head-batch':
                RT_r = torch.squeeze(re_relation * re_tail + im_relation * im_tail)
                RT_i = torch.squeeze(re_relation * im_tail - im_relation * re_tail)
                # score = torch.squeeze(head.transpose(2, 1) @ RT)  # (b,neg,c) @ (b,c,neg)
                re_score = torch.squeeze(torch.einsum('dc,bc->bd', re_head, RT_r))
                im_score = torch.squeeze(torch.einsum('dc,bc->bd', im_head, RT_i))
            else:
                HR_r = torch.squeeze(re_head * re_relation - im_head * im_relation)
                HR_i = torch.squeeze(re_head * im_relation + im_head * re_relation)
                # score = torch.squeeze(HT @ tail.transpose(2, 1))  # (b,neg,c) @ (b,c,neg)
                re_score = torch.squeeze(torch.einsum('bc,dc->bd', HR_r, re_tail))
                im_score = torch.squeeze(torch.einsum('bc,dc->bd', HR_i, im_tail))
            score = re_score + im_score
        else:
            if mode == 'head-batch':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                score = re_head * re_score + im_head * im_score
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                score = re_score * re_tail + im_score * im_tail

            score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, head_t, tail_t, relation_t, mode, if_CE):
        # preprocess
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / np.pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)
        if if_CE == 3:  # 1vsAll
            raise NotImplementedError
        else:
            if mode == 'head-batch':
                re_score = re_relation * re_tail + im_relation * im_tail
                im_score = re_relation * im_tail - im_relation * re_tail
                re_score = re_score - re_head
                im_score = im_score - im_head
            else:
                re_score = re_head * re_relation - im_head * im_relation
                im_score = re_head * im_relation + im_head * re_relation
                re_score = re_score - re_tail
                im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)
        score = self.gamma.item() - score.sum(dim=2)

        return score

    def pRotatE(self, head, relation, tail, head_t, tail_t, relation_t, mode, if_CE):
        pi = 3.14159262358979323846

        # Make phases of entities and relations uniformly distributed in [-pi, pi]

        phase_head = head / (self.embedding_range.item() / pi)
        phase_relation = relation / (self.embedding_range.item() / pi)
        phase_tail = tail / (self.embedding_range.item() / pi)

        if mode == 'head-batch':
            score = phase_head + (phase_relation - phase_tail)
        else:
            score = (phase_head + phase_relation) - phase_tail

        score = torch.sin(score)
        score = torch.abs(score)

        score = self.gamma.item() - score.sum(dim=2) * self.modulus
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()

        optimizer.zero_grad()

        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        if args.if_CE:
            pred, _ = model((positive_sample, negative_sample), mode=mode, if_CE=args.if_CE)
            if args.if_CE==3:  # 1vsAll
                if mode == 'head-batch':
                    label = positive_sample[:, 0]
                elif mode == 'tail-batch':
                    label = positive_sample[:, 2]
            else:
                label = torch.zeros_like(pred, device=torch.device('cuda'))
                label[:, 0] = 1.
            loss = nn.CrossEntropyLoss(reduction='mean')(pred, label)

            if args.if_Mutup:
                pred, mix_label = model((positive_sample, negative_sample), mode=mode, if_CE=args.if_CE,
                                        if_Mutup=args.if_Mutup, if_Mixup=args.if_Mixup, if_OnlyLS=args.if_OnlyLS)
                if args.neg_label is not None:
                    neg_label = args.neg_label
                else:
                    if args.if_Mutup:
                        neg_label = 1. - mix_label
                    else:
                        neg_label = 1. - args.if_Mutup
                label = neg_label * torch.ones_like(pred, device=torch.device('cuda'))
                label[:, 0] = 1.
                loss = nn.CrossEntropyLoss(reduction='mean')(pred, label) / negative_sample.size(1) \
                       + args.CE_coef * loss

            positive_sample_loss = loss
            negative_sample_loss = loss

            if args.negative_adversarial_sampling:
                negative_score, _ = model((positive_sample, negative_sample), mode=mode)
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)

                if args.uni_weight:
                    negative_sample_loss = - negative_score.mean()
                else:
                    negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

                loss = loss + negative_sample_loss

        else:
            negative_score, _ = model((positive_sample, negative_sample), mode=mode)

            if args.negative_adversarial_sampling:
                # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
                negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                                  * F.logsigmoid(-negative_score)).sum(dim=1)
            else:
                negative_score = F.logsigmoid(-negative_score).mean(dim=1)

            positive_score, _ = model(positive_sample)

            positive_score = F.logsigmoid(positive_score).squeeze(dim=1)

            if args.uni_weight:
                positive_sample_loss = - positive_score.mean()
                negative_sample_loss = - negative_score.mean()
            else:
                positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
                negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

            loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                model.entity_embedding.norm(p=3) ** 3 +
                model.relation_embedding.norm(p=3) ** 3
                #model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, all_true_triples, args):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Otherwise use standard (filtered) MRR, MR, HITS@1, HITS@3, and HITS@10 metrics
        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'head-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                all_true_triples,
                args.nentity,
                args.nrelation,
                'tail-batch'
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        logs = []

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, filter_bias, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()
                        filter_bias = filter_bias.cuda()

                    batch_size = positive_sample.size(0)

                    score, _ = model((positive_sample, negative_sample), mode)
                    score += filter_bias

                    # Explicitly sort all the entities to ensure that there is no test exposure bias
                    argsort = torch.argsort(score, dim=1, descending=True)

                    if mode == 'head-batch':
                        positive_arg = positive_sample[:, 0]
                    elif mode == 'tail-batch':
                        positive_arg = positive_sample[:, 2]
                    else:
                        raise ValueError('mode %s not supported' % mode)

                    for i in range(batch_size):
                        # Notice that argsort is not ranking
                        ranking = (argsort[i, :] == positive_arg[i]).nonzero()
                        assert ranking.size(0) == 1

                        # ranking + 1 is the true ranking used in evaluation metrics
                        ranking = 1 + ranking.item()
                        logs.append({
                            'MRR': 1.0 / ranking,
                            'MR': float(ranking),
                            'HITS@1': 1.0 if ranking <= 1 else 0.0,
                            'HITS@3': 1.0 if ranking <= 3 else 0.0,
                            'HITS@10': 1.0 if ranking <= 10 else 0.0,
                        })

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

        metrics = {}
        for metric in logs[0].keys():
            metrics[metric] = sum([log[metric] for log in logs]) / len(logs)

        return metrics
