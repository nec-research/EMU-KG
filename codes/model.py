"""
*          NAME OF THE PROGRAM THIS FILE BELONGS TO
*
*   file: model.py
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


    def forward(self, sample, mode='single', if_CE=False, if_EMU=False, if_Mixup=False, if_OnlyLS=False):
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
                    if if_EMU:
                        if if_Mixup:  # Mixup
                            _mut_ratio = rng.beta(2, 1, size=(head.size(0), head.size(1), 1)).astype(np.float32)
                            mix_label = _mut_ratio
                        else:  # EMU
                            _mut_ratio = rng.choice([0, 1], size=head.size(),
                                                    p=[1. - if_EMU, if_EMU]).astype(np.float32)
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
                    if if_EMU:
                        if if_Mixup:  # Mixup
                            _mut_ratio = rng.beta(2, 1, size=(tail.size(0), tail.size(1), 1)).astype(np.float32)
                            mix_label = _mut_ratio
                        else:
                            _mut_ratio = rng.choice([0, 1], size=tail.size(),
                                                    p=[1. - if_EMU, if_EMU]).astype(np.float32)
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

            if args.if_EMU:
                pred, mix_label = model((positive_sample, negative_sample), mode=mode, if_CE=args.if_CE,
                                        if_EMU=args.if_EMU, if_Mixup=args.if_Mixup, if_OnlyLS=args.if_OnlyLS)
                if args.neg_label is not None:
                    neg_label = args.neg_label
                else:
                    if args.if_EMU:
                        neg_label = 1. - mix_label
                    else:
                        neg_label = 1. - args.if_EMU
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
