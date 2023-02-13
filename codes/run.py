"""
*
*     SOFTWARE NAME
*
*        File:  run.py
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


import argparse
import json
import logging
import os
import random
import pickle

import numpy as np

import torch
from torch.utils.data import DataLoader

from model import KGEModel
from dataloader import TrainDataset
from dataloader import BidirectionalOneShotIterator


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--evaluate_train', action='store_true', help='Evaluate on training data')

    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--model', default='TransE', type=str)
    parser.add_argument('-de', '--double_entity_embedding', action='store_true')
    parser.add_argument('-dr', '--double_relation_embedding', action='store_true')

    parser.add_argument('-n', '--negative_sample_size', default=128, type=int)
    parser.add_argument('-d', '--hidden_dim', default=500, type=int)
    parser.add_argument('-g', '--gamma', default=12.0, type=float)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('-a', '--adversarial_temperature', default=1.0, type=float)
    parser.add_argument('-khop', '--negative_k_hop_sampling', default=0, type=int)
    parser.add_argument('-nrw', '--negative_n_random_walks', default=0, type=int)
    parser.add_argument('-b', '--batch_size', default=1024, type=int)
    parser.add_argument('-r', '--regularization', default=0.0, type=float)
    parser.add_argument('--test_batch_size', default=4, type=int, help='valid/test batch size')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int)
    parser.add_argument('-init', '--init_checkpoint', default=None, type=str)
    parser.add_argument('-save', '--save_path', default=None, type=str)
    parser.add_argument('--max_steps', default=100000, type=int)
    parser.add_argument('--warm_up_steps', default=None, type=int)

    parser.add_argument('--save_checkpoint_steps', default=10000, type=int)
    parser.add_argument('--valid_steps', default=10000, type=int)
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('-if_CE', type=int, default=0)
    parser.add_argument('-if_Mutup', type=float, default=0.)
    parser.add_argument('-if_Mixup', type=float, default=0.)
    parser.add_argument('-if_OnlyLS', type=float, default=0.)
    parser.add_argument('-CE_coef', type=float, default=0.)
    parser.add_argument('-neg_label', type=float, default=None)
    parser.add_argument('-if_Adagrad', type=int, default=0)
    parser.add_argument('-if_adv_neg', type=int, default=None)
    parser.add_argument('-best_model_name', type=str, default=None)
    parser.add_argument('-if_Optuna', type=int, default=0)

    return parser.parse_args(args)


def override_config(args):
    '''
    Override model and data configuration
    '''

    with open(os.path.join(args.init_checkpoint, 'config.json'), 'r') as fjson:
        argparse_dict = json.load(fjson)

    if args.data_path is None:
        args.data_path = argparse_dict['data_path']
    args.model = argparse_dict['model']
    args.double_entity_embedding = argparse_dict['double_entity_embedding']
    args.double_relation_embedding = argparse_dict['double_relation_embedding']
    args.hidden_dim = argparse_dict['hidden_dim']
    args.test_batch_size = argparse_dict['test_batch_size']


def save_model(model, optimizer, save_variable_list, args, if_Best=False):
    '''
    Save the parameters of the model and the optimizer,
    as well as some other variables such as step and learning_rate
    '''

    argparse_dict = vars(args)
    with open(os.path.join(args.save_path, 'config.json'), 'w') as fjson:
        json.dump(argparse_dict, fjson)

    torch.save({
        **save_variable_list,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()},
        os.path.join(args.save_path, 'checkpoint')
    )

    entity_embedding = model.entity_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'entity_embedding'),
        entity_embedding
    )

    relation_embedding = model.relation_embedding.detach().cpu().numpy()
    np.save(
        os.path.join(args.save_path, 'relation_embedding'),
        relation_embedding
    )
    if if_Best:
        torch.save(model.state_dict(), os.path.join(args.save_path, 'best_model'))
        entity_embedding = model.entity_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'best_entity_embedding'),
            entity_embedding
        )

        relation_embedding = model.relation_embedding.detach().cpu().numpy()
        np.save(
            os.path.join(args.save_path, 'best_relation_embedding'),
            relation_embedding
        )


def read_triple(file_path, entity2id, relation2id):
    '''
    Read triples and map them into ids.
    '''
    triples = []
    with open(file_path) as fin:
        for line in fin:
            h, r, t = line.strip().split('\t')
            triples.append((entity2id[h], relation2id[r], entity2id[t]))
    return triples


def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''

    if args.do_train:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'train.log')
    else:
        log_file = os.path.join(args.save_path or args.init_checkpoint, 'test.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def log_metrics(mode, step, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('%s %s at step %d: %f' % (mode, metric, step, metrics[metric]))


def main(args,
         if_Mutup=None, neg_label=None, CE_coef=None):
    #  for Optuna
    if args.if_Optuna:
        if if_Mutup is not None:
            args.__setattr__('if_Mutup', if_Mutup)
        if neg_label is not None:
            args.__setattr__('neg_label', neg_label)
        if CE_coef is not None:
            args.__setattr__('CE_coef', CE_coef)


    if (not args.do_train) and (not args.do_valid) and (not args.do_test):
        raise ValueError('one of train/val/test mode must be choosed.')

    if args.init_checkpoint:
        override_config(args)
    elif args.data_path is None:
        raise ValueError('one of init_checkpoint/data_path must be choosed.')

    if args.do_train and args.save_path is None:
        raise ValueError('Where do you want to save your trained model?')

    if args.save_path and not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # Write logs to checkpoint and console
    set_logger(args)

    with open(os.path.join(args.data_path, 'entities.dict')) as fin:
        entity2id = dict()
        for line in fin:
            eid, entity = line.strip().split('\t')
            entity2id[entity] = int(eid)

    with open(os.path.join(args.data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split('\t')
            relation2id[relation] = int(rid)

    nentity = len(entity2id)
    nrelation = len(relation2id)

    args.nentity = nentity
    args.nrelation = nrelation

    logging.info('Model: %s' % args.model)
    logging.info('Data Path: %s' % args.data_path)
    logging.info('#entity: %d' % nentity)
    logging.info('#relation: %d' % nrelation)

    train_triples = read_triple(os.path.join(args.data_path, 'train.txt'), entity2id, relation2id)
    logging.info('#train: %d' % len(train_triples))
    valid_triples = read_triple(os.path.join(args.data_path, 'valid.txt'), entity2id, relation2id)
    logging.info('#valid: %d' % len(valid_triples))
    test_triples = read_triple(os.path.join(args.data_path, 'test.txt'), entity2id, relation2id)
    logging.info('#test: %d' % len(test_triples))

    # All true triples
    all_true_triples = train_triples + valid_triples + test_triples

    # past best model for generating adversarial samples
    if args.if_adv_neg is not None:
        best_kge_model = KGEModel(
            model_name=args.model,
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            double_entity_embedding=args.double_entity_embedding,
            double_relation_embedding=args.double_relation_embedding
        )
        best_kge_model.load_state_dict(torch.load(os.path.join(args.best_model_name)))
        if args.cuda:
            best_kge_model.cuda()
        print('prepare best modes: ', args.best_model_name)
    else:
        best_kge_model = None

    kge_model = KGEModel(
        model_name=args.model,
        nentity=nentity,
        nrelation=nrelation,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        double_entity_embedding=args.double_entity_embedding,
        double_relation_embedding=args.double_relation_embedding,
    )

    logging.info('Model Parameter Configuration:')
    for name, param in kge_model.named_parameters():
        logging.info('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))

    if args.cuda:
        kge_model = kge_model.cuda()
    if args.do_train:
        # Set training dataloader iterator
        train_dataloader_head = DataLoader(
            TrainDataset(train_triples,
                         nentity,
                         nrelation,
                         args.negative_sample_size,
                         'head-batch',
                         args.negative_k_hop_sampling,
                         args.negative_n_random_walks,
                         dsn=args.data_path),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_dataloader_tail = DataLoader(
            TrainDataset(train_triples,
                         nentity,
                         nrelation,
                         args.negative_sample_size,
                         'tail-batch',
                         args.negative_k_hop_sampling,
                         args.negative_n_random_walks,
                         dsn=args.data_path,
                         if_adv_neg=args.if_adv_neg,
                         best_model=best_kge_model),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=max(1, args.cpu_num//2),
            collate_fn=TrainDataset.collate_fn
        )
        train_iterator = BidirectionalOneShotIterator(train_dataloader_head, train_dataloader_tail)
        del(best_kge_model)

        # Set training configuration
        current_learning_rate = args.learning_rate
        if args.if_Adagrad:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
        else:
            optimizer = torch.optim.Adagrad(
                filter(lambda p: p.requires_grad, kge_model.parameters()),
                lr=current_learning_rate
            )
        if args.warm_up_steps:
            warm_up_steps = args.warm_up_steps
        else:
            warm_up_steps = args.max_steps // 2

    if args.init_checkpoint:
        # Restore model from checkpoint directory
        logging.info('Loading checkpoint %s...' % args.init_checkpoint)
        checkpoint = torch.load(os.path.join(args.init_checkpoint, 'checkpoint'))
        init_step = checkpoint['step']
        kge_model.load_state_dict(checkpoint['model_state_dict'])
        if args.do_train:
            current_learning_rate = checkpoint['current_learning_rate']
            warm_up_steps = checkpoint['warm_up_steps']
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        logging.info('Ramdomly Initializing %s Model...' % args.model)
        init_step = 0

    step = init_step

    logging.info('Start Training...')
    logging.info('init_step = %d' % init_step)
    logging.info('batch_size = %d' % args.batch_size)
    logging.info('negative_adversarial_sampling = %d' % args.negative_adversarial_sampling)
    logging.info('negative_k_hop_sampling = %d' % args.negative_k_hop_sampling)
    logging.info('negative_n_random_walks = %d' % args.negative_n_random_walks)
    logging.info('hidden_dim = %d' % args.hidden_dim)
    logging.info('gamma = %f' % args.gamma)
    logging.info('negative_adversarial_sampling = %s' % str(args.negative_adversarial_sampling))
    if args.negative_adversarial_sampling:
        logging.info('adversarial_temperature = %f' % args.adversarial_temperature)

    # Set valid dataloader as it would be evaluated during training

    if args.do_train:
        logging.info('learning_rate = %d' % current_learning_rate)

        training_logs = []

        best_score = -1.

        # Training Loop
        for step in range(init_step, args.max_steps):

            log = kge_model.train_step(kge_model, optimizer, train_iterator, args)

            training_logs.append(log)

            if step >= warm_up_steps:
                current_learning_rate = current_learning_rate / 10
                logging.info('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                optimizer = torch.optim.Adam(
                    filter(lambda p: p.requires_grad, kge_model.parameters()),
                    lr=current_learning_rate
                )
                warm_up_steps = warm_up_steps * 3

            if step % args.save_checkpoint_steps == 0:
                save_variable_list = {
                    'step': step,
                    'current_learning_rate': current_learning_rate,
                    'warm_up_steps': warm_up_steps
                }
                save_model(kge_model, optimizer, save_variable_list, args)

            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)
                log_metrics('Training average', step, metrics)
                training_logs = []

            if args.do_valid and step % args.valid_steps == 0:
                logging.info('Evaluating on Valid Dataset...')
                metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
                log_metrics('Valid', step, metrics)

                if best_score < metrics['MRR']:
                    print('best model!!', best_score, metrics['MRR'])
                    best_score = metrics['MRR']
                    save_model(kge_model, optimizer, save_variable_list, args, if_Best=True)

        save_variable_list = {
            'step': step,
            'current_learning_rate': current_learning_rate,
            'warm_up_steps': warm_up_steps
        }
        save_model(kge_model, optimizer, save_variable_list, args)

    print('loading best model...')
    kge_model.load_state_dict(torch.load(os.path.join(args.save_path, 'best_model')))
    kge_model.cuda()

    if args.do_valid:
        logging.info('Evaluating on Valid Dataset...')
        metrics = kge_model.test_step(kge_model, valid_triples, all_true_triples, args)
        log_metrics('Valid', step, metrics)

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        metrics = kge_model.test_step(kge_model, test_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)
        flnm = 'results/' + args.save_path.split('/')[1] + '.pickle'
        print('save result into pickle file...', flnm)
        with open(flnm, 'wb') as f:
            pickle.dump(metrics, f)

    if args.evaluate_train:
        logging.info('Evaluating on Training Dataset...')
        metrics = kge_model.test_step(kge_model, train_triples, all_true_triples, args)
        log_metrics('Test', step, metrics)

    if args.if_Optuna:
        return metrics['MRR']

if __name__ == '__main__':
    main(parse_args())
