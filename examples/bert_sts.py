# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import io
import numpy as np
import logging

import os
import torch
from transformers import BertTokenizer, BertModel
import argparse

# Set PATHs
HOME = os.path.join(os.path.expanduser('~'))
PATH_TO_SENTEVAL = os.path.join(HOME, "senteval_bert")
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL,"data")
PATH_TO_BERT = os.path.join(HOME,"bert-base-uncased/") # original bert
PATH_TO_BERT = os.path.join(HOME,"all-MiniLM-L6-v2") # sentence bert
PATH_TO_BERT = os.path.join(HOME,"unsup-simcse-bert-base-uncased") # simCSE(unsup)

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# prepare and batcher
def prepare(params, samples):
    # count all vocab of training dataset to make OoV token null embbeding.
    # sub-word tokenizning doesn't need this
    return

# no training at all
class sts_batcher:
    def __init__(self, args):
        self.model=wrapped_bert_encoder(**args)
        self.tokenizer = BertTokenizer.from_pretrained(args['model_path'])
    def __call__(self, params, batch):
        batch = [sent if sent != [] else ['.'] for sent in batch]
        encoding=[]
        for sent in batch:
            encoding.append(self.tokenizer.encode(sent, padding="max_length", return_tensors="pt", max_length=512))
        embeddings = self.model(torch.stack(encoding).squeeze(1))
        return embeddings

class wrapped_bert_encoder(torch.nn.Module):
    def __init__(self, model_path, pad_token=0, freeze_bert=False):
        super().__init__()
        self.net = BertModel.from_pretrained(model_path)
        if freeze_bert:
            for param in self.net.parameters():
                param.requires_grad=False
        self.pad_token = pad_token

    def make_input_dict(self, input):
        # due to line 53 at classifier.py, cast input to int again
        return {"input_ids":input.int(), "token_type_ids": torch.zeros_like(input, dtype=torch.int32) , "attention_mask": (input.int()!=self.pad_token).int()}

    def forward(self, input):
        input_dict=self.make_input_dict(input)
        output=self.net(**input_dict)
        return output['last_hidden_state'][:,0,:]  #(batch, 1, hidden_dim)


def get_params():
    # Set params for SentEval
    ret = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'batch_size': 128}
    ret['encoder_config'] ={
        "model_path": PATH_TO_BERT,
        "pad_token":BertTokenizer.from_pretrained(os.path.join(PATH_TO_BERT,"vocab.txt")).vocab['[PAD]'],
        "freeze_bert":True
    }
    return ret

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def main(args):
    params_senteval=get_params()
    batcher=sts_batcher(params_senteval['encoder_config'] )
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    '''
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    '''
    transfer_tasks = args.task
    if 'all' in transfer_tasks: transfer_tasks=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness' ]
    print(transfer_tasks)
    results = se.eval(transfer_tasks)
    print(results)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        nargs='+',
                        default=['STS12'],
                        choices=['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness' , 'all'], #SICKEntailment
                        help="example: python bert_sts.py --task STS12 STS13. all for all 7 tasks."
                        )
    return parser.parse_args()

if __name__ == "__main__":
    args=get_args()
    main(args)
