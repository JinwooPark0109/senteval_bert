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

# Set PATHs
HOME = os.path.join(os.path.expanduser('~'))
PATH_TO_SENTEVAL = os.path.join(HOME, "senteval_bert")
PATH_TO_DATA = os.path.join(PATH_TO_SENTEVAL,"data")
PATH_TO_BERT = os.path.join(HOME,"bert-base-uncased/")

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval

# prepare and batcher
def prepare(params, samples):
    # count all vocab of training dataset to make OoV token null embbeding.
    # sub-word tokenizning doesn't need this
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    embeddings = []
    for sent in batch:
        embeddings.append(params.tokenizer.encode(sent, padding="max_length", return_tensors="pt",max_length=512))
    embeddings = np.vstack(embeddings)
    return embeddings

class custom_CE_loss:
    def __init__(self):
        pass
    def __foward__(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)

class wrapped_bert_encoder(torch.nn.Module):
    def __init__(self, pad_token=0, freeze_bert=False):
        super().__init__()
        self.net = BertModel.from_pretrained(PATH_TO_BERT)
        if freeze_bert:
            for param in self.net.parameters():
                param.requires_grad=False
        self.pad_token = pad_token

    def make_input_dict(self, input):
        # due to line 53 at classifier.py, cast input to int again
        return {"input_ids":input.int(), "token_type_ids": torch.zeros_like(input, dtype=torch.int32) , "attention_mask": (input!=self.pad_token).int()}

    def forward(self, input):
        input_dict=self.make_input_dict(input)
        output=self.net(**input_dict)
        return output['last_hidden_state'][:,0,:]


def get_params():
    # Set params for SentEval
    ret = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    ret['tokenizer']=BertTokenizer.from_pretrained(os.path.join(PATH_TO_BERT,"vocab.txt"))
    encoder_config={'encoder_builder': wrapped_bert_encoder,
                    'encoder_args': {
                        "pad_token":ret['tokenizer'].vocab['[PAD]'],
                        "freeze_bert":False
                        }
                    }
    ret['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, # nhid 0 for linear classifier, else 2 layer MLP with {nhid}-dim hidden layer
                        'tenacity': 3, 'epoch_size': 2,
                        'bert_encoder': encoder_config,
                        'custom_loss': custom_CE_loss()
                        }
    return ret

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

def main():
    params_senteval=get_params()
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    '''
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    '''
    transfer_tasks = ['STS12', 'MR']
    results = se.eval(transfer_tasks)
    print(results)

if __name__ == "__main__":
    main()
