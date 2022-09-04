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
from transformers import BertTokenizer, BertModel

# Set PATHs
PATH_TO_SENTEVAL = '../'
PATH_TO_DATA = '../data'
# PATH_TO_BERT =

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

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")  # this can replaced by custom BERT

    input_dict=tokenizer(batch, padding="max_length", return_tensors="pt")
    # input_ids:  (batch, seq len(currently padded to max len), dim(768) )
    # token_type_ids: (batch, seq len), 0 and 1 for next sentence prediction(NSP) of BERT pretraining. dummy at inference or finetuning
    # attention_mask: (batch, seq len), 0 for padding and 1 for non-padding token.

    output=mode(**input_dict)

    embeddings=output['last_hidden_state'][:,0,:]
    # last_hidden_state: (batch, seq len, dim)
    # pooler_output: (batch, dim) in case usning pooler(MLP on CLS output)

    return embeddings

# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
# nhid 0:  linear classifier
# else : 2 layer MLP with {nhid}-dim hidden layer
params_senteval['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                                 'tenacity': 3, 'epoch_size': 2}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results = se.eval(transfer_tasks)
    print(results)
