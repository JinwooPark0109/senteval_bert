# cusotomized SentEval for contrastive learning

STS task (no finetuning)

python examples/bert_sts.py --task ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', 'SICKRelatedness' ]

'all' for run all task


transfer learning task (with finetuning)

python examples/bert_transfer_learning.py --task ['MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'TREC', 'MRPC']
