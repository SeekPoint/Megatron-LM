# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""MNLI dataset."""

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")

LABELS = {'contradiction': 0, 'entailment': 1, 'neutral': 2}


class MNLIDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label='contradiction'):
        gd.debuginfo(prj='ds', info=f"C:{self.__class__.__name__}")
        self.test_label = test_label
        super().__init__('MNLI', name, datapaths,
                         tokenizer, max_seq_length)

    def process_samples_from_single_path(self, filename):
        """"Implement abstract method."""
        gd.debuginfo(prj="mt", info=f' > Processing {filename} ...')

        samples = []
        total = 0
        first = True
        is_test = False
        with open(filename, 'r') as f:
            for line in f:
                row = line.strip().split('\t')
                if first:
                    first = False
                    if len(row) == 10:
                        is_test = True
                        gd.debuginfo(prj="mt", info=f'   reading {row[0].strip()}, {row[8].strip()} '
                                                    f'and {row[9].strip()} columns and setting labels to {self.test_label}')
                    else:
                        gd.debuginfo(prj="mt", info=f'reading {row[0].strip()}, {row[8].strip()}, {row[9].strip()}, '
                                                    f'and {row[-1].strip()} columns ...')
                    continue

                text_a = clean_text(row[8].strip())
                text_b = clean_text(row[9].strip())
                unique_id = int(row[0].strip())
                label = row[-1].strip()
                if is_test:
                    label = self.test_label

                assert len(text_a) > 0
                assert len(text_b) > 0
                assert label in LABELS
                assert unique_id >= 0

                sample = {'text_a': text_a,
                          'text_b': text_b,
                          'label': LABELS[label],
                          'uid': unique_id}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    gd.debuginfo(prj="mt", info=f'  > processed {total} so far ...')

        gd.debuginfo(prj="mt", info=f' >> processed {len(samples)} samples.')
        return samples
