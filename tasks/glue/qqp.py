# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""QQP dataset."""

from megatron import print_rank_0
from tasks.data_utils import clean_text
from .data import GLUEAbstractDataset
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")

LABELS = [0, 1]


class QQPDataset(GLUEAbstractDataset):

    def __init__(self, name, datapaths, tokenizer, max_seq_length,
                 test_label=0):
        self.test_label = test_label
        super().__init__('QQP', name, datapaths,
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
                    if len(row) == 3:
                        is_test = True
                        gd.debuginfo(prj="mt", info=f'   reading {}, {}, and {} columns and '
                                     'setting labels to {}'.format(
                                         row[0].strip(), row[1].strip(),
                                         row[2].strip(), self.test_label))
                    else:
                        assert len(row) == 6
                        gd.debuginfo(prj="mt", info=f'    reading {}, {}, {}, and {} columns'
                                     ' ...'.format(
                                         row[0].strip(), row[3].strip(),
                                         row[4].strip(), row[5].strip()))
                    continue

                if is_test:
                    assert len(row) == 3, 'expected length 3: {}'.format(row)
                    uid = int(row[0].strip())
                    text_a = clean_text(row[1].strip())
                    text_b = clean_text(row[2].strip())
                    label = self.test_label
                    assert len(text_a) > 0
                    assert len(text_b) > 0
                else:
                    if len(row) == 6:
                        uid = int(row[0].strip())
                        text_a = clean_text(row[3].strip())
                        text_b = clean_text(row[4].strip())
                        label = int(row[5].strip())
                    else:
                        gd.debuginfo(prj="mt", info=f'***WARNING*** index error, '
                                     'skipping: {}'.format(row))
                        continue
                    if len(text_a) == 0:
                        gd.debuginfo(prj="mt", info=f'***WARNING*** zero length a, '
                                     'skipping: {}'.format(row))
                        continue
                    if len(text_b) == 0:
                        gd.debuginfo(prj="mt", info=f'***WARNING*** zero length b, '
                                     'skipping: {}'.format(row))
                        continue
                assert label in LABELS
                assert uid >= 0

                sample = {'uid': uid,
                          'text_a': text_a,
                          'text_b': text_b,
                          'label': label}
                total += 1
                samples.append(sample)

                if total % 50000 == 0:
                    gd.debuginfo(prj="mt", info=f'  > processed {total} so far ...')

        gd.debuginfo(prj="mt", info=f' >> processed {len(samples)} samples.')
        return samples
