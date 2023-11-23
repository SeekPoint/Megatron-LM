
import glob
import json
import os
import time

from torch.utils.data import Dataset

from megatron import print_rank_0
from tasks.data_utils import build_sample
from tasks.data_utils import build_tokens_types_paddings_from_ids
from tasks.data_utils import clean_text


NUM_CHOICES = 4
MAX_QA_LENGTH = 128
from pydebug import gd, infoTensor

class RaceDataset(Dataset):

    def __init__(self, dataset_name, datapaths, tokenizer, max_seq_length,
                 max_qa_length=MAX_QA_LENGTH):

        self.dataset_name = dataset_name
        gd.debuginfo(prj="mt", info=f' > building RACE dataset for {self.dataset_name}')

        string = '  > paths:'
        for path in datapaths:
            string += ' ' + path
        gd.debuginfo(prj="mt", info=string)

        self.samples = []
        for datapath in datapaths:
            self.samples.extend(process_single_datapath(datapath, tokenizer,
                                                        max_qa_length,
                                                        max_seq_length))

        gd.debuginfo(prj="mt", info=f'  >> total number of samples: {len(self.samples)}')

        # This indicates that each "sample" has multiple samples that
        # will collapse into batch dimension
        self.sample_multiplier = NUM_CHOICES

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def process_single_datapath(datapath, tokenizer, max_qa_length, max_seq_length):
    """Read in RACE files, combine, clean-up, tokenize, and convert to
    samples."""

    gd.debuginfo(prj="mt", info=f'   > working on {datapath}')
    start_time = time.time()

    # Get list of files.
    filenames = glob.glob(os.path.join(datapath, '*.txt'))

    samples = []
    num_docs = 0
    num_questions = 0
    num_samples = 0
    # Load all the files
    for filename in filenames:
        with open(filename, 'r') as f:
            for line in f:
                data = json.loads(line)
                num_docs += 1

                context = data["article"]
                questions = data["questions"]
                choices = data["options"]
                answers = data["answers"]
                # Check the length.
                assert len(questions) == len(answers)
                assert len(questions) == len(choices)

                # Context: clean up and convert to ids.
                context = clean_text(context)
                context_ids = tokenizer.tokenize(context)

                # Loop over questions.
                for qi, question in enumerate(questions):
                    num_questions += 1
                    # Label.
                    label = ord(answers[qi]) - ord("A")
                    assert label >= 0
                    assert label < NUM_CHOICES
                    assert len(choices[qi]) == NUM_CHOICES

                    # For each question, build num-choices samples.
                    ids_list = []
                    types_list = []
                    paddings_list = []
                    for ci in range(NUM_CHOICES):
                        choice = choices[qi][ci]
                        # Merge with choice.
                        if "_" in question:
                            qa = question.replace("_", choice)
                        else:
                            qa = " ".join([question, choice])
                        # Clean QA.
                        qa = clean_text(qa)
                        # Tokenize.
                        qa_ids = tokenizer.tokenize(qa)
                        # Trim if needed.
                        if len(qa_ids) > max_qa_length:
                            qa_ids = qa_ids[0:max_qa_length]

                        # Build the sample.
                        ids, types, paddings \
                            = build_tokens_types_paddings_from_ids(
                                qa_ids, context_ids, max_seq_length,
                                tokenizer.cls, tokenizer.sep, tokenizer.pad)

                        ids_list.append(ids)
                        types_list.append(types)
                        paddings_list.append(paddings)

                    # Convert to numpy and add to samples
                    samples.append(build_sample(ids_list, types_list,
                                                paddings_list, label,
                                                num_samples))
                    num_samples += 1

    elapsed_time = time.time() - start_time
    gd.debuginfo(prj="mt", info=f'> processed {num_docs} document, '
                                f'{num_questions} questions, '
                                f'and {num_samples} samples '
                                f'in {elapsed_time:.2f} seconds')

    return samples
