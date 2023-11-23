# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Main tasks functionality."""

from megatron import get_args, print_rank_0
from megatron.indexer import IndexBuilder
from tasks.orqa.evaluate_utils import ORQAEvaluator
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")
def main():
    """
    Main program
    """

    args = get_args()

    """
    Create a BlockData data structure by running an IndexBuilder over an
    ICT Dataset and then evaluate on NQ task
    """

    gd.debuginfo(prj="mt", info=f"Starting index builder!")

    index_builder = IndexBuilder()
    index_builder.build_and_save_index()
    gd.debuginfo(prj="mt", info=f"Build and save indices: done!")


    gd.debuginfo(prj="mt", info=f"Starting evaluations!")

    # Set up the model and evaluator
    evaluator = ORQAEvaluator()

    # Run evaluation
    if args.qa_data_dev is not None:
        evaluator.evaluate(args.qa_data_dev, "DEV")

    if args.qa_data_test is not None:
        evaluator.evaluate(args.qa_data_test, "TEST")

