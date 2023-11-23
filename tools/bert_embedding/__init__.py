# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

from .embed import BertEmbedder, DiskDataParallelBertEmbedder
from pydebug import gd, infoTensor
gd.debuginfo(prj="mt")