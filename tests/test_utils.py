import logging
import os
import tempfile
import warnings
from contextlib import contextmanager

import matplotlib
import numpy as np
import pytest
from pytest import raises

import aspire
from aspire import __version__
from aspire.utils import (
    LogFilterByCount,
    all_pairs,
    all_triplets,
    get_full_version,
    mem_based_cpu_suggestion,
    num_procs_suggestion,
    physical_core_cpu_suggestion,
    powerset,
    utest_tolerance,
    virtual_core_cpu_suggestion,
)
from aspire.utils.misc import (
    bump_3d,
    fuzzy_mask,
    gaussian_1d,
    gaussian_2d,
    gaussian_3d,
    grid_3d,
)

logger = logging.getLogger(__name__)


def test_log_filter_by_count(caplog):
    msg = "A is for ASCII"

    stream_handler = logger.handlers[0]
    print(logging.getLevelName(stream_handler.level))
    
    # Should log.
    logger.info(msg)
    assert msg in caplog.text
    caplog.clear()
