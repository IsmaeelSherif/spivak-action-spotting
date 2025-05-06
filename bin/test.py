#!/usr/bin/env python3

# Copyright 2023, Yahoo Inc.
# Licensed under the Apache License, Version 2.0.
# See the accompanying LICENSE file for terms.

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add it to the Python path if it's not already there
if parent_dir not in sys.path:
    print('added to path')
    sys.path.insert(0, parent_dir)

import logging

from spivak.application.argument_parser import get_args
from spivak.application.test_utils import test


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    test(get_args())
