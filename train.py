#!/usr/bin/env python
import logging
import os
import sys
from allennlp.commands import main  # pylint: disable=wrong-import-position

from aarcnlp import models, training
import aarcnlp.data.dataset_readers
import aarcnlp.models
import aarcnlp.models.redn
import aarcnlp.predictors

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logging.basicConfig(filename='allennlp.data.token_indexers.wordpiece_indexer', level=logging.ERROR)

if __name__ == "__main__":
    main(prog="python run.py")
