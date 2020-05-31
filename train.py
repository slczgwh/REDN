#!/usr/bin/env python
import logging
import os
import sys
from allennlp.commands import main  # pylint: disable=wrong-import-position

from redn import models, training
import redn.data.dataset_readers
import redn.models
import redn.models.redn
import redn.predictors

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.INFO)
logging.basicConfig(filename='allennlp.data.token_indexers.wordpiece_indexer', level=logging.ERROR)

if __name__ == "__main__":
    main(prog="python run.py")
