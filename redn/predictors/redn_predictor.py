from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from typing import List, Iterator, Optional
import json


@Predictor.register('redn')
class RednPredictor(Predictor):

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)

    def predict(self, tokens: List) -> JsonDict:
        return self.predict_json({"tokens": tokens})

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        return self._dataset_reader.text_to_instance(json_dict["token"])

    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        return json.loads(line)



