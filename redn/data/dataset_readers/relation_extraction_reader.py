import logging
import json

from typing import Dict, List, Union
from overrides import overrides

from allennlp.data import Token
from allennlp.data.fields import LabelField, SpanField
from allennlp.data.fields import TextField
from allennlp.data.fields import Field
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("relation_extraction_reader")
class RelationExtractionReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False) -> None:
        super().__init__(lazy=lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        lines = open(file_path).readlines()
        for line in lines:
            data = json.loads(line)

            yield self.text_to_instance(tokens=data["token"],
                                        h_span_start=data["h"]["pos"][0],
                                        h_span_end=data["h"]["pos"][1],
                                        t_span_start=data["t"]["pos"][0],
                                        t_span_end=data["t"]["pos"][1],
                                        rel_label=data["relation"] if "relation" in data else None
                                        )

    def text_to_instance(self,
                         tokens: List,
                         h_span_start,
                         h_span_end,
                         t_span_start,
                         t_span_end,
                         rel_label: Union[str, int] = None) -> Instance:  # type: ignore
        """
        Parameters
        ----------
        text : ``str``, required.
            The text to classify
        rel_label : ``str``, optional, (default = None).
            The label for this text.
        Returns
        -------
        An ``Instance`` containing the following fields:
            tokens : ``TextField``
                The tokens in the sentence or phrase.
            label : ``LabelField``
                The label label of the sentence or phrase.
        """
        fields: Dict[str, Field] = {}

        text_filed = TextField([Token(t) for t in tokens], token_indexers=self._token_indexers)
        fields["tokens"] = text_filed
        h_sp = SpanField(h_span_start, h_span_end - 1, text_filed)
        t_sp = SpanField(t_span_start, t_span_end - 1, text_filed)
        fields["span_h"] = h_sp
        fields["span_t"] = t_sp
        if rel_label is not None:
            fields["rel_label"] = LabelField(rel_label)
        return Instance(fields)
