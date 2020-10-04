# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from fairseq.data.dictionary import TokenShapesDictionary, Dictionary
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

logger = logging.getLogger(__name__)


@register_task('sentence_prediction_with_token_shapes')
class SentencePredictionWithTokenShapesTask(SentencePredictionTask):

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        if source:
            dictionary = TokenShapesDictionary.load(filename)
        else:
            dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

