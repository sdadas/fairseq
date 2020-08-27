import logging
import os

from fairseq import tokenizer, utils
from fairseq.data.dictionary import TokenShapesDictionary
from fairseq.tasks import register_task
from fairseq.tasks.masked_lm import MaskedLMTask

logger = logging.getLogger(__name__)


@register_task('masked_lm_with_token_shapes')
class MaskedLMWithTokenShapesTask(MaskedLMTask):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = TokenShapesDictionary.load(os.path.join(paths[0], 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    @classmethod
    def load_dictionary(cls, filename):
        return TokenShapesDictionary.load(filename)

    @classmethod
    def build_dictionary(cls, filenames, workers=1, threshold=-1, nwords=-1, padding_factor=8):
        d = TokenShapesDictionary()
        for filename in filenames:
            TokenShapesDictionary.add_file_to_dictionary(
                filename, d, tokenizer.tokenize_line, workers
            )
        d.finalize(threshold=threshold, nwords=nwords, padding_factor=padding_factor)
        return d