# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unicodedata
from typing import Callable, List, Dict

from fairseq import file_utils
from fairseq.data.encoders import register_bpe
from sentencepiece import SentencePieceProcessor


@register_bpe('sentencepiece')
class SentencepieceBPE(object):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--sentencepiece-model', type=str, help='path to sentencepiece model')
        parser.add_argument('--sentencepiece-encode-shapes', type=bool, help='encode shapes', default=False)
        # fmt: on

    def __init__(self, args):
        sentencepiece_model = file_utils.cached_path(args.sentencepiece_model)
        try:
            import sentencepiece as spm
            self.sp = spm.SentencePieceProcessor()
            self.sp.Load(sentencepiece_model)
            self.encode_shapes = args.sentencepiece_encode_shapes
        except ImportError:
            raise ImportError('Please install sentencepiece with: pip install sentencepiece')

    def encode(self, x: str) -> str:
        if self.encode_shapes:
            return self._encode_shapes(x)
        else:
            return ' '.join(self.sp.EncodeAsPieces(x))

    def decode(self, x: str) -> str:
        return x.replace(' ', '').replace('\u2581', ' ').strip()

    def _encode_shapes(self, line: str):
        line = unicodedata.normalize('NFD', line)
        line = u"".join([c for c in line if not unicodedata.combining(c)])
        line = self.sp.DecodePieces(self.sp.EncodeAsPieces(line))
        lower = line.lower()
        pieces = [val for val in self.sp.EncodeAsPieces(lower)]
        piece_indices = [0]
        current_idx = 0
        for piece in pieces:
            piece_size = len(piece)
            current_idx += piece_size
            piece_indices.append(current_idx)
        line = " " + line
        shaped_pieces = []
        for i in range(len(piece_indices)-1):
            start = piece_indices[i]
            end = piece_indices[i+1]
            piece = line[start:end].replace(" ", "▁")
            id = self.sp.PieceToId(piece.lower())
            shaped_pieces.append(piece if id != self.sp.unk_id() else "<unk>")
        return ' '.join(shaped_pieces)

    def is_beginning_of_word(self, x: str) -> bool:
        if x in ['<unk>', '<s>', '</s>', '<pad>']:
            # special elements are always considered beginnings
            # HACK: this logic is already present in fairseq/tasks/masked_lm.py
            # but these special tokens are also contained in the sentencepiece
            # vocabulary which causes duplicate special tokens. This hack makes
            # sure that they are all taken into account.
            return True
        return x.startswith('\u2581')
