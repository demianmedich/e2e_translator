# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# [Note] Please add me fixed index to all vocabs like ordering in SPECIAL_TOKENS...
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'

SPECIAL_TOKENS = [
    PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
]

PAD_TOKEN_ID = SPECIAL_TOKENS.index(PAD_TOKEN)
SOS_TOKEN_ID = SPECIAL_TOKENS.index(SOS_TOKEN)
EOS_TOKEN_ID = SPECIAL_TOKENS.index(EOS_TOKEN)
UNK_TOKEN_ID = SPECIAL_TOKENS.index(UNK_TOKEN)
