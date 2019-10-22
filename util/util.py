# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class AttributeDict(dict):
    def __getattr__(self, name):
        return self[name]
