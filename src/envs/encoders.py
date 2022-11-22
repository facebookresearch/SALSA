# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from abc import ABC, abstractmethod
import numpy as np
import math


class Encoder(ABC):
    """
    Base class for encoders, encodes and decodes matrices
    abstract methods for encoding/decoding numbers
    """
    def __init__(self, params, single, output=False):
        self.int_base = params.input_int_base if not output else params.output_int_base
        self.balanced = params.balanced_base
        if self.balanced:
            max_digit = (self.int_base - 1) // 2
            self.symbols = [str(i) for i in range(-max_digit-1, max_digit+1)]
        else:
            self.symbols = [str(i) for i in range(self.int_base)]
        self.separator = "|"
        self.ab_separator = "+"
        self.no_separator = params.no_separator
        self.int_len = math.floor(math.log(params.Q, self.int_base)) + 1

        self.dim = 1 if single else params.N

    def write_int(self, val):
        if self.balanced:
            return self.write_int_balanced(val)
        else:
            return self.write_int_normal(val)

    def write_int_normal(self, val):
        res = []
        init_val = val
        v = val
        for i in range(self.int_len):
            res.append(str(v % self.int_base))
            v = v // self.int_base
        return res

    def write_int_balanced(self, val):
        """
        Convert a decimal integer to a representation in the given base.
        The base can be negative.
        In balanced bases (positive), digits range from -(base-1)//2 to (base-1)//2
        """
        init_val = val
        base = self.int_base
        balanced = self.balanced
        res = []
        max_digit = abs(base)
        if balanced:
            max_digit = (base - 1) // 2
        else:
            if base > 0:
                neg = val < 0
                val = -val if neg else val
        for i in range(self.int_len):
            rem = val % base
            val = val // base
            if rem < 0 or rem > max_digit:
                rem -= base
                val += 1
            res.append(str(rem))
        while len(res) < self.int_len:
            res.append('0')
        return res

    def parse_int(self, lst):
        if self.balanced:
            return self.parse_int_balanced(lst)
        else:
            return self.parse_int_normal(lst)

    def parse_int_balanced(self, lst):
        """
        Parse a list that starts with an integer.
        Return the integer value, and the position it ends in the list.
        """
        base = self.int_base
        val = 0
        i = 0
        for x in lst: #[1:]:
            if not (x.isdigit() or x[0] == '-' and x[1:].isdigit()):
                break
            val = val * base + int(x)
            i += 1
        return val, i #+ 1

    def parse_int_normal(self, lst):
        res = 0
        for i in range(self.int_len):
            if i >= len(lst) or not lst[i].isdigit():
                return -1, i
            res = res * self.int_base + int(lst[i])
        return res, i        

    def encode(self, vector):
        lst = []
        for val in vector:
            lst.extend(self.write_int(val))
            if not self.no_separator:
                lst.append(self.separator)
        return lst

    def decode(self, lst):
        h = lst
        m = []
        for i in range(self.dim):
            val, pos = self.parse_int(h)
            if val == -1:
                return None
            if not self.no_separator:
                if h[pos] != self.separator:
                    return None
                pos += 1
            h = h[pos:]
            m.append(val)
        return m

    

