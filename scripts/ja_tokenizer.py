#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""Tokenize japanese texts.

Usage 1 (file):
```
ja_tokenizer.py < input.txt > output.txt
```

Usage 2 (stdin):
```
$ echo '日本語分かち書きのテスト。' | ja_tokenizer.py
日本語 分かち書き の テスト 。
```
"""

import sys
import neologdn
import MeCab
dict_dir = "/usr/local/Cellar/mecab/0.996/lib/mecab/dic/mecab-ipadic-neologd"

def main():
    m = MeCab.Tagger("-Owakati -d %s" % dict_dir)
    for tok in sys.stdin.readlines():
        in_text = tok.strip().decode('utf-8')
        out_text = m.parse(neologdn.normalize(in_text).encode('utf-8'))
        sys.stdout.write(out_text)

if __name__ == "__main__":
    main()
