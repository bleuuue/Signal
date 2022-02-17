from __future__ import unicode_literals
from __future__ import division

import hgtk

DECOMPOSED = 'ㄴㅏㄹᴥㅣᴥ'
COMPOSED = '가연'


def test_compose():
    print ("compose", hgtk.text.compose(DECOMPOSED))
    assert hgtk.text.compose(DECOMPOSED) == COMPOSED


def test_decompose():
    print ("decompose", hgtk.text.decompose(COMPOSED))
    assert hgtk.text.decompose(COMPOSED) == DECOMPOSED

#test_compose()