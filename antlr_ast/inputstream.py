from operator import methodcaller

from antlr4 import InputStream


class CaseTransformInputStream(InputStream):
    """Support case insensitive languages
    https://github.com/antlr/antlr4/blob/master/doc/case-insensitive-lexing.md#custom-character-streams-approach
    """

    def __init__(self, *args, upper=None, **kwargs):
        self.upper = upper
        super().__init__(*args, **kwargs)

    def _loadString(self):
        self._index = 0
        if self.upper:
            transform = methodcaller("upper")
        elif self.upper is False:
            transform = methodcaller("lower")
        elif self.upper is None:
            transform = lambda x: x

        self.data = [ord(transform(c)) for c in self.strdata]
        self._size = len(self.data)
