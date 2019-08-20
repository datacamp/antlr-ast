from operator import methodcaller

from antlr4 import InputStream


class CaseTransformInputStream(InputStream):
    """Support case insensitive languages
    https://github.com/antlr/antlr4/blob/master/doc/case-insensitive-lexing.md#custom-character-streams-approach
    """
    UPPER = "upper"
    LOWER = "lower"

    def __init__(self, *args, transform=None, **kwargs):
        if transform is None:
            self.transform = lambda x: x
        elif transform == self.UPPER:
            self.transform = methodcaller("upper")
        elif transform == self.LOWER:
            self.transform = methodcaller("lower")
        elif callable(transform):
            self.transform = transform
        else:
            raise ValueError("Invalid transform")

        super().__init__(*args, **kwargs)

    def _loadString(self):
        self._index = 0

        self.data = [ord(self.transform(c)) for c in self.strdata]
        self._size = len(self.data)

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.transform)
