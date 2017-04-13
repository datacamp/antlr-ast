__version__ = '0.0.1'

from ast import AST
from antlr4.Token import CommonToken
import json

from collections import OrderedDict
def dump_node(obj):
    if isinstance(obj, AstNode):
        fields = OrderedDict()
        for name in obj._get_field_names():
            attr = getattr(obj, name, None)
            if attr is None: continue
            elif isinstance(attr, AstNode): fields[name] = attr._dump()
            elif isinstance(attr, list):    fields[name] = [dump_node(x) for x in attr]
            else:                           fields[name] = attr
        return {'type': obj.__class__.__name__, 'data': fields}
    elif isinstance(obj, list):
        return [dump_node(x) for x in obj]
    else:
        return obj

class AstNode(AST):         # AST is subclassed only so we can use ast.NodeVisitor...
    _fields = []            # contains child nodes to visit
    _priority = 1           # whether to descend for selection (greater descends into lower)

    def __init__(self, _ctx = None, **kwargs):
        # TODO: ensure key is in _fields?
        for k, v in kwargs.items(): setattr(self, k, v)
        self._ctx = _ctx

    # default visiting behavior, which uses fields
    @classmethod
    def _from_fields(cls, visitor, ctx, fields=None):

        fields = cls._fields if fields is None else fields

        field_dict = {}
        for mapping in fields:
            # parse mapping for -> and indices [] -----
            k, *name = mapping.split('->')
            name = k if not name else name[0]

            # currently, get_field_names will show field multiple times,
            # e.g. a->x and b->x will produce two x fields
            if field_dict.get(name): continue

            # get node -----
            #print(k)
            child = getattr(ctx, k, getattr(ctx, name, None))
            # when not alias needs to be called
            if callable(child): child = child()
            # when alias set on token, need to go from CommonToken -> Terminal Node
            elif isinstance(child, CommonToken):
                # giving a name to lexer rules sets it to a token,
                # rather than the terminal node corresponding to that token
                # so we need to find it in children
                child = next(filter(lambda c: getattr(c, 'symbol', None) is child, ctx.children))

            # set attr -----
            if isinstance(child, list):
                field_dict[name] = [visitor.visit(el) for el in child]
            elif child:
                field_dict[name] = visitor.visit(child)
            else:
                field_dict[name] = child
        return cls(ctx, **field_dict)

    def _get_field_names(self):
        return [el.split('->')[-1] for el in self._fields]

    def _get_text(self, text):
        return text[self._ctx.start.start: self._ctx.stop.stop + 1]

    def _get_pos(self):
        ctx = self._ctx
        d = {'line_start': ctx.start.line,
                'column_start': ctx.start.column,
                'line_end': ctx.stop.line,
                'column_end': ctx.stop.column + ctx.stop.stop - ctx.stop.start}
        return d

    def _dump(self):
        return dump_node(self)

    def _dumps(self):
        return json.dumps(self._dump())

    def _load(self):
        raise NotImplementedError()

    def _loads(self):
        raise NotImplementedError()

    def __str__(self):
        els = [k for k in self._get_field_names() if getattr(self, k, None) is not None]
        return "{}: {}".format(self.__class__.__name__, ", ".join(els))

    def __repr__(self):
        field_reps = {k: repr(getattr(self, k)) for k in self._get_field_names() if getattr(self, k, None) is not None}
        args = ", ".join("{} = {}".format(k, v) for k, v in field_reps.items())
        return "{}({})".format(self.__class__.__name__, args)
