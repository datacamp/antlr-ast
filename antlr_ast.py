__version__ = '0.2.0'

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
    _rules = []

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
        od = OrderedDict([(el.split('->')[-1], None) for el in self._fields])
        return list(od)

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
        field_reps = [(k, repr(getattr(self, k))) for k in self._get_field_names() if getattr(self, k, None) is not None]
        args = ", ".join("{} = {}".format(k, v) for k, v in field_reps)
        return "{}({})".format(self.__class__.__name__, args)

    @classmethod
    def _bind_to_visitor(cls, visitor_cls, method='_from_fields'):
        for rule in cls._rules:
            if not isinstance(rule, str): rule, method = rule[:2]
            bind_to_visitor(visitor_cls, cls, rule, method)



# Helper functions -------

def create_visitor(node, method='_from_fields'):
    f = getattr(node, method)
    assert callable(f)
    def visitor(self, ctx):
        return f(self, ctx)

    return visitor

def bind_to_visitor(visitor_cls, node_cls, rule_name, method):
    """Assign AST node class constructors to parse tree visitors."""
    f = create_visitor(node_cls, method)
    # TODO raise warning if attr already on visitor?
    method_name = 'visit%s' %rule_name.capitalize()
    setattr(visitor_cls, method_name, f)


# Speaker class ---------------------------------------------------------------

class Speaker:
    def __init__(self, **cfg):
        """Initialize speaker instance, for a set of AST nodes.

        Arguments:
            nodes:  dictionary of node names, and their human friendly names.
                    Each entry for a node may also be a dictionary containing
                    name: human friendly name, fields: a dictionary to override
                    the field names for that node.
            fields: dictionary of human friendly field names, used as a default
                    for each node.
        """
        self.node_names  = cfg['nodes']
        self.field_names = cfg.get('fields', {})

    def describe(self, node, fmt = "{node_name}", field = None, **kwargs):
        cls_name = node.__class__.__name__
        def_field_name = self.field_names.get(field) or field.replace('_', ' ') if field else ""

        node_cfg = self.node_names.get(cls_name, cls_name)
        node_name, field_names = self.get_info(node_cfg)

        d = {'node': node,
             'field_name': field_names.get(field, def_field_name),
             'node_name': node_name.format(node=node)
             }

        return fmt.format(**d, **kwargs)

    @staticmethod
    def get_info(node_cfg):
        """Return a tuple with the verbal name of a node, and a dict of field names."""

        node_cfg = node_cfg if isinstance(node_cfg, dict) else {'name': node_cfg}

        return node_cfg.get('name'), node_cfg.get('fields', {})



        

