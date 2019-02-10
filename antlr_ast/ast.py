import copy
from ast import AST
from antlr4.Token import CommonToken
from antlr4 import CommonTokenStream, ParserRuleContext, ParseTreeVisitor
from antlr_ast.inputstream import CaseTransformInputStream
import json

from collections import OrderedDict, namedtuple
from typing import List
import warnings


def parse(grammar, text, start, strict=False, upper=True, error_listener=None):
    input_stream = CaseTransformInputStream(text, upper=upper)

    lexer = grammar.Lexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = grammar.Parser(token_stream)
    parser.buildParseTrees = True  # default

    if strict:
        error_listener = StrictErrorListener()

    if error_listener is not None:
        parser.removeErrorListeners()
        if error_listener:
            parser.addErrorListener(error_listener)

    return getattr(parser, start)()


def dump_node(obj):
    if isinstance(obj, AstNode):
        fields = OrderedDict()
        for name in obj._fields:
            attr = getattr(obj, name, None)
            if attr is None:
                continue
            elif isinstance(attr, AstNode):
                fields[name] = dump_node(attr)
            elif isinstance(attr, list):
                fields[name] = [dump_node(x) for x in attr]
            else:
                fields[name] = attr
        return {"type": obj.__class__.__name__, "data": fields}
    elif isinstance(obj, list):
        return [dump_node(x) for x in obj]
    else:
        return obj


FieldSpec = namedtuple("FieldSpec", ["name", "origin"])


def parse_field_spec(spec):
    # parse mapping for = and .  # old: and indices [] -----
    name, *origin = spec.split("=")
    origin = name if not origin else origin[0]
    origin = origin.split(".")
    return FieldSpec(name, origin)


class AstNodeMeta(type):
    @property
    def _fields(cls):
        od = OrderedDict([(parse_field_spec(el).name, None) for el in cls._fields_spec])
        return list(od)


class AstNode(AST, metaclass=AstNodeMeta):
    """AST is subclassed so we can use ast.NodeVisitor on the custom AST"""

    # defines class properties
    # - as a property name to copy from ANTLR nodes
    # - as a property name defined in terms of (nested) ANTLR node properties
    _fields_spec = []

    # Defines which ANTLR nodes to convert to this node. Elements can be:
    # - a string: uses AstNode._from_fields as visitor
    # - a tuple ('node_name', 'ast_node_class_method_name'): uses ast_node_class_method_name as visitor
    # subclasses use _bind_to_visitor to create visit methods for the nodes in _rules on the ParseTreeVisitor
    # using this information
    _rules = []

    # whether to descend for selection (greater descends into lower)
    _priority = 1

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        # necessary because AST implements this field
        instance._fields = cls._fields
        return instance

    def __init__(self, _ctx=None, **kwargs):
        for k, v in kwargs.items():
            if k not in self._fields:
                warnings.warn("Key not in fields: {}".format(k))
            setattr(self, k, v)

        self._ctx = _ctx

    @classmethod
    def _from_fields(
        cls, visitor, ctx: ParserRuleContext, fields_spec: List[str] = None
    ):
        """default visiting behavior, which uses fields"""

        fields_spec = cls._fields_spec if fields_spec is None else fields_spec

        field_dict = {}
        for field_spec in fields_spec:
            name, path = parse_field_spec(field_spec)

            # _fields_spec can contain field multiple times
            # e.g. x=a and x=b
            if field_dict.get(name):
                # or / elif behaviour
                continue

            # get node -----
            field_dict[name] = visitor.visit_path(ctx, path)
        return cls(ctx, **field_dict)

    def _get_field_names(self):
        return self._fields

    def _get_text(self, text):
        return text[self._ctx.start.start : self._ctx.stop.stop + 1]

    def _get_pos(self):
        ctx = self._ctx
        d = {
            "line_start": ctx.start.line,
            "column_start": ctx.start.column,
            "line_end": ctx.stop.line,
            "column_end": ctx.stop.column + ctx.stop.stop - ctx.stop.start,
        }
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
        field_reps = [
            (k, repr(getattr(self, k)))
            for k in self._get_field_names()
            if getattr(self, k, None) is not None
        ]
        args = ", ".join("{} = {}".format(k, v) for k, v in field_reps)
        return "{}({})".format(self.__class__.__name__, args)

    @classmethod
    def _bind_to_visitor(cls, visitor_cls, method="_from_fields"):
        for rule in cls._rules:
            if not isinstance(rule, str):
                rule, method = rule[:2]
            visitor = get_visitor(cls, method)
            bind_to_visitor(visitor_cls, rule, visitor)


# Helper functions -------


def get_visitor(node, method="_from_fields"):
    visit_node = getattr(node, method)
    assert callable(visit_node)

    def visitor(self, ctx):
        return visit_node(self, ctx)

    return visitor


def bind_to_visitor(visitor_cls, rule_name, visitor):
    """Assign AST node class constructors to parse tree visitors."""
    setattr(visitor_cls, rule_to_visitor_name(rule_name), visitor)


def rule_to_visitor_name(rule_name):
    return "visit{}".format(rule_name[0].upper() + rule_name[1:])


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
        self.node_names = cfg["nodes"]
        self.field_names = cfg.get("fields", {})

    def describe(self, node, fmt="{node_name}", field=None, **kwargs):
        cls_name = node.__class__.__name__
        def_field_name = (
            self.field_names.get(field) or field.replace("_", " ") if field else ""
        )

        node_cfg = self.node_names.get(cls_name, cls_name)
        node_name, field_names = self.get_info(node_cfg)

        d = {
            "node": node,
            "field_name": field_names.get(field, def_field_name),
            "node_name": node_name.format(node=node),
        }

        return fmt.format(**d, **kwargs)

    @staticmethod
    def get_info(node_cfg):
        """Return a tuple with the verbal name of a node, and a dict of field names."""

        node_cfg = node_cfg if isinstance(node_cfg, dict) else {"name": node_cfg}

        return node_cfg.get("name"), node_cfg.get("fields", {})


# Error Listener ------------------------------------------------------------------

from antlr4.error.ErrorListener import ErrorListener

# from antlr4.error.Errors import RecognitionException


class AntlrException(Exception):
    def __init__(self, msg, orig):
        self.msg, self.orig = msg, orig


class StrictErrorListener(ErrorListener):
    def syntaxError(self, recognizer, badSymbol, line, col, msg, e):
        if e is not None:
            msg = "line {line}: {col} {msg}".format(line=line, col=col, msg=msg)
            raise AntlrException(msg, e)
        else:
            raise AntlrException(msg, None)

    def reportAmbiguity(
        self, recognizer, dfa, startIndex, stopIndex, exact, ambigAlts, configs
    ):
        return
        # raise Exception("TODO")

    def reportAttemptingFullContext(
        self, recognizer, dfa, startIndex, stopIndex, conflictingAlts, configs
    ):
        return
        # raise Exception("TODO")

    def reportContextSensitivity(
        self, recognizer, dfa, startIndex, stopIndex, prediction, configs
    ):
        return
        # raise Exception("TODO")


# Parse Tree Visitor ----------------------------------------------------------
# TODO: visitor inheritance not really needed, but indicates compatibility
# TODO: make general nodes accessible in class property?


class Unshaped(AstNode):
    _fields_spec = ["arr"]

    def __init__(self, ctx, arr=tuple()):
        self.arr = arr
        self._ctx = ctx


class Terminal(AstNode):
    """This is a thin node wrapper for a string.

    The node is transparent when not in debug mode.
    In debug mode, it keeps the link to the corresponding ANTLR node.
    """

    _fields_spec = ["value"]
    DEBUG = True
    DEBUG_INSTANCES = []

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        if cls.DEBUG:
            cls.DEBUG_INSTANCES.append(instance)
            return instance
        else:
            return kwargs.get("value", "")

    def __str__(self):
        # currently just used for better formatting in debugger
        return self.value

    def __repr__(self):
        # used for serialization
        return "'{}'".format(self.value)


class BaseAstVisitor(ParseTreeVisitor):
    def visitChildren(self, node, predicate=None, simplify=True):
        """This is the default visiting behaviour

        :param node: current ANTLR node
        :param predicate: skip a child if this function evaluates to false for the child
        :param simplify: whether the result of the visited children should be combined if possible
        :return:
        """
        result = self.defaultResult()
        for child in node.getChildren(predicate):
            if not self.shouldVisitNextChild(node, result):
                return result

            result = self.aggregateResult(result, self.visit(child))

        return self.result_to_ast(node, result, simplify=simplify)

    @staticmethod
    def result_to_ast(node, result, simplify=True):
        if len(result) == 0:
            return None
        elif simplify and len(result) == 1:
            return result[0]
        elif simplify and (
            all(isinstance(res, Terminal) for res in result)
            or all(isinstance(res, str) for res in result)
        ):
            if simplify:
                try:
                    ctx = copy.copy(result[0]._ctx)
                    ctx.symbol = copy.copy(ctx.symbol)
                    ctx.symbol.stop = result[-1]._ctx.symbol.stop
                except AttributeError:
                    ctx = node
                return Terminal(
                    ctx, value=" ".join(map(lambda t: getattr(t, "value", t), result))
                )
        elif all(
            isinstance(res, AstNode) and not isinstance(res, Unshaped) for res in result
        ) or (not simplify and all(res is not None for res in result)):
            return result
        else:
            if all(res is None for res in result):
                # return unparsed text
                result = node.start.getInputStream().getText(
                    node.start.start, node.stop.stop
                )
            return Unshaped(node, result)

    def defaultResult(self):
        return list()

    def aggregateResult(self, aggregate, nextResult):
        aggregate.append(nextResult)
        return aggregate

    def visitTerminal(self, ctx):
        """Converts case insensitive keywords and identifiers to lowercase"""
        text = ctx.getText()
        quotes = ["'", '"']
        if not (text[0] in quotes and text[-1] in quotes):
            text = text.lower()
        return Terminal(ctx, value=text)

    def visitErrorNode(self, node):
        return None

    def get_field(self, ctx, field):
        # future todo: split get_field
        # when not alias needs to be called
        if callable(field):
            field = field()
        # when alias set on token, need to go from CommonToken -> Terminal Node
        elif isinstance(field, CommonToken):
            # giving a name to lexer rules sets it to a token,
            # rather than the terminal node corresponding to that token
            # so we need to find it in children
            field = next(
                filter(lambda c: getattr(c, "symbol", None) is field, ctx.children)
            )
        return field

    def visit_field(self, ctx, field):
        field = self.get_field(ctx, field)
        if isinstance(field, list):
            result = [self.visit(el) for el in field]
            # simplify arg could be used for * unpacking field spec (e.g. select.order_by)
            # if simplify and len(result) == 1:
            #     result = result[0]
        elif field:
            result = self.visit(field)
        else:
            result = field
        return result

    def visit_path(self, ctx, path):
        result = ctx
        for i in range(len(path)):
            result = getattr(result, path[i], None)
            if result is None:
                break
            elif i == len(path) - 1:
                result = self.visit_field(ctx, result)
            else:
                result = result()

        return result


class ObjectNode:
    def __init__(self, fields, ctx):
        self._fields_data = fields
        self._ctx = ctx

    def __repr__(self):
        return str({"@type": self.__class__.__name__, **self._fields_data})

    def __getattr__(self, item):
        try:
            return self._fields_data[item]
        except KeyError:
            raise AttributeError


class ObjectAstVisitor(BaseAstVisitor):
    """PROTOTYPE: Visitor that creates a high level tree

    ~ ANTLR tree serializer

    TODO:
    - [done] support labels
    - make compatible with AST: _fields = () (should only every child once)
    - include child_index to filter unique elements + order
    - memoize dynamic classes, to have list + make instance checks work?
    - flatten nested list (see select with dynamic clause ordering)
    - eliminate overhead of alias parsing (store ref to child index, get children on alias access)
    - grammar must use lexer or grammar rules for elements that should be in the tree
      and literals for elements that cannot
    - [done] alternative dynamic class naming:
      - pass parse start to visitor constructor, use as init for self.current_node
      - set self.current_node to field.__name__ before self.visit_field
      - use self.current_node to create dynamic classes
      (does not use #RuleAlias names in grammar)
      (other approach: transforming returned dict, needs more work for arrays + top level)

    Higher order visitor (or integrated)
    - allow node aliases (~ AstNode._rules) by dynamically creating a class inheriting from the dynamic node class
      (multiple inheritance if node is alias for multiple nodes, class has combined _fields for AST compatibility
    - allow field aliases using .aliases property with defaultdict(list) (~ AstNode._fields_spec)
        - dynamic fields? (~ visit_path)

    test code in parse:
        from antlr_ast.ast import ObjectAstVisitor
        tst = ObjectAstVisitor().visit(tree)
    """

    def visitChildren(self, node, predicate=None, simplify=True):
        fields = self.get_field_names(node)
        return self.visit_fields(node, fields, simplify)

    def visit_fields(self, ctx, fields, simplify):
        field_dict = {}
        for field_name in fields:
            field = getattr(ctx, field_name, None)
            value = self.visit_field(ctx, field)
            if value is not None:
                if not isinstance(value, (dict, list)) or len(value) > 0:
                    field_dict[field_name] = value

        cls_name = type(ctx).__name__.split('Context')[0]
        cls = type(cls_name, (ObjectNode,), {'_fields': tuple(fields)})
        instance = cls(field_dict, ctx)

        # simplifies tree, but loses intermediate path
        if simplify and len(field_dict) == 1:
            instance = list(field_dict.values())[0]

        return instance

    @staticmethod
    def get_labels(ctx):
        labels = [
            label for label in ctx.__dict__ if
            label not in ['children', 'exception', 'invokingState', 'parentCtx', 'parser', 'start', 'stop']
        ]
        return labels

    @staticmethod
    def get_field_names(ctx):
        # this does not include labels
        # only rule names and token names are in the tree (not literals)
        fields = [
            field
            for field in type(ctx).__dict__
            if not field.startswith("__")
            and field not in ["accept", "enterRule", "exitRule", "getRuleIndex"]
        ]
        return fields

    _remove_terminal = []
