import copy
import json
import warnings

from functools import reduce
from collections import OrderedDict, namedtuple
from typing import List

from ast import AST, NodeTransformer
from antlr4.Token import CommonToken
from antlr4 import CommonTokenStream, ParseTreeVisitor
from antlr4.tree.Tree import ParseTree

from antlr_ast.inputstream import CaseTransformInputStream


def parse(grammar, text, start, strict=False, upper=True, error_listener=None):
    input_stream = CaseTransformInputStream(text, upper=upper)

    lexer = grammar.Lexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = grammar.Parser(token_stream)
    parser.buildParseTrees = True  # default

    if strict:
        error_listener = StrictErrorListener()

    if error_listener is not None and error_listener is not True:
        parser.removeErrorListeners()
        if error_listener:
            parser.addErrorListener(error_listener)

    return getattr(parser, start)()


def process_tree(antlr_tree, Transformer=None, simplify=True):
    tree = BaseAstVisitor().visit(antlr_tree)
    if Transformer is not None:
        tree = AliasVisitor(Transformer()).visit(tree)
    if simplify:
        tree = simplify_tree(tree, unpack_lists=False)
    return tree


# TODO
#  duplicated in ast-viewer (also for Python)
#  structure vs to_json()?
def dump_node(node, node_class=AST):
    if isinstance(node, node_class):
        fields = OrderedDict()
        for name in node._fields:
            attr = getattr(node, name, None)
            if attr is None:
                continue
            elif isinstance(attr, node_class):
                fields[name] = dump_node(attr)
            elif isinstance(attr, list):
                fields[name] = [dump_node(x) for x in attr]
            else:
                fields[name] = attr
        return {"type": node.__class__.__name__, "data": fields}
    elif isinstance(node, list):
        return [dump_node(x) for x in node]
    else:
        return node


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
        return tuple(od)


# Helper functions -------


def bind_to_visitor(visitor_cls, rule_name, visitor):
    """Assign AST node class constructors to parse tree visitors."""
    setattr(visitor_cls, rule_to_visitor_name(rule_name), visitor)


def rule_to_visitor_name(rule_name):
    return "visit_{}".format(rule_name[0].upper() + rule_name[1:])


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
# TODO: make general node (Terminal) accessible in class property (.subclasses)?


class BaseNode(AST):
    """AST is subclassed so we can use Python ast module  visiting and walking on the custom AST"""

    def __init__(self, children, field_references, label_references, ctx=None):
        self.children = children

        self._field_references = field_references
        self.children_by_field = materialize(self._field_references, self.children)

        self._label_references = label_references
        self.children_by_label = materialize(self._label_references, self.children)

        self._ctx = ctx

    subclasses = {}

    _fields = ()

    # whether to descend for selection (greater descends into lower)
    _priority = 2

    # getattr: return None or raise for nonexistent attr
    # in Transformer conditionals:
    # - getattr(obj, attr, None) works with both
    # - hasattr(obj, attr) if strict
    # - obj.attr if not strict
    _strict = False

    @classmethod
    def create(cls, ctx, children=None):
        if children is None:
            children = ctx.children

        field_names = get_field_names(ctx)
        children_by_field = get_field_references(ctx, field_names)

        label_names = get_label_names(ctx)
        children_by_label = get_field_references(ctx, label_names)

        cls_name = type(ctx).__name__.split("Context")[0]
        subclass = cls.get_node_cls(cls_name, tuple(field_names))
        return subclass(children, children_by_field, children_by_label, ctx)

    @classmethod
    def get_node_cls(cls, cls_name, field_names=tuple()):
        if cls_name not in cls.subclasses:
            cls.subclasses[cls_name] = type(
                cls_name, (BaseNode,), {"_fields": field_names}
            )
        return cls.subclasses[cls_name]

    def __getattr__(self, name):
        fallback = None
        if self._strict:
            fallback = AttributeError

        result = self.children_by_label.get(
            name, self.children_by_field.get(name, fallback)
        )
        if isinstance(result, AttributeError):
            raise AttributeError('{}.{} is invalid.'.format(self.__class__.__name__, name))

        return result

    @classmethod
    def combine(cls, *fields):
        """Combine fields

        Creates a list field from other fields
        Filters None and combines other elements in a flat list
        Use in transformer methods.
        """
        result = reduce(cls.extend_node_list, fields, [])

        return result

    @staticmethod
    def extend_node_list(acc, new):
        """Extend accumulator with Node(s) from new"""
        if new is None:
            new = []
        elif not isinstance(new, list):
            new = [new]
        return acc + new

    def isinstance(self, class_name):
        return type(self).__name__ == class_name

    def get_text(self, full_text=None):
        # TODO implement as __str__?
        #  + easy to combine with str/Terminal
        #  + use Python instead of custom interface
        # (-) very different from repr / json
        if isinstance(self._ctx, ParseTree):
            if full_text is None:
                text = self._ctx.getText()
            else:
                text = full_text[self._ctx.start.start : self._ctx.stop.stop + 1]
        else:
            text = None

        return text

    def get_position(self):
        ctx = self._ctx
        d = {
            "line_start": ctx.start.line,
            "column_start": ctx.start.column,
            "line_end": ctx.stop.line,
            "column_end": ctx.stop.column + (ctx.stop.stop - ctx.stop.start),
        }
        return d

    def to_json(self):
        return {
            "@type": self.__class__.__name__,
            "@fields": self._fields,
            "field_references": self._field_references,
            "label_references": self._label_references,
            "children": self.children,
        }

    def __repr__(self):
        return str({**self.children_by_field, **self.children_by_label})


# TODO:
AstNode = BaseNode


class Terminal(BaseNode):
    """This is a thin node wrapper for a string.

    The node is transparent when not in debug mode.
    In debug mode, it keeps the link to the corresponding ANTLR node.
    """

    _fields = tuple(["value"])
    DEBUG = False
    DEBUG_INSTANCES = []

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        if cls.DEBUG:
            cls.DEBUG_INSTANCES.append(instance)
            return instance
        else:
            return args[0][0]

    def __eq__(self, other):
        return self.value == other

    def __str__(self):
        # currently just used for better formatting in debugger
        return self.value

    def to_json(self):
        return str(self)

    def __repr__(self):
        return "'{}'".format(self.value)


class AliasNode(BaseNode, metaclass=AstNodeMeta):
    # TODO: look at AstNode methods
    # defines class properties
    # - as a property name to copy from ANTLR nodes
    # - as a property name defined in terms of (nested) ANTLR node properties
    # the field will be set to the first definition that is not undefined
    _fields_spec = []

    # Defines which ANTLR nodes to convert to this node. Elements can be:
    # - a string: uses AstNode._from_fields as visitor
    # - a tuple ('node_name', 'ast_node_class_method_name'): uses ast_node_class_method_name as visitor
    # subclasses use _bind_to_visitor to create visit methods for the nodes in _rules on the ParseTreeVisitor
    # using this information
    _rules = []

    _priority = 1

    # TODO: test clear failure in SCTs
    _strict = True

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        # necessary because AST implements this field
        instance._fields = cls._fields
        return instance

    def __init__(self, node: BaseNode, fields=None):
        # TODO: keep reference to node?
        # TODO: **fields? (easier notation, but hard to name future arguments
        super().__init__(
            node.children, node._field_references, node._label_references, node._ctx
        )

        fields = fields or {}
        for field, value in fields.items():
            if field not in self._fields:
                warnings.warn("Key not in fields: {}".format(field))
            setattr(self, field, value)

    @classmethod
    def from_spec(cls, node):
        # TODO: no fields_spec argument as before
        field_dict = {}
        for field_spec in cls._fields_spec:
            name, path = parse_field_spec(field_spec)

            # _fields_spec can contain field multiple times
            # e.g. x=a and x=b
            if field_dict.get(name):
                # or / elif behaviour
                continue

            # get node -----
            field_dict[name] = cls.get_path(node, path)
        return cls(node, field_dict)

    @classmethod
    def get_path(cls, node, path):
        # TODO: can be defined on FieldNode too
        result = node
        for i in range(len(path)):
            result = getattr(result, path[i], None)
            if result is None:
                break

        return result

    @classmethod
    def bind_to_transformer(cls, transformer_cls, default_visit_method="from_spec"):
        for rule in cls._rules:
            if isinstance(rule, str):
                visit_method = default_visit_method
            else:
                rule, visit_method = rule[:2]
            visitor = cls.get_transformer(visit_method)
            bind_to_visitor(transformer_cls, rule, visitor)

    @classmethod
    def get_transformer(cls, method_name):
        """Get method to bind to visitor"""
        visit_node = getattr(cls, method_name)
        assert callable(visit_node)

        def method(self, node):
            return visit_node(node)

        return method


# TODO: test: if 'visit' in method, it has to be as 'visit_'
class AliasVisitor(NodeTransformer):
    def __init__(self, transformer_visitor):
        self.transformer_visitor = transformer_visitor

    def __getattr__(self, item):
        transformer = getattr(self.transformer_visitor, item)
        if transformer is None:
            raise AttributeError

        def visitor(node):
            alias = transformer(node)
            if isinstance(alias, AliasNode) or alias == node:
                # this prevents infinite recursion and visiting
                # AliasNodes with a name that is also the name of a BaseNode
                self.generic_visit(alias)
            else:
                # visit BaseNode (e.g. result of Transformer method)
                if isinstance(alias, list):
                    # Transformer method can return array instead of node
                    alias = [self.visit(el) for el in alias]  # TODO: test
                elif isinstance(alias, BaseNode):
                    alias = self.visit(alias)

            return alias

        return visitor

    def visit_Terminal(self, terminal):
        return self.terminal_visitor(terminal)

    def visit_str(self, string):
        return self.terminal_visitor(string)

    def terminal_visitor(self, terminal):
        """Helper to transparently handle Terminal"""
        return terminal


def simplify_tree(tree, unpack_lists=True, in_list=False):
    """Recursively unpack single-item lists and objects where fields and labels only reference a single child

    :param tree: the tree to simplify (mutating!)
    :param unpack_lists: whether single-item lists should be replaced by that item
    :param in_list: this is used to prevent unpacking a node in a list as AST visit can't handle nested lists
    """
    # TODO: copy (or (de)serialize)? outside this function?
    if isinstance(tree, BaseNode) and not isinstance(tree, Terminal):
        used_fields = [field for field in tree._fields if getattr(tree, field, False)]
        if len(used_fields) == 1:
            result = getattr(tree, used_fields[0])
        else:
            result = None
        if len(used_fields) != 1 or isinstance(tree, AliasNode) or (in_list and isinstance(result, list)):
            result = tree
            for field in tree._fields:
                setattr(result, field, simplify_tree(getattr(tree, field), unpack_lists=unpack_lists))
            return result
        assert result is not None
    elif isinstance(tree, list) and len(tree) == 1 and unpack_lists:
        result = tree[0]
    else:
        if isinstance(tree, list):
            result = [simplify_tree(el, unpack_lists=unpack_lists, in_list=True) for el in tree]
        else:
            result = tree
        return result

    return simplify_tree(result, unpack_lists=unpack_lists)


class BaseAstVisitor(ParseTreeVisitor):
    """Visitor that creates a high level tree

    ~ ANTLR tree serializer
    + automatic node creation using field and label detection
    + alias nodes can work on tree without (ANTLR) visitor

    Used from BaseAstVisitor: visitTerminal, visitErrorNode

    TODO:
     - [done] support labels
     - [done] make compatible with AST: _fields = () (should only every child once)
     - [done] include child_index to filter unique elements + order
     - [done] memoize dynamic classes, to have list + make instance checks work
     - [done] tree simplification as part of AliasNode
     - [done] flatten nested list (see select with dynamic clause ordering)
     - combine terminals / error nodes
     - serialize highlight info
     - make compatible with AstNode & AstModule in protowhat (+ shellwhat usage: bashlex + osh parser)
         - combining fields & labels dicts needed?
     - use exact ANTLR names in _rules (capitalize name without changing other casing)
     - add labels to _fields if not overlapping with fields from rules
     - [done] eliminate overhead of alias parsing (store ref to child index, get children on alias access)
     - [necessary?] grammar must use lexer or grammar rules for elements that should be in the tree
       and literals for elements that cannot
       currently:
       - Use AliasNode to add labels to _fields, define custom fields and omit fields
       - Use Transformer to replace a node by a combination of fields
     - [rejected] alternative dynamic class naming:
       - pass parse start to visitor constructor, use as init for self.current_node
       - set self.current_node to field.__name__ before self.visit_field
       - use self.current_node to create dynamic classes
       (does not use #RuleAlias names in grammar)
       (other approach: transforming returned dict, needs more work for arrays + top level)

    Higher order visitor (or integrated)
    - [alternative] allow node aliases (~ AstNode._rules) by dynamically creating a class inheriting from the dynamic node class
      (multiple inheritance if node is alias for multiple nodes, class has combined _fields for AST compatibility
    - [alternative] allow field aliases using .aliases property with defaultdict(list) (~ AstNode._fields_spec)
        - dynamic fields? (~ visit_path)

    test code in parse:
        tree = parse_ast(grammar, sql_text, start, **kwargs)
        field_tree = BaseAstVisitor().visit(tree)
        alias_tree = AliasVisitor(Transformer()).visit(field_tree)

        import ast
        nodes = [el for el in ast.walk(field_tree)]
        import json
        json_str = json.dumps(field_tree, default=lambda o: o.to_json())
    """

    def visitChildren(self, node, predicate=None, simplify=False):
        # children is None if all parts of a grammar rule are optional and absent
        children = [self.visit(child) for child in node.children or []]

        instance = BaseNode.create(node, children)

        return instance

    def visitTerminal(self, ctx):
        """Converts case insensitive keywords and identifiers to lowercase"""
        text = ctx.getText()
        quotes = ["'", '"']
        if not (text[0] in quotes and text[-1] in quotes):
            text = text.lower()
        return Terminal([text], {"value": 0}, {}, ctx)

    def visitErrorNode(self, node):
        return None


# ANTLR helpers


def get_field(ctx, field):
    """Helper to get the value of a field"""
    # field can be a string or a node attribute
    if isinstance(field, str):
        field = getattr(ctx, field, None)
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


def get_field_references(ctx, field_names, simplify=False):
    """
    Create a mapping from fields to corresponding child indices
    :param ctx: ANTLR node
    :param field_names: list of strings
    :param simplify: if True, omits fields with empty lists or None
        this makes it easy to detect nodes that only use a single field
        but it requires more work to combine fields that can be empty
    :return: mapping str -> int | int[]
    """
    field_dict = {}
    for field_name in field_names:
        field = get_field(ctx, field_name)
        if (
            not simplify
            or field is not None
            and (not isinstance(field, list) or len(field) > 0)
        ):
            if isinstance(field, list):
                value = [ctx.children.index(el) for el in field]
            elif field is not None:
                value = ctx.children.index(field)
            else:
                value = None
            field_dict[field_name] = value
    return field_dict


def materialize(reference_dict, source):
    """
    Replace indices by actual elements in a reference mapping
    :param reference_dict: mapping str -> int | int[]
    :param source: list of elements
    :return: mapping str -> element | element[]
    """
    materialized_dict = {}
    for field in reference_dict:
        reference = reference_dict[field]
        if isinstance(reference, list):
            materialized_dict[field] = [source[index] for index in reference]
        elif reference is not None:
            materialized_dict[field] = source[reference]
        else:
            materialized_dict[field] = None
    return materialized_dict


def get_field_names(ctx):
    """Get fields defined in an ANTLR context for a parser rule"""
    # this does not include labels and literals, only rule names and token names
    # TODO: check ANTLR parser template for full exclusion list
    fields = [
        field
        for field in type(ctx).__dict__
        if not field.startswith("__")
        and field not in ["accept", "enterRule", "exitRule", "getRuleIndex", "copyFrom"]
    ]
    return fields


def get_label_names(ctx):
    """Get labels defined in an ANTLR context for a parser rule"""
    labels = [
        label
        for label in ctx.__dict__
        if not label.startswith("_")
        and label
        not in [
            "children",
            "exception",
            "invokingState",
            "parentCtx",
            "parser",
            "start",
            "stop",
        ]
    ]
    return labels
