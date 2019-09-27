import warnings
import inspect

from typing import Dict, Optional, List, Union, Type, Any, Callable

from functools import reduce
from collections import OrderedDict, namedtuple

from ast import AST, NodeTransformer

from antlr4.Token import CommonToken
from antlr4 import CommonTokenStream, ParseTreeVisitor, ParserRuleContext, RuleContext
from antlr4.tree.Tree import ErrorNode, TerminalNodeImpl, ParseTree

from antlr_ast.inputstream import CaseTransformInputStream
from antlr4.error.ErrorListener import ErrorListener, ConsoleErrorListener


def parse(
    grammar,
    text: str,
    start: str,
    strict=False,
    transform: Union[str, Callable] = None,
    error_listener: ErrorListener = None,
) -> ParseTree:
    input_stream = CaseTransformInputStream(text, transform=transform)

    lexer = grammar.Lexer(input_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(LexerErrorListener())

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


def process_tree(
    antlr_tree: ParseTree,
    base_visitor_cls: Type["BaseAstVisitor"] = None,
    transformer_cls: Type["BaseNodeTransformer"] = None,
    simplify=True,
) -> "BaseNode":
    cls_registry = BaseNodeRegistry()

    if not base_visitor_cls:
        base_visitor_cls = BaseAstVisitor
    elif not issubclass(base_visitor_cls, BaseAstVisitor):
        raise ValueError("base_visitor_cls must be a BaseAstVisitor subclass")
    tree = base_visitor_cls(cls_registry).visit(antlr_tree)

    if transformer_cls is not None:
        if not issubclass(transformer_cls, BaseNodeTransformer):
            raise ValueError("transformer_cls must be a BaseNodeTransformer subclass")
        tree = transformer_cls(cls_registry).visit(tree)

    if simplify:
        tree = simplify_tree(tree, unpack_lists=False)

    return tree


# TODO use protowhat dump + DumpConfig
#  duplicated in ast-viewer (also for Python)
#  structure vs to_json()?
def dump_node(node, node_class=AST):
    if isinstance(node, node_class):
        fields = OrderedDict()
        for name in node._fields:
            attr = getattr(node, name, None)
            if attr is not None:
                fields[name] = dump_node(attr, node_class=node_class)
        return {"type": node.__class__.__name__, "data": fields}
    elif isinstance(node, list):
        return [dump_node(x, node_class=node_class) for x in node]
    else:
        return node


FieldSpec = namedtuple("FieldSpec", ["name", "origin"])


def parse_field_spec(spec: str) -> FieldSpec:
    # parse mapping for = and .  # old: and indices [] -----
    name, *origin = [part.strip() for part in spec.split("=")]
    origin = name if not origin else origin[0]
    origin = origin.split(".")
    return FieldSpec(name, origin)


class AstNodeMeta(type):
    @property
    def _fields(cls):
        od = OrderedDict([(parse_field_spec(el).name, None) for el in cls._fields_spec])
        return tuple(od)


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


# from antlr4.error.Errors import RecognitionException


class AntlrException(Exception):
    def __init__(self, msg, orig):
        self.msg, self.orig = msg, orig


class StrictErrorListener(ErrorListener):
    # The recognizer will be the parser instance
    def syntaxError(self, recognizer, badSymbol, line, col, msg, e):
        msg = "line {line}:{col} {msg}".format(
            badSymbol=badSymbol, line=line, col=col, msg=msg
        )
        raise AntlrException(msg, e)

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


class LexerErrorListener(ConsoleErrorListener):
    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        if isinstance(e.input, CaseTransformInputStream):
            msg = msg + " " + repr(e.input)
        super().syntaxError(recognizer, offendingSymbol, line, column, msg, e)


# Parse Tree Visitor ----------------------------------------------------------
# TODO: visitor inheritance not really needed, but indicates compatibility
# TODO: make general node (Terminal) accessible in class property (.subclasses)?

IndexReferences = Dict[str, Union[int, List[int]]]


class BaseNode(AST):
    """AST is subclassed so we can use Python ast module  visiting and walking on the custom AST"""

    def __init__(
        self,
        children: list,
        field_references: IndexReferences,
        label_references: IndexReferences,
        ctx: Optional[ParserRuleContext] = None,
        position: Optional[dict] = None,
        text: Optional[str] = None,
    ):
        self.children = children

        self._field_references = field_references
        self.children_by_field = materialize(self._field_references, self.children)

        self._label_references = label_references
        self.children_by_label = materialize(self._label_references, self.children)

        self._ctx = ctx
        self.position = position
        self.text = text

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
    def create(
        cls,
        ctx: ParserRuleContext,
        children: Optional[list] = None,
        registry: Optional["BaseNodeRegistry"] = None,
    ) -> "BaseNode":
        if registry is None:
            registry = BaseNodeRegistry()
        if children is None:
            children = ctx.children

        field_names = get_field_names(ctx)
        children_by_field = get_field_references(ctx, field_names)

        label_names = get_label_names(ctx)
        children_by_label = get_field_references(ctx, label_names)

        cls_name = type(ctx).__name__.split("Context")[0]
        subclass = registry.get_cls(cls_name, tuple(field_names))

        return subclass(children, children_by_field, children_by_label, ctx)

    @classmethod
    def create_cls(cls, cls_name: str, field_names: tuple) -> Type["BaseNode"]:
        return type(cls_name, (cls,), {"_fields": field_names})

    def __getattr__(self, name):
        try:
            result = self.children_by_label.get(name) or self.children_by_field[name]
        except KeyError:
            if self._strict:
                raise AttributeError(
                    "{}.{} is invalid.".format(self.__class__.__name__, name)
                )
            else:
                result = None

        return result

    @classmethod
    def combine(cls, *fields: "BaseNode") -> List["BaseNode"]:
        """Combine fields

        Creates a list field from other fields
        Filters None and combines other elements in a flat list
        Use in transformer methods.
        """
        result = reduce(cls.extend_node_list, fields, [])

        return result

    @staticmethod
    def extend_node_list(
        acc: List["BaseNode"], new: Union[List["BaseNode"], "BaseNode"]
    ) -> List["BaseNode"]:
        """Extend accumulator with Node(s) from new"""
        if new is None:
            new = []
        elif not isinstance(new, list):
            new = [new]
        return acc + new

    def get_text(self, full_text: str = None) -> Optional[str]:
        # TODO implement as __str__?
        #  + easy to combine with str/Terminal
        #  + use Python instead of custom interface
        # (-) very different from repr / json
        text = None
        if isinstance(self._ctx, (TerminalNodeImpl, RuleContext)):
            if full_text is None:
                text = self._ctx.getText()
            elif getattr(self._ctx, "start", None) and getattr(self._ctx, "stop", None):
                text = full_text[self._ctx.start.start : self._ctx.stop.stop + 1]
            elif (
                getattr(self._ctx, "symbol", None)
                and getattr(self._ctx.symbol, "start", None)
                and getattr(self._ctx.symbol, "stop", None)
            ):
                text = full_text[self._ctx.symbol.start : self._ctx.symbol.stop + 1]
        if text is None and self.text:
            text = self.text

        return text

    def get_position(self) -> Optional[Dict[str, int]]:
        position = None
        ctx = self._ctx
        if ctx is not None:
            if isinstance(ctx, TerminalNodeImpl):
                position = {
                    "line_start": ctx.symbol.line,
                    "column_start": ctx.symbol.column,
                    "line_end": ctx.symbol.line,
                    "column_end": ctx.symbol.column
                    + (ctx.symbol.stop - ctx.symbol.start),
                }
            elif getattr(ctx, "start", None) and getattr(ctx, "stop", None):
                position = {
                    "line_start": ctx.start.line,
                    "column_start": ctx.start.column,
                    "line_end": ctx.stop.line,
                    "column_end": ctx.stop.column + (ctx.stop.stop - ctx.stop.start),
                }

        return position or self.position

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
    DEBUG = True
    DEBUG_INSTANCES = []

    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, *args, **kwargs)
        if cls.DEBUG:
            cls.DEBUG_INSTANCES.append(instance)
            return instance
        else:
            return args[0][0]

    @classmethod
    def from_text(cls, text: str, ctx: Optional[ParserRuleContext] = None):
        return cls([text], {"value": 0}, {}, ctx)

    def __eq__(self, other):
        return self.value == other

    def __str__(self):
        # currently just used for better formatting in debugger
        return self.value

    def __repr__(self):
        return "'{}'".format(self.value)


class AliasNode(BaseNode, metaclass=AstNodeMeta):
    # TODO: look at AstNode methods
    # defines class properties
    # - as a property name to copy from ANTLR nodes
    # - as a property name defined in terms of (nested) ANTLR node properties
    # the field will be set to the first definition that is not undefined
    _fields_spec = []

    _fields = AstNodeMeta._fields

    # Defines which ANTLR nodes to convert to this node. Elements can be:
    # - a string: uses AstNode._from_fields as visitor
    # - a tuple ('node_name', 'ast_node_class_method_name'): uses ast_node_class_method_name as visitor
    # subclasses use _bind_to_visitor to create visit methods for the nodes in _rules on the ParseTreeVisitor
    # using this information
    _rules = []

    _priority = 1

    _strict = True

    def __init__(self, node: BaseNode, fields: Optional[Dict[str, Any]] = None):
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
    def from_spec(cls, node: BaseNode) -> "AliasNode":
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
    def get_path(cls, node: BaseNode, path: List[str]):
        # TODO: can be defined on FieldNode too
        result = node
        for i in range(len(path)):
            result = getattr(result, path[i], None)
            if result is None:
                break

        return result

    @classmethod
    def bind_to_transformer(
        cls,
        transformer_cls: Type["BaseNodeTransformer"],
        default_transform_method: str = "from_spec",
    ):
        for rule in cls._rules:
            if isinstance(rule, str):
                cls_method = default_transform_method
            else:
                rule, cls_method = rule[:2]
            transformer_method = cls.get_transformer(cls_method)
            bind_to_transformer(transformer_cls, rule, transformer_method)

    @classmethod
    def get_transformer(cls, method_name: str):
        """Get method to bind to visitor"""
        transform_function = getattr(cls, method_name)
        assert callable(transform_function)

        def transformer_method(self, node):
            kwargs = {}
            if inspect.signature(transform_function).parameters.get("helper"):
                kwargs["helper"] = self.helper
            return transform_function(node, **kwargs)

        return transformer_method


class BaseNodeRegistry:
    def __init__(self):
        self.dynamic_node_classes = {}

    def get_cls(self, cls_name: str, field_names: tuple) -> Type[BaseNode]:
        """"""
        if cls_name not in self.dynamic_node_classes:
            self.dynamic_node_classes[cls_name] = BaseNode.create_cls(
                cls_name, field_names
            )
        else:
            existing_cls = self.dynamic_node_classes[cls_name]
            all_fields = tuple(set(existing_cls._fields) | set(field_names))
            if len(all_fields) > len(existing_cls._fields):
                existing_cls._fields = all_fields

        return self.dynamic_node_classes[cls_name]

    def isinstance(self, instance: BaseNode, class_name: str) -> bool:
        """Check if a BaseNode is an instance of a registered dynamic class"""
        if isinstance(instance, BaseNode):
            klass = self.dynamic_node_classes.get(class_name, None)
            if klass:
                return isinstance(instance, klass)
            # Not an instance of a class in the registry
            return False
        else:
            raise TypeError("This function can only be used for BaseNode objects")


# TODO: test: if 'visit' in method, it has to be as 'visit_'
class BaseNodeTransformer(NodeTransformer):
    def __init__(self, registry: BaseNodeRegistry):
        self.helper = TransformerHelper(registry)

    def visit(self, node: BaseNode):
        # TODO: I think transform_  + node.__class__.__name__ would be better/clearer then
        #  as the node methods don't need to do any visiting (which is completely done by visit and generic_visit)
        method = "visit_" + type(node).__name__
        transformer = getattr(self, method, None)

        if transformer is None:
            return self.generic_visit(node)
        else:
            alias = transformer(node)
            if isinstance(alias, AliasNode) or alias == node:
                # this prevents infinite recursion and visiting
                # AliasNodes with a name that is also the name of a BaseNode
                if isinstance(alias, BaseNode):
                    self.generic_visit(alias)
            else:
                # visit BaseNode (e.g. result of Transformer method)
                if isinstance(alias, list):
                    # Transformer method can return array instead of node
                    alias = [
                        self.visit(el) if isinstance(el, BaseNode) else el
                        for el in alias
                    ]  # TODO: test
                elif isinstance(alias, BaseNode):
                    alias = self.visit(alias)

            return alias

    def visit_Terminal(self, terminal: Terminal) -> Terminal:
        """Handle Terminal the same as other non-node types"""
        return terminal

    @classmethod
    def bind_alias_nodes(cls, alias_classes: List[Type[AliasNode]]):
        for item in alias_classes:
            if getattr(item, "_rules", None) is not None:
                item.bind_to_transformer(cls)


def bind_to_transformer(
    transformer_cls: Type[BaseNodeTransformer],
    rule_name: str,
    transformer_method: Callable,
):
    """Assign AST node class constructors to parse tree visitors."""
    setattr(transformer_cls, get_transformer_method_name(rule_name), transformer_method)


def get_transformer_method_name(rule_name: str) -> str:
    return "visit_{}".format(rule_name[0].upper() + rule_name[1:])


class TransformerHelper:
    def __init__(self, registry: BaseNodeRegistry):
        self.registry = registry

    def isinstance(self, *args):
        return self.registry.isinstance(*args)


def get_alias_nodes(items) -> List[Type[AstNode]]:
    return list(
        filter(
            lambda item: inspect.isclass(item) and issubclass(item, AliasNode), items
        )
    )


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
        if (
            len(used_fields) != 1
            or isinstance(tree, AliasNode)
            or (in_list and isinstance(result, list))
        ):
            result = tree
            for field in tree._fields:
                old_value = getattr(tree, field, None)
                if old_value:
                    setattr(
                        result,
                        field,
                        simplify_tree(old_value, unpack_lists=unpack_lists),
                    )
            return result
        assert result is not None
    elif isinstance(tree, list) and len(tree) == 1 and unpack_lists:
        result = tree[0]
    else:
        if isinstance(tree, list):
            result = [
                simplify_tree(el, unpack_lists=unpack_lists, in_list=True)
                for el in tree
            ]
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
     - [done] make compatible with AstNode & AstModule in protowhat (+ shellwhat usage: bashlex + osh parser)
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

    def __init__(self, registry: BaseNodeRegistry):
        self.registry = registry

    def visitChildren(
        self, node: ParserRuleContext, predicate=None, simplify=False
    ) -> BaseNode:
        # children is None if all parts of a grammar rule are optional and absent
        children = [self.visit(child) for child in node.children or []]

        instance = BaseNode.create(node, children, self.registry)

        return instance

    def visitTerminal(self, ctx: ParserRuleContext) -> Terminal:
        """Converts case insensitive keywords and identifiers to lowercase"""
        text = ctx.getText()
        return Terminal.from_text(text, ctx)

    def visitErrorNode(self, node: ErrorNode):
        return None


# ANTLR helpers


def get_field(ctx: ParserRuleContext, field: str):
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


def get_field_references(
    ctx: ParserRuleContext, field_names: List[str], simplify=False
) -> Dict[str, Any]:
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


def materialize(reference_dict: IndexReferences, source: List[Any]) -> Dict[str, Any]:
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


def get_field_names(ctx: ParserRuleContext) -> List[str]:
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


def get_label_names(ctx: ParserRuleContext) -> List[str]:
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
