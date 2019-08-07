from antlr_ast.ast import (
    AliasNode,
    BaseNode as AstNode,  # used in other tests
    parse as parse_ast,
    process_tree,
    BaseNodeTransformer,
    Terminal,
)

from . import grammar


class SubExpr(AliasNode):
    _fields_spec = ["expression=expr"]


class BinaryExpr(AliasNode):
    _fields_spec = ["left", "right", "op"]


class NotExpr(AliasNode):
    _fields_spec = ["op=NOT", "expr"]


class Transformer(BaseNodeTransformer):
    def visit_BinaryExpr(self, node):
        return BinaryExpr.from_spec(node)

    def visit_SubExpr(self, node):
        return SubExpr.from_spec(node)

    def visit_NotExpr(self, node):
        return NotExpr.from_spec(node)


def parse(text, start="expr", **kwargs):
    antlr_tree = parse_ast(grammar, text, start, upper=False, **kwargs)
    simple_tree = process_tree(antlr_tree, transformer_cls=Transformer)

    return simple_tree


def test_binary():
    node = parse("1 + 2")
    assert isinstance(node, BinaryExpr)
    assert node.left == "1"
    assert node.right == "2"
    assert node.op == "+"


def test_not():
    node = parse("not 2")
    assert isinstance(node, NotExpr)
    assert node.expr == "2"


def test_subexpr():
    node = parse("(1 + 1)")
    assert isinstance(node, SubExpr)
    assert isinstance(node.expression, BinaryExpr)
    assert isinstance(node.expression.left, Terminal)


def test_fields():
    assert NotExpr._fields == ("op", "expr")
    not_expr = parse("not 2")
    assert not_expr._fields == ("op", "expr")

    assert SubExpr._fields == ("expression",)
    sub_expr = parse("(1 + 1)")
    assert sub_expr._fields == ("expression",)


# Speaker ---------------------------------------------------------------------

from antlr_ast.ast import Speaker


def test_speaker_default():
    speaker = Speaker(
        nodes={"BinaryExpr": "binary expression"}, fields={"left": "left part"}
    )

    node = parse("1 + 1")
    str_tmp = "The {field_name} of the {node_name}"

    assert speaker.describe(node, str_tmp, "left") == str_tmp.format(
        field_name="left part", node_name="binary expression"
    )


def test_speaker_node_cfg():
    node_cnfg = {"name": "binary expression", "fields": {"left": "left part"}}

    speaker = Speaker(
        nodes={"BinaryExpr": node_cnfg}, fields={"left": "should not occur!"}
    )

    node = parse("1 + 1")
    str_tmp = "The {field_name} of the {node_name}"

    assert speaker.describe(node, str_tmp, "left") == str_tmp.format(
        field_name="left part", node_name="binary expression"
    )


# BaseNode.get_position -------------------------------------------------------


def test_get_position():
    # Given
    code = "1 + (2 + 2)"
    correct_position = {
        "line_start": 1,
        "column_start": 4,
        "line_end": 1,
        "column_end": 10,
    }

    # When
    result = parse(code)
    positions = result.right.get_position()

    # Then
    assert positions["line_start"] == correct_position["line_start"]
    assert positions["line_end"] == correct_position["line_end"]
    assert positions["column_start"] == correct_position["column_start"]
    assert positions["column_end"] == correct_position["column_end"]


def test_terminal_get_position():
    # Given
    code = "(2 + 2) + 1"
    correct_position = {
        "line_start": 1,
        "column_start": 10,
        "line_end": 1,
        "column_end": 10,
    }

    # When
    result = parse(code)
    positions = result.right.get_position()

    # Then
    assert positions["line_start"] == correct_position["line_start"]
    assert positions["line_end"] == correct_position["line_end"]
    assert positions["column_start"] == correct_position["column_start"]
    assert positions["column_end"] == correct_position["column_end"]
