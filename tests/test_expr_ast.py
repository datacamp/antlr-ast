from antlr_ast.ast import AliasNode, parse as parse_ast, process_tree

from . import grammar


class SubExpr(AliasNode):
    _fields_spec = ["expression=expr"]


class BinaryExpr(AliasNode):
    _fields_spec = ["left", "right", "op"]


class NotExpr(AliasNode):
    _fields_spec = ["op=NOT", "expr"]


class Transformer:
    def visit_BinaryExpr(self, node):
        return BinaryExpr.from_spec(node)

    def visit_SubExpr(self, node):
        return SubExpr.from_spec(node)

    def visit_NotExpr(self, node):
        return NotExpr.from_spec(node)

    def visit_Terminal(self, node):
        return node.get_text()


def parse(text, start="expr", **kwargs):
    antlr_tree = parse_ast(grammar, text, start, upper=False, **kwargs)
    simple_tree = process_tree(antlr_tree, Transformer)

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
