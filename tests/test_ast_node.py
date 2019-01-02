import pytest

from antlr_ast import AstNode, parse_field_spec


def test_double_field():
    class Test(AstNode):
        _fields_spec = ["a->x", "b->x"]

    assert Test._fields == ["x"]


def test_spec_parse():
    field_spec_str = "a->x"
    field_spec = parse_field_spec(field_spec_str)

    assert field_spec.name == "x"
    assert field_spec.origin == "a"
    assert field_spec == ("x", "a")
