import pytest

from antlr_ast.ast import AliasNode, parse_field_spec


def test_double_field():
    class Test(AliasNode):
        _fields_spec = ["x=a", "x=b"]

    assert Test._fields == ("x",)


def test_spec_parse():
    field_spec_str = "x=a.b"
    field_spec = parse_field_spec(field_spec_str)

    assert field_spec.name == "x"
    assert field_spec.origin == ["a", "b"]
    assert field_spec == ("x", ["a", "b"])
