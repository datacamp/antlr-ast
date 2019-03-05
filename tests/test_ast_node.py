import pytest

from antlr_ast.ast import AliasNode, parse_field_spec


def test_double_field():
    class Test(AliasNode):
        _fields_spec = ["x=a", "x=b"]

    assert Test._fields == ("x",)


@pytest.mark.parametrize("field_spec_str, field_spec", [
    ("x", ("x", ["x"])),
    ("x=a", ("x", ["a"])),
    ("x=a.b", ("x", ["a", "b"])),
    ("x = a", ("x", ["a"])),
    ("x= a ", ("x", ["a"])),
])
def test_spec_parse(field_spec_str, field_spec):
    spec = parse_field_spec(field_spec_str)

    assert spec.name == field_spec[0]
    assert spec.origin == field_spec[1]
    assert spec == field_spec
