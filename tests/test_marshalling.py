import importlib
import pytest

from json import dumps, JSONDecoder, load, loads
from antlr_ast.marshalling import AstEncoder, get_decoder


@pytest.fixture
def ast():
    return importlib.import_module("tests.test_expr_ast")


def test_marshalling(ast):
    # Given
    code = "not 2"
    correct_json = load(open("tests/json/test_marshalling/test_marshalling.json"))

    # When
    ast_tree = ast.parse(code)
    json_tree = dumps(ast_tree, cls=AstEncoder)

    # Then
    assert loads(json_tree) == correct_json
    ast_tree = JSONDecoder(object_hook=get_decoder()).decode(json_tree)
    assert isinstance(ast_tree, ast.AstNode)
    assert isinstance(ast_tree.children_by_field["NOT"], str)
    assert isinstance(ast_tree.NOT, str)
    assert isinstance(ast_tree.children_by_field["expr"], ast.AstNode)
    assert isinstance(ast_tree.expr, ast.AstNode)
