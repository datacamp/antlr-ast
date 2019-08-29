import importlib

import pytest


@pytest.fixture
def ast():
    return importlib.import_module("tests.test_expr_ast")


def test_marshalling(ast):
    from json import dumps, JSONDecoder
    from antlr_ast.marshalling import AstEncoder, get_decoder

    ast_tree = ast.parse("not 2")

    json_tree = dumps(ast_tree, cls=AstEncoder)
    ast_tree = JSONDecoder(object_hook=get_decoder()).decode(json_tree)
    assert isinstance(ast_tree, ast.AstNode)
    assert isinstance(ast_tree.children_by_field["NOT"], str)
    assert isinstance(ast_tree.NOT, str)
    assert isinstance(ast_tree.children_by_field["expr"], ast.AstNode)
    assert isinstance(ast_tree.expr, ast.AstNode)
