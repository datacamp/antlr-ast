import pytest

from antlr_ast.ast import rule_to_visitor_name


@pytest.mark.parametrize(
    "text, result",
    [
        ("Test", "visit_Test"),
        ("teSt", "visit_TeSt"),
        ("Test_method", "visit_Test_method"),
        ("test_method", "visit_Test_method"),
        ("Test_Method", "visit_Test_Method"),
    ],
)
def test_upper_first(text, result):
    assert rule_to_visitor_name(text) == result
