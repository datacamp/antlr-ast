import pytest

from antlr_ast import rule_to_visitor_name


@pytest.mark.parametrize(
    "text, result",
    [
        ("Test", "visitTest"),
        ("teSt", "visitTeSt"),
        ("Test_method", "visitTest_method"),
        ("test_method", "visitTest_method"),
        ("Test_Method", "visitTest_Method"),
    ],
)
def test_upper_first(text, result):
    assert rule_to_visitor_name(text) == result
