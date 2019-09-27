import pytest

from antlr_ast.ast import BaseNodeRegistry


def test_base_node_registry_get_cls():
    # Given
    base_node_registry = BaseNodeRegistry()

    # When
    cls1 = base_node_registry.get_cls("cls1", ("field1",))
    cls1_2 = base_node_registry.get_cls("cls1", ("field1", "field2"))

    # Then
    assert set(cls1._fields) == {"field1", "field2"}
    assert set(cls1_2._fields) == {"field1", "field2"}


def test_base_node_registry_isinstance():
    # Given
    base_node_registry = BaseNodeRegistry()

    # When
    Cls1 = base_node_registry.get_cls("cls1", ("field1",))
    Cls1_2 = base_node_registry.get_cls("cls1", ("field1", "field2"))
    Cls2 = base_node_registry.get_cls("cls2", ("field_a", "field_b"))

    cls1_obj = Cls1([], [], [])
    cls1_2_obj = Cls1_2([], [], [])
    cls2_obj = Cls2([], [], [])

    # Then
    assert isinstance(cls1_obj, type(cls1_2_obj))
    assert base_node_registry.isinstance(cls1_obj, "cls1")
    assert base_node_registry.isinstance(cls1_2_obj, "cls1")
    assert base_node_registry.isinstance(cls2_obj, "cls2")
    assert not base_node_registry.isinstance(cls1_obj, "cls2")
    assert not base_node_registry.isinstance(cls1_2_obj, "cls2")
    assert not base_node_registry.isinstance(cls2_obj, "cls1")

    assert not base_node_registry.isinstance(cls2_obj, "cls3")

    with pytest.raises(
        TypeError, match="This function can only be used for BaseNode objects"
    ):
        base_node_registry.isinstance([], "cls1")
