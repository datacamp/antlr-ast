from functools import partial
from json import JSONEncoder, JSONDecoder

from antlr_ast.ast import BaseNode, BaseNodeRegistry, Terminal


class AstEncoder(JSONEncoder):
    """JSON encoder for BaseNodes"""

    def default(self, o):
        if isinstance(o, Terminal):
            encoded = str(o)
        elif isinstance(o, BaseNode):
            encoded = {
                "@type": o.__class__.__name__,
                "@fields": o._fields,
                "@position": o.get_position(),
                "@text": o.get_text(),
                "field_references": o._field_references,
                "label_references": o._label_references,
                "children": o.children,
            }
        else:
            encoded = o
        return encoded


def decode_ast(registry, ast_json):
    """JSON decoder for BaseNodes"""
    if ast_json.get("@type"):
        subclass = registry.get_cls(ast_json["@type"], tuple(ast_json["@fields"]))
        children = [
            decode_ast(registry, child) if isinstance(child, dict) else child
            for child in ast_json["children"]
        ]
        return subclass(
            children,
            ast_json["field_references"],
            ast_json["label_references"],
            position=ast_json.get("@position", None),
            text=ast_json.get("@text", None),
        )
    else:
        return ast_json


def get_decoder(registry=None):
    """Get a JSON decoding hook that shares a dynamic node registry between decoding calls"""
    if registry is None:
        registry = BaseNodeRegistry()
    return partial(decode_ast, registry)
