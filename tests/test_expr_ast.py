from antlr_ast import AstNode

from tests.ExprVisitor import ExprVisitor

class SubExpr(AstNode):
    _fields = ['expr->expression']

class BinaryExpr(AstNode):
    _fields = ['left', 'right', 'op']

class NotExpr(AstNode):
    _fields = ['NOT->op', 'expr']

class AstVisitor(ExprVisitor):
    
    def visitBinaryExpr(self, ctx):
        return BinaryExpr._from_fields(self, ctx)

    def visitSubExpr(self, ctx):
        return SubExpr._from_fields(self, ctx)

    def visitNotExpr(self, ctx):
        return NotExpr._from_fields(self, ctx)

    def visitTerminal(self, ctx):
        return ctx.getText()

def parse(text, start='expr', strict=False):
    from antlr4.InputStream import InputStream
    from antlr4 import FileStream, CommonTokenStream

    from tests.ExprLexer import ExprLexer
    from tests.ExprParser import ExprParser

    input_stream = InputStream(text)

    lexer = ExprLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = ExprParser(token_stream)
    ast = AstVisitor()     

    return ast.visit(getattr(parser, start)())

def test_binary():
    node = parse('1 + 2')
    assert isinstance(node, BinaryExpr)
    assert node.left == '1'
    assert node.right == '2'
    assert node.op == '+'

def test_not():
    node = parse('not 2')
    assert isinstance(node, NotExpr)
    assert node.expr == '2'

def test_subexpr():
    node = parse('(1 + 1)')
    assert isinstance(node.expression, BinaryExpr)

# Speaker ---------------------------------------------------------------------

import pytest
from antlr_ast import Speaker

def test_speaker_default():
    speaker = Speaker(nodes =  {'BinaryExpr': 'binary expression'},
                      fields = {'left': 'left part'})

    node = parse('1 + 1')
    str_tmp = "The {field_name} of the {node_name}"

    assert speaker.describe(node, str_tmp, 'left') == \
                str_tmp.format(field_name = 'left part', node_name = 'binary expression')

def test_speaker_node_cfg():
    node_cnfg = {'name': 'binary expression', 'fields': {'left': 'left part'}}

    speaker = Speaker(nodes =  {'BinaryExpr': node_cnfg},
                      fields = {'left': 'should not occur!'})

    node = parse('1 + 1')
    str_tmp = "The {field_name} of the {node_name}"

    assert speaker.describe(node, str_tmp, 'left') == \
                str_tmp.format(field_name = 'left part', node_name = 'binary expression')


