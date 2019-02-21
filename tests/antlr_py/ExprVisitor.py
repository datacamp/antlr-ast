# Generated from tests/Expr.g4 by ANTLR 4.7
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .ExprParser import ExprParser
else:
    from ExprParser import ExprParser

# This class defines a complete generic visitor for a parse tree produced by ExprParser.

class ExprVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by ExprParser#Integer.
    def visitInteger(self, ctx:ExprParser.IntegerContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ExprParser#SubExpr.
    def visitSubExpr(self, ctx:ExprParser.SubExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ExprParser#BinaryExpr.
    def visitBinaryExpr(self, ctx:ExprParser.BinaryExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by ExprParser#NotExpr.
    def visitNotExpr(self, ctx:ExprParser.NotExprContext):
        return self.visitChildren(ctx)



del ExprParser