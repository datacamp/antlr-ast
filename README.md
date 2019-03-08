# antlr-ast

[![Build Status](https://travis-ci.org/datacamp/antlr-ast.svg?branch=master)](https://travis-ci.org/datacamp/antlr-ast)

This package allows you to take an Antlr4 parser output, and generate an abstract syntax tree (AST).

Antlr4 parser outputs are often processed using the [visitor pattern](https://en.wikipedia.org/wiki/Visitor_pattern).
By using `antlr-ast`'s `AstNode`, you can define how an antlr visitor should generate an AST.

## Install

```bash
pip install antlr-ast
```

**Note:** this package is not python2 compatible.

## Running Tests

```bash
# may need:
# pip install pytest
py.test
```

## Usage

**Note: This tutorial assumes you know how to parse code, and create a python visitor.**
See the Antlr4 [getting started guide](https://github.com/antlr/antlr4/blob/4.7.2/doc/getting-started.md) if you have never installed Antlr.
The [Antlr Mega Tutorial](https://tomassetti.me/antlr-mega-tutorial/#python-setup) has useful python examples.

Using `antlr-ast` involves three steps:

1. Using Antlr to generate the necessary python files for a grammar.
2. Shaping parsed output to an AST by subclassing `antlr-ast`'s AstNode.
3. Connecting everything to a Visitor class.

### Example Grammar

Below, we'll use a simple grammar to explain how `antlr-ast` works.
This grammar can be found in `/tests/Expr.g4`.

```g4
grammar Expr;

expr:   left=expr op=('+'|'-') right=expr       #BinaryExpr
    |   NOT expr                                #NotExpr
    |   INT                                     #Integer
    |   '(' expr ')'                            #SubExpr
    ;

INT :   [0-9]+ ;         // match integers
NOT :   'not' ;

WS  :   [ \t]+ -> skip ; // toss out whitespace
```

Antlr4 can use the grammar above to generate a parser in a number of languages.
To generate a python parser, you can use the following command.

```bash
antlr4 -Dlanguage=Python3 -visitor /tests/Expr.g4
```

This will generate a number of files in the `/tests/` directory, including a Lexer (`ExprLexer.py`),
a parser (`ExprParser.py`), and a visitor (`ExprVisitor.py`).

You can use and import these directly in python. For example, from the root of this repo...

```bash
from tests import ExprVisitor
```

### Creating AST Classes

By subclassing AstNode, we can define how Antlr4 should shape parser output to our desired AST.
For example, `/tests/test_expr_ast.py` defines the following classes for the grammar above.

```python
from antlr_ast import AstNode

class SubExpr(AstNode):
    _fields = ['expr->expression']

class BinaryExpr(AstNode):
    _fields = ['left', 'right', 'op']

class NotExpr(AstNode):
    _fields = ['NOT->op', 'expr']
```

Note that `_fields` allows you to redefine rule names.
For example, in the grammar, the rule for matching expressions is named "expr".
In this case, using `'expr->expression'` says that the expr field should be called "expression" instead.

### Connecting to Visitor

Once you have defined your AST classes, you can connect them to the visitor, using their `_from_fields` method.
For example..

```python
AstVisitor(ExprVisitor):                                  # antlr visitor subclass
    def visitBinaryExpr(self, ctx):                       # method on visitor
        return BinaryExpr._from_fields(self, ctx)         # _from_fields method
```

Then, whenever the visitor passes through a BinaryExpr node in the parse tree, it will convert it to the BinaryExpr AST Node.

Note that the parse tree has a node named `BinaryExpr`, with the fields `left`, `right`, and `expr`, due to this line of the grammar:

```g4
expr:   left=expr op=('+'|'-') right=expr       #BinaryExpr
```

For a full example, see the parse function of `/tests/test_expr_ast.py`.

### Unshaped nodes

When parts of the parse tree do not have instructions for mapping to an AST node, they are returned as an `Unshaped` node.
