# antlr-ast

[![Build Status](https://travis-ci.org/datacamp/antlr-ast.svg?branch=master)](https://travis-ci.org/datacamp/antlr-ast)
[![codecov](https://codecov.io/gh/datacamp/antlr-ast/branch/master/graph/badge.svg)](https://codecov.io/gh/datacamp/antlr-ast)

This package allows you to use ANTLR grammars and use the parser output to generate an abstract syntax tree (AST).

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

Using `antlr-ast` involves four steps:

1. Using ANTLR to define a grammar and to generate the necessary Python files to parse this grammar
2. Using `parse` to get the ANTLR runtime output based on the generated grammar files
3. Using `process_tree` on the output of the previous step
    1. A `BaseAstVisitor` (customisable by providing a subclass) transforms the ANTLR output to a serializable tree of `BaseNode`s,
       dynamically created based on the rules in the ANTLR grammar
    2. A `BaseNodeTransformer` subclass can be used to transform each kind of node
    3. The simplify option can be used to shorten paths in the tree by skipping nodes that only have a single descendant
4. Using the resulting tree

The next sections go into more detail about these steps.

To visualize the process of creating and transforrming these parse trees, you can use [this ast-viewer](https://github.com/datacamp/ast-viewer).

### Using ANTLR

**Note: For this part of this tutorial you need to know how to parse code**  
See the ANTLR [getting started guide](https://github.com/antlr/antlr4/blob/4.7.2/doc/getting-started.md) if you have never installed ANTLR.  
The [ANTLR Mega Tutorial](https://tomassetti.me/antlr-mega-tutorial/#python-setup) has useful Python examples.

[This page explains how to write ANTLR parser rules](https://github.com/antlr/antlr4/blob/master/doc/parser-rules.md).  
The rule definition below is an example with descriptive names for important ANTLR parser grammar elements:

```g4
rule_name: rule_element? rule_element_label='literal'    #RuleAlternativeLabel
         | TOKEN+                                        #RuleAlternativeLabel
         ;
```

Rule element and alternative labels are optional.  
`+`, `*`, `?`, `|` and `()` have the same meaning as in RegEx.

Below, we'll use a simple grammar to explain how `antlr-ast` works.
This grammar can be found in `/tests/Expr.g4`.

```g4
grammar Expr;

// parser

expr:   left=expr op=('+'|'-') right=expr       #BinaryExpr
    |   NOT expr                                #NotExpr
    |   INT                                     #Integer
    |   '(' expr ')'                            #SubExpr
    ;

// lexer

INT :   [0-9]+ ;         // match integers
NOT :   'not' ;

WS  :   [ \t]+ -> skip ; // toss out whitespace
```

ANTLR can use the grammar above to generate a parser in a number of languages.
To generate a Python parser, you can use the following command.

```bash
antlr4 -Dlanguage=Python3 -visitor /tests/Expr.g4
```

This will generate a number of files in the `/tests/` directory, including a Lexer (`ExprLexer.py`),
a parser (`ExprParser.py`), and a visitor (`ExprVisitor.py`).

You can use and import these directly in Python. For example, from the root of this repo:

```bash
from tests import ExprVisitor
```

To easily use the generated files, they are put in the `antlr_py` package.
The `__init__.py` file exports the generated files under an alias that doesn't include the name of the grammar.

### Base nodes

A `BaseNode` subclass has fields for all rule elements and labels for all rule element labels in its corresponding grammar rule.
Both fields and labels are available as properties on `BaseNode` instances.
Labels take precedence over fields if the names would collide.

The name of a `BaseNode` is the name of the corresponding ANTLR grammar rule, but starting with an uppercase character.
If rule alternative labels are specified for an ANTLR rule, these are used instead of the rule name.

### Transforming nodes

Typically, there is no 1-to-1 mapping between ANTLR rules and the concepts of a language: the rule hierarchy is more nested.
Transformations can be used to make the initial tree of BaseNodes based on ANTLR rules more similar to an AST.

#### Transformer

The `BaseNodeTransformer` will walk over the tree from the root node to the leaf nodes.
When visiting a node, it is possible to transform it.
The tree is updated with transformed node before continuing the walk over the tree.

To define a node transform, add a static method to the `BaseNodeTransformer` subclass passed to `process_tree`.

- The name of the method you should define follows this pattern: `visit_<BaseNode>`,
  where `<BaseNode>` should be replaced by the name of the `BaseNode` subclass to transform.
- The method should return the transformed node.

This is a simple example:

```python
class Transformer(BaseNodeTransformer):
    @staticmethod
    def visit_My_antlr_rule(node):
        return node.name_of_part
```

#### Custom nodes

A custom node can represent a part of the parsed language, a type of node present in an AST.

To make it easy to return a custom node, you can define `AliasNode` subclasses.
Normally, fields of `AliasNode`s are like symlinks to navigate the tree of `BaseNode`s.

Instances of custom nodes are created from a `BaseNode`.
Fields and labels of the source `BaseNode` are also available on the `AliasNode`.
If an `AliasNode` field name collides with these, it takes precedence when accessing that property.

This is what a custom node looks like:

```python
class NotExpr(AliasNode):
    _fields_spec = ["expr", "op=NOT"]
```

This code defines a custom node, `NotExpr` with an `expr` and an `op` field.

##### Field specs

The `_fields_spec` class property is a list that defines the fields the custom node should have.

This is how a field spec in this list is used when creating an custom node from a `BaseNode` (the source node):

- If a field spec does not exist on the source node, it is set to `None`
- If multiple field specs define the same field, the first one that isn't `None` is used
- If a field spec is just a name, it is copied from the source node
- If a field spec is an assignment, the left side is the name of the field on the `AliasNode`
  and the right side is the path that should be taken starting in the source node to get the node
  that should be the value for the field on the custom node.
  Parts of this path are separated using `.`

##### Connecting to the transformer

To use this custom node, add a method to the transformer:

```python
class Transformer(BaseNodeTransformer):
    # ...

    # here the BaseNode name is the same as the custom node name
    # but that isn't required
    @staticmethod
    def visit_NotExpr(node):
        return NotExpr.from_spec(node)
```

Instead of defining methods on the transformer class to use custom nodes, it's possible to do this automatically:

```python
Transformer.bind_alias_nodes(alias_nodes)
```

To make this work, the `AliasNode` classes in the list should have a `_rules` class property
with a list of the `BaseNode` names it should transform.

This is the result:

```python
class NotExpr(AliasNode):
    _fields_spec = ["expr", "op=NOT"]
    _rules = ["NotExpr"]

class Transformer(BaseNodeTransformer):
    pass

alias_nodes = [NotExpr]
Transformer.bind_alias_nodes(alias_nodes)
```

An item in `_rules` can also be a tuple.
In that case, the first item in the tuple is a `BaseNode` name
and the second item is the name of a class method of the custom node.

It's not useful in the example above, but it is equivalent to this:

```python
class NotExpr(AliasNode):
    _fields_spec = ["expr", "op=NOT"]
    _rules = [("NotExpr", "from_not")]

    @classmethod
    def from_not(cls, node):
        return cls.from_spec(node)

class Transformer(BaseNodeTransformer):
    pass

alias_nodes = [NotExpr]
Transformer.bind_alias_nodes(alias_nodes)
```

### Using the final tree

It's easy to use a tree that has a mix of `AliasNode`s and dynamic `BaseNode`s:
the whole tree is just a nested Python object.

When searching nodes in a tree, the priority of nodes can be taken into account.
By default, `BaseNode`s have priority 3 and `AliasNode`s have priority 2.

When writing code to work with trees, it can be affected by changes in the grammar, the transforms and the custom nodes.
The grammar is the most likely to change.

To make grammar updates have no impact on your code, don't rely on `BaseNode`s.
You can still check whether the `AliasNode` parent node of a `BaseNode` has the correct fields set
and search for nested `AliasNode`s in a subtree.

If you do rely on `BaseNode`s, code could break by the addition of `AliasNode`s that replace some of these
if a field name collides with a field name on a used `BaseNode`.
