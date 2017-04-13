#!/usr/bin/env python

import re
import ast
from setuptools import setup

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('antlr_ast.py', 'rb') as f:
    version = str(ast.literal_eval(_version_re.search(
        f.read().decode('utf-8')).group(1)))

setup(
	name = 'antlr-ast',
	version = version,
	py_modules= ['antlr_ast'],
	install_requires = ['antlr4-python3-runtime'],
        description = 'AST shaping for antlr parsers',
        author = 'Michael Chow',
        author_email = 'michael@datacamp.com',
        url = 'https://github.com/datacamp/antlr-ast')
