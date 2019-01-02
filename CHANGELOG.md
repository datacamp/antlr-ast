# Changelog

All notable changes to this project will be documented in this file.

## v0.3.0

### Added

- Helper to handle case sensitivity during lexing of ANTLR grammar

### Changed

- The fields for AstNode subclasses are now defined in `_fields_spec` instead of `_fields` so `_fields` is now compatible with how the `ast` module defines it.
- `parse()` doesn't accept a visitor but returns the parsed input.
