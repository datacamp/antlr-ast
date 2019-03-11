# Changelog

All notable changes to this project will be documented in this file.

## v0.5.0

- Rewrite as a staged approach
  - autodetection of ANTLR fields and labels: Unshaped doesn’t exist anymore
  - more isolated steps: enables serialization (and possibility to let ANTLR parser run in a separate service)
  - more powerful definition of reshaped node (= AliasNode) fields using tree paths
  - easier definition of transforms (simplification & AliasNodes) (no ANTLR API knowledge needed, no other visiting in transformation methods)
  - more shared code

## v0.4.2

### Added

- Add parameter to `parse` function to set a custom error listener (or remove the default listener)

## v0.4.1

- Fix setup.py

## v0.4.0

## Changed

- Better package structure

## v0.3.0

### Added

- Helper to handle case sensitivity during lexing of ANTLR grammar

### Changed

- The fields for AstNode subclasses are now defined in `_fields_spec` instead of `_fields` so `_fields` is now compatible with how the `ast` module defines it.
- `parse()` doesn't accept a visitor but returns the parsed input.
