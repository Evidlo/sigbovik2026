# Evan Widloski - 2026-03-27

# run all lines in target in single shell, quit on error
.ONESHELL:
.SHELLFLAGS = -ec

.PHONY: release
release:
	typst compile main.typ
	gh release create --latest --title "" -n "" main main.pdf