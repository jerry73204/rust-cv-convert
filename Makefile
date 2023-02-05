.PHONY: default test generate

default:
	@echo 'Please run `make generate` or `make test`.'

test:
	cargo run --bin cv-convert-generate --release -- test

generate:
	cargo run --bin cv-convert-generate --release -- generate --manifest-dir cv-convert
