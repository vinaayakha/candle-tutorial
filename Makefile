# Candle Tutorial — Interview Prep Makefile
# Usage: make help

SHELL := /bin/bash

# ── Exercise Targets ─────────────────────────────────────────────────────────

# Check all exercises compile (fast, no codegen)
check:
	cargo check --examples

# Run a specific exercise: make run Q=01
run:
	@test -n "$(Q)" || (echo "Usage: make run Q=01" && exit 1)
	@example=$$(ls exercises/q$(Q)*.rs 2>/dev/null | head -1 | sed 's|exercises/||;s|\.rs||'); \
	if [ -z "$$example" ]; then echo "No exercise matching q$(Q)*"; exit 1; fi; \
	echo "Running $$example..."; \
	cargo run --example "$$example"

# Test a specific exercise: make test Q=01
test:
	@test -n "$(Q)" || (echo "Usage: make test Q=01" && exit 1)
	@example=$$(ls exercises/q$(Q)*.rs 2>/dev/null | head -1 | sed 's|exercises/||;s|\.rs||'); \
	if [ -z "$$example" ]; then echo "No exercise matching q$(Q)*"; exit 1; fi; \
	echo "Testing $$example..."; \
	cargo test --example "$$example"

# Test all exercises
test-all:
	cargo test --examples

# List all exercises
list:
	@echo "Available exercises:"
	@ls exercises/*.rs | sed 's|exercises/||;s|\.rs||' | while read f; do \
		echo "  $$f"; \
	done

# ── Cleanup / Disk Management ────────────────────────────────────────────────

# Remove build artifacts older than 7 days (shared target)
sweep:
	cargo sweep --time 7 ~/.cargo/shared-target 2>/dev/null || \
	cargo sweep --time 7

# Full clean of this project's cached artifacts
clean:
	cargo clean

# Show shared target size
size:
	@du -sh ~/.cargo/shared-target 2>/dev/null || echo "No shared target dir"

# ── Help ─────────────────────────────────────────────────────────────────────

help:
	@echo "Candle Tutorial — Interview Prep"
	@echo ""
	@echo "Exercises:"
	@echo "  make check        Check all exercises compile"
	@echo "  make run Q=04     Run exercise q04"
	@echo "  make run Q=04b    Run exercise q04b"
	@echo "  make test Q=04    Test exercise q04"
	@echo "  make test-all     Test all exercises"
	@echo "  make list         List all exercises"
	@echo ""
	@echo "Disk management:"
	@echo "  make size         Show shared target dir size"
	@echo "  make sweep        Remove artifacts older than 7 days"
	@echo "  make clean        Full clean"

.PHONY: check run test test-all list sweep clean size help
