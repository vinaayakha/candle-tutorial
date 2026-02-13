This is a Rust project using the Candle ML framework (HuggingFace). Not a JS/Bun project.

## Build

- Use `make` for all common operations. Never use raw cargo commands.
- `make check` — check all exercises compile (fast, no codegen)
- `make test Q=04` — test a specific exercise
- `make test-all` — test all exercises
- `make run Q=04b` — run a specific exercise
- `make list` — list all exercises

## Disk Management

- Shared target dir at `~/.cargo/shared-target` (set in `~/.cargo/config.toml`)
- `make size` — show shared target size
- `make sweep` — remove artifacts older than 7 days
- `make clean` — full clean
- Never create a local `target/` directory. The shared target handles it.

## Project Structure

- `exercises/` — interview prep exercises (Cargo examples). Each file is `q{N}_{topic}.rs`.
- `src/` — reference implementations (Qwen3, RoBERTa, XLM-RoBERTa models)
- Exercises are registered as `[[example]]` entries in `Cargo.toml`

## Conventions

- Exercises use `todo!()` as stubs — they compile but panic until solved
- Each exercise has `#[cfg(test)] mod tests` with unit tests
- Grading: read the user's solution, run `make test Q=XX`, grade as interviewer
- When creating new exercises, always register them in `Cargo.toml` as `[[example]]`
