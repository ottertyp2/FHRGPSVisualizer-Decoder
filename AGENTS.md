# AGENTS.md

## Mandatory startup and finish checklist

Every agent instance working in this repository must do these checks explicitly:

- Read this `AGENTS.md` before making changes.
- Before finishing a shareable task, review `git status` and `git remote -v`.
- If the task is ready to share, commit only the intended files.
- Push the exact same branch/commit through `origin` so both GitHub and GitLab receive it.
- If the worktree is dirty with unrelated changes, do not silently skip sync; report the situation clearly.
- If one remote push fails, say which host failed and which host succeeded.

## Purpose

This repository hosts an educational offline GPS L1 C/A decoder and visualizer.
Agents working here should preserve the project's core priorities:

- clear and didactic DSP implementation
- responsive GUI for offline inspection
- visual separation of PRN- and satellite-specific evidence through the workflow
- clean separation between GUI, DSP, models, and tests
- support for large recordings through windowed reads and streaming
- keep ambiguous real-world captures diagnosable through sample-rate surveys and segment-consistency evidence
- no PVT or position solution unless explicitly added later

## Project structure

- `app/main.py`: Qt application entry point
- `app/gui/`: main window, tabs, and worker plumbing
- `app/dsp/`: IQ I/O, PRN generation, acquisition, tracking, bit sync, navigation decode, and benchmarking
- `app/models/`: shared dataclasses for GUI and DSP state
- `app/tests/`: unit and smoke tests

## Management rules for agents

- Keep the app runnable with `python -m app.main`.
- Prefer understandable implementations over black-box optimizations.
- Keep GUI code and DSP code separate; do not bury signal-processing logic inside widgets.
- When handling large recordings, support both RAM-preload and bounded-window workflows; keep the user-visible mode clear in the GUI.
- Treat `complex64` little-endian IQ as the current baseline input format unless the repo is explicitly extended.
- Preserve the real-capture diagnosis path: sample-rate hypotheses, repeated-segment evidence, and per-PRN explanations should stay understandable in the GUI.
- Do not add position solving, maps, or full PVT implicitly.
- Preserve the benchmark path so laptop suitability remains visible inside the GUI.
- Keep PRN-specific acquisition, tracking, and navigation evidence understandable and visually separated when possible.
- Add or update tests for new DSP behavior whenever practical.

## Git and release hygiene

- Do not commit local recordings, caches, or generated binary artifacts.
- Keep `README.md`, `requirements.txt`, and this file aligned with major feature changes.
- If a change affects workflow or repository conventions, update `AGENTS.md`.
- Prefer small, reviewable commits with clear messages.
- Treat GitHub and GitLab as first-class remotes for this project and keep them in sync.
- After finishing a change that is ready to share, agents should commit the intended files and push the same branch/commit to both hosted repositories.
- Prefer keeping `origin` configured with both push URLs so one clean `git push origin <branch>` updates GitHub and GitLab together.
- Do not push partial, broken, or unrelated work just to satisfy the sync rule; stage and commit only the files that belong to the task being delivered.
- If one remote push fails, report it clearly so the repository state can be corrected instead of assuming both hosts were updated.
- Use `powershell -ExecutionPolicy Bypass -File tools/check_git_sync.ps1` as a quick pre-push verification when needed.

## Large-file policy

- Recorded IQ captures such as `.bin` files are local analysis assets and should stay out of Git by default.
- Any sample data committed for demos should be small, synthetic, or compressed enough for source control.
