# AGENTS.md

## Purpose

This repository hosts an educational offline GPS L1 C/A decoder and visualizer.
Agents working here should preserve the project's core priorities:

- clear and didactic DSP implementation
- responsive GUI for offline inspection
- clean separation between GUI, DSP, models, and tests
- support for large recordings through windowed reads and streaming
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
- When handling large recordings, avoid loading whole files into memory when a streamed or windowed approach is possible.
- Treat `complex64` little-endian IQ as the current baseline input format unless the repo is explicitly extended.
- Do not add position solving, maps, or full PVT implicitly.
- Preserve the benchmark path so laptop suitability remains visible inside the GUI.
- Add or update tests for new DSP behavior whenever practical.

## Git and release hygiene

- Do not commit local recordings, caches, or generated binary artifacts.
- Keep `README.md`, `requirements.txt`, and this file aligned with major feature changes.
- If a change affects workflow or repository conventions, update `AGENTS.md`.
- Prefer small, reviewable commits with clear messages.

## Large-file policy

- Recorded IQ captures such as `.bin` files are local analysis assets and should stay out of Git by default.
- Any sample data committed for demos should be small, synthetic, or compressed enough for source control.
