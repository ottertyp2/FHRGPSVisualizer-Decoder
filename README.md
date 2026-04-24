# GPS L1 C/A Offline Decoder GUI

An educational desktop tool for inspecting recorded GPS L1 C/A IQ data offline.

## Critical Repo Sync Policy

This repository treats shareable work as incomplete until it is committed and pushed through `origin` to both GitHub and GitLab.

- Local edits alone are not a finished delivery.
- If the work is ready to share, run tests, review `git status`, commit the intended files, and push.
- If commit or push is intentionally skipped, that must be said explicitly.

The project is built to make the decoding chain understandable end to end:

- inspect a recorded IQ file without loading the whole capture blindly
- visualize raw signal, spectrum, waterfall, and IQ behavior
- acquire GPS PRNs over Doppler and code phase
- separate believable satellite evidence from one-off noise peaks
- track a selected PRN with readable loop-state plots
- extract 50 bps navigation bits and inspect LNAV framing/parity
- solve WGS-84 receiver position from valid satellite positions and pseudoranges
- benchmark whether a laptop can keep up with large recordings

The application remains focused on offline signal analysis and diagnosis. It now includes the tested least-squares PVT core needed for a receiver position solution, while map display and network-assisted positioning remain out of scope.

## Why This Tool Exists

Many GNSS examples jump quickly from a binary file to a black-box "lock" result. This app is aimed at the opposite workflow: show the evidence at each step, keep PRN-specific results visually separated, and make uncertain real-world captures diagnosable instead of opaque.

That means the GUI favors:

- didactic DSP over overly clever implementations
- per-satellite acquisition, tracking, and navigation evidence
- sample-rate and search-center hypothesis testing for ambiguous recordings
- workflows that still work on multi-gigabyte captures

## Current Scope

Supported well today:

- little-endian `complex64` IQ recordings (`float32 I` + `float32 Q`)
- offline preview of selected windows
- acquisition for one PRN or PRN scan across `1..32`
- repeated-segment consistency scoring for weak detections
- automatic survey of common GNSS sample-rate hypotheses
- IF / search-center sweeps when the capture is not true baseband
- simple tracking with Early/Prompt/Late correlators and user-selectable long tracking windows
- 1 ms prompt integrations and 20 ms bit decisions
- LNAV preamble detection, word sync, and parity checks
- WGS-84 ECEF/LLA conversion and least-squares position solving from four or more pseudoranges
- benchmark of file I/O, FFT, acquisition, and tracking throughput

Out of scope for now:

- live SDR streaming
- non-`complex64` input as a first-class GUI path
- map display or network-assisted GNSS
- claiming a position fix when navigation parity / ephemeris evidence is insufficient

## Installation

The app runs with Python and a small Qt/NumPy stack.

### PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Dependencies

`requirements.txt` currently includes:

- `numpy`
- `scipy`
- `PySide6`
- `pyqtgraph`
- `pytest`

Optional acceleration:

- install a CuPy build that matches your CUDA runtime if you want GPU-accelerated FFT-heavy paths on supported Windows systems
- on Windows without a full CUDA Toolkit install, the app can also use pip-installed NVIDIA runtime packages such as `nvidia-cuda-nvrtc-cu12`, `nvidia-cuda-runtime-cu12`, `nvidia-cufft-cu12`, and `nvidia-nvjitlink-cu12`
- without CuPy, the app stays fully functional and falls back to multi-core CPU execution
- if GPU runtime pieces are incomplete or fail at runtime, the app falls back to CPU execution instead of aborting the DSP step

## Running The App

```powershell
python -m app.main
```

If `test3min.bin` or `test1.bin` is present in the working directory, the GUI pre-fills that file path on startup to make first use easier.

## Running Tests

```powershell
pytest app/tests
```

The test suite includes unit coverage for DSP helpers plus GUI smoke tests.

## Development Workflow

This repository expects shareable changes to be committed and pushed to both GitHub and GitLab through `origin`.

```powershell
powershell -ExecutionPolicy Bypass -File tools/check_git_sync.ps1
```

Use that check before pushing when you want a quick confirmation that the worktree status and dual-push remote setup look correct.

## Compute Acceleration

The app can now auto-detect logical CPU cores and an optional CuPy/CUDA GPU backend.

- `Auto` compute mode prefers a usable optional GPU for FFT-heavy paths and otherwise uses the CPU
- CPU worker count defaults to logical cores minus one to keep the GUI responsive
- acquisition scans, sample-rate surveys, center sweeps, spectrum averaging, and waterfall FFT rows can use internal parallel workers
- tracking remains single-threaded on purpose because its loop state is sequential and easier to reason about that way

## Input Format

The main GUI workflow currently assumes:

- sample type: little-endian `complex64`
- layout: `float32` real followed by `float32` imaginary
- signal family: GPS L1 C/A

Default working assumptions in the UI:

- sample rate: `6.061 MSa/s` (`200e6 / 33 = 6060606.0606 Sa/s` for the latest X310 sample files)
- center frequency: `1575.42 MHz`
- signal mode: baseband unless you specify a nonzero IF / search center

The sample-rate field is freely editable, so you can enter exact recorder values instead of rounding to a nearby preset. Those defaults are practical starting points, not universal truth. If a real capture behaves strangely, the app provides tools to test other sample-rate and IF hypotheses instead of forcing one interpretation.

## First-Run Workflow

1. Launch the app with `python -m app.main`.
2. Open a `.bin` or `.dat` IQ file, or generate the built-in demo signal.
3. In `File / Session`, set sample rate, signal mode, start sample, window size, and tracking duration.
   For the latest real sample, start near `6.061 MSa/s` and use `Auto Detect Capture` to refine nearby recorder-clock offsets.
4. Click `Preview` to inspect a bounded window before committing to heavier DSP steps.
5. Go to `Acquisition` and run either a single-PRN acquisition or a PRN scan.
6. If the capture is uncertain, use `Auto Detect Capture` or `Sweep Search Center`.
7. Track the highlighted PRN once acquisition shows a repeated candidate; acquisition alone is not proof of a real satellite.
8. Decode bits and inspect LNAV framing in `Bits / Navigation`.
9. Run `Benchmark` if you want a quick laptop suitability estimate for larger files.

## What Each Tab Is For

### File / Session

This is the control center for loading data and choosing how much of the source is read:

- file path and metadata preview
- sample-rate, baseband/IF, and window controls
- tracking duration for long navigation-bit captures
- RAM preload policy
- preview magnitude plot
- session log and worker progress

### Raw Signal

Use this tab to sanity-check the selected window in the time domain before acquisition.

### Spectrum / Waterfall

Use this view to look for occupied bandwidth, DC behavior, and whether the selected acquisition hypothesis matches visible spectral structure.

### IQ Plane

This helps spot clipping, bias, unexpected constellations, and the change from raw IQ to tracked channel views.

### Acquisition

This tab is the main diagnosis surface for initial satellite detection:

- code phase vs Doppler heatmap
- best-candidate tables
- PRN scan table for `1..32`
- repeated-segment evidence text
- sample-rate hypothesis ranking
- IF / center-frequency sweep ranking

The app deliberately treats repeated evidence across segments as more meaningful than a single high peak.

### Tracking

This view shows how one selected PRN behaves after acquisition:

- prompt I/Q
- Early/Prompt/Late magnitudes
- code and carrier error traces
- Doppler and code-frequency estimates
- lock metric

### Bits / Navigation

This view stays PRN-specific and shows:

- 1 ms prompt values
- 20 ms bit accumulations
- hard bit decisions
- LNAV preamble detections
- word labels, parity results, and bit/hex summaries

### PVT Core

The DSP layer includes a tested least-squares WGS-84 solver for four or more satellite pseudoranges. A real position fix should only be trusted after the navigation path has enough parity-valid ephemeris and timing evidence for the selected PRNs.

### Benchmark

The benchmark estimates how well the current machine handles the workload by measuring:

- file/window reading
- sequential streaming
- FFT throughput
- acquisition cost
- tracking cost

The slowest measured component is highlighted as the bottleneck and compared against the current session sample rate target.

## Large Recording Behavior

The app is designed to stay usable on large recordings.

- Preview and plotting operate on bounded windows.
- Acquisition loads only the short segment it needs unless the chosen mode requires more.
- Tracking can work from streamed or bounded source data rather than requiring the entire file to be resident.
- Full-file RAM preload is optional and clearly exposed in the GUI.
- Large RAM loads show warnings and a progress dialog with cancel support.

In practice, this means you can use the tool in two different styles:

- preload mode: load the whole source once, then work quickly from RAM
- bounded-window mode: inspect and process only the selected window to reduce memory pressure

## Diagnosing Weak Or Ambiguous Captures

This repository puts unusual emphasis on diagnosis instead of pretending every file is clean.

When acquisition is weak:

- run `Scan PRNs 1-32` to see whether any satellite stands out consistently
- use `Auto Detect Capture` to compare common sample-rate hypotheses
- use `Sweep Search Center` if the recording may have a residual IF or incorrect search center
- look at repeated-segment consistency, not only the best raw metric
- use the spectrum and IQ views to check whether the recording itself looks plausible

A longer file alone does not guarantee success. Wrong sample-rate assumptions, wrong IF/baseband assumptions, front-end filtering, low SNR, or repeated artifacts can all dominate the outcome.

## Demo And Local Sample Files

The app can generate a synthetic demo signal for a self-contained workflow test.

If local files such as `test1.bin` or `test3min.bin` are present, they can be used as convenient offline examples. Large real captures should remain local analysis assets and should not be committed to Git by default.

## Project Layout

- `app/main.py`: Qt application entry point
- `app/gui/`: main window, worker plumbing, and tab widgets
- `app/gui/tabs/`: session, visualization, acquisition, tracking, navigation, and benchmark tabs
- `app/dsp/`: IQ I/O, PRN generation, acquisition, tracking, bit sync, navigation decode, PVT solving, benchmark logic, and demo generation
- `app/models/`: shared dataclasses for GUI and DSP state
- `app/tests/`: DSP tests and GUI smoke tests

## Development Notes

When extending the project, please keep these repository goals intact:

- keep the app runnable with `python -m app.main`
- keep GUI code separate from DSP logic
- preserve readable, didactic implementations
- keep large-file workflows explicit and understandable
- keep PRN-specific evidence visually separated when possible
- add or update tests for DSP behavior when practical

If a change materially alters workflow or project scope, update `README.md`, `requirements.txt`, and `AGENTS.md` together.
