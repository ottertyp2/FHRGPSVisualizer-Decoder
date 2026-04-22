# GPS L1 C/A Offline Decoder GUI

An educational offline analysis tool for recorded GPS L1 C/A IQ data. The first version focuses on:

- file loading for `complex64` IQ (`float32 I + float32 Q`)
- raw signal, spectrum, waterfall, and IQ-plane visualization
- PRN acquisition over Doppler and code phase
- segment-consistency scoring so repeated weak PRN hits can be separated from one-off noise peaks
- automatic survey of common GNSS sample-rate hypotheses for uncertain captures
- simple tracking with Early/Prompt/Late correlators
- 50 bps bit extraction from 1 ms prompt integrations
- LNAV preamble detection, word sync, and parity checks
- laptop benchmark for large-recording suitability and bottleneck detection
- per-PRN satellite-oriented views so acquisition, tracking, and decoded bits can be inspected separately
- configurable RAM loading with full-source preload, RAM status display, warnings, and progress dialog

This version intentionally does **not** compute position, pseudoranges, or a full PVT solution.

The current default assumption is a `6 MSa/s` little-endian `complex64` recording. That is generally enough for GPS L1 C/A offline acquisition and tracking; if decoding is still weak, the limiting factors are more likely IF/search-center assumptions, front-end filtering, or signal strength than raw sample-rate alone.

For ambiguous recordings, the acquisition tab can now auto-survey common GNSS sample rates such as `2.046 MSa/s`, `4.092 MSa/s`, and `6.000 MSa/s`, then rank which hypothesis produces the most repeatable PRN evidence across the file.

## Large recordings

- The GUI is designed to work on very large recordings by reading only selected windows for display.
- Acquisition loads only the short segment it actually needs.
- Tracking can stream 1 ms blocks directly from the file instead of loading the whole recording into RAM.
- For multi-gigabyte files, a RAM disk can still help throughput, but the tool does not require the entire file to fit in memory.
- The benchmark measures windowed reads, sequential streaming, FFT throughput, acquisition cost, and tracking cost.
- The slowest measured subsystem is marked as the bottleneck and compared against 6 MSa/s.

## RAM loading modes

- The session tab shows a RAM status line with planned load size and available system RAM.
- `Preload full window to RAM` can be enabled to load the complete source before DSP starts.
- If disabled, the tool loads only the selected analysis window.
- Large RAM loads trigger a warning and use a progress dialog with cancel support.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

## Example workflow

1. Open the app.
2. Load `test3min.bin`, `test1.bin`, or generate a demo signal.
3. Start with the default `6 MSa/s` sample rate unless you know the capture used a different rate.
4. Preview a selected window.
5. Run acquisition for a PRN.
6. Optionally run `Auto Detect Capture` to compare common sample-rate hypotheses and pick the most repeatable one.
7. Optionally scan PRNs 1..32 to rank likely visible satellites.
8. Start tracking for the selected PRN from the acquisition tab or tracking tab.
9. Decode navigation bits and inspect LNAV word sync for that PRN.
10. Run the benchmark to estimate how well the laptop handles large recordings.

## Notes on input

- Supported v1 file format: little-endian `complex64`
- That means each sample is stored as `float32 real` + `float32 imag`
- The file importer shows the estimated sample count and duration from the chosen sample rate

## Project layout

- `app/main.py`: application entry point
- `app/gui/`: Qt main window, tabs, and worker wrappers
- `app/dsp/`: I/O, acquisition, tracking, bit sync, nav decode, and demo signal generation
- `app/models/`: dataclasses for session and processing results
- `app/tests/`: unit and smoke tests

## Where to extend next

- `app/dsp/navdecode.py`: add full LNAV field parsing and ephemeris decoding
- `app/dsp/tracking.py`: add stronger lock metrics, CN0 estimation, and more refined DLL/FLL/PLL tuning
- `app/models/session.py`: extend session state for pseudorange observables
- future `app/dsp/pvt.py`: add satellite position, clock correction, pseudorange solution, and PVT
