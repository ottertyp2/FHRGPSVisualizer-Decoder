# GPS L1 C/A Offline Decoder GUI

An educational offline analysis tool for recorded GPS L1 C/A IQ data. The first version focuses on:

- file loading for `complex64` IQ (`float32 I + float32 Q`)
- raw signal, spectrum, waterfall, and IQ-plane visualization
- PRN acquisition over Doppler and code phase
- simple tracking with Early/Prompt/Late correlators
- 50 bps bit extraction from 1 ms prompt integrations
- LNAV preamble detection, word sync, and parity checks
- laptop benchmark for large-recording suitability and bottleneck detection

This version intentionally does **not** compute position, pseudoranges, or a full PVT solution.

## Large recordings

- The GUI is designed to work on very large recordings by reading only selected windows for display.
- Acquisition loads only the short segment it actually needs.
- Tracking can stream 1 ms blocks directly from the file instead of loading the whole recording into RAM.
- For multi-gigabyte files, a RAM disk can still help throughput, but the tool does not require the entire file to fit in memory.
- The benchmark measures windowed reads, sequential streaming, FFT throughput, acquisition cost, and tracking cost.
- The slowest measured subsystem is marked as the bottleneck and compared against 6 MSa/s.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m app.main
```

## Example workflow

1. Open the app.
2. Load `test1.bin` or generate a demo signal.
3. Enter the sample rate and center frequency manually.
4. Preview a selected window.
5. Run acquisition for a PRN.
6. Start tracking from the best acquisition candidate.
7. Decode navigation bits and inspect LNAV word sync.
8. Run the benchmark to estimate how well the laptop handles large recordings.

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
