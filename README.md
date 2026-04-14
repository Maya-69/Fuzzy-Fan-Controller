# Fuzzy Logic Fan Controller

This project is a Streamlit web app that controls fan speed using manual fuzzy logic.

The app takes two crisp inputs:

- Temperature (0 to 50)
- Crowd (0 to 10)

It computes one crisp output:

- Fan Speed (0 to 100)

## Stack

- Python
- Streamlit
- NumPy
- Matplotlib

## Core rules for implementation

- Membership functions are implemented manually
- Defuzzification is implemented manually
- No built-in fuzzy helper functions are used

## Fuzzy system design

### Input 1: Temperature

- Cold
- Moderate
- Hot

### Input 2: Crowd

- Empty
- Normal
- Crowded

### Output: Fan Speed

- Low
- Medium
- High

## How it works

### 1. Membership functions

The app uses custom functions:

- `triangular_mf(x, a, b, c)`
- `trapezoidal_mf(x, a, b, c, d)`

These functions support scalar and array input through NumPy.

### 2. Fuzzification

The function `fuzzify_values(temperature, crowd)` computes:

- Membership degrees for temperature and crowd
- Membership curves for plotting
- Output universe and output fuzzy sets

### 3. Rule base and inference

The function `get_rules_for_mode(mode)` defines fuzzy rules for:

- Energy Saving
- Balanced
- Aggressive

The function `run_mamdani_inference(...)` evaluates all rules with:

- `min` for AND
- `max` for OR

Hot temperature rules are designed so high temperature drives high fan speed regardless of crowd level.

### 4. Aggregation

The function `aggregate_outputs(rule_results, speed_sets)`:

- Clips each output set by rule strength
- Aggregates all clipped sets using max composition

### 5. Defuzzification

The function `compute_centroid(x_vals, mu_vals)` calculates crisp fan speed using centroid:

$$
z^* = \frac{\sum \mu(z) \cdot z}{\sum \mu(z)}
$$

### 6. UI

The app layout is compact and white themed:

- Left panel: collapsible sections for fuzzification, rules, aggregation, and defuzzification
- Center panel: animated 4-blade fan with progress bar
- Right panel: controls for mode, temperature, and crowd

## Project files

- `app.py`: full application logic and UI
- `README.md`: project documentation

## Run instructions

Install dependencies:

```bash
pip install streamlit numpy matplotlib
```
or
```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Notes

- The app is fully manual fuzzy logic implementation.
- All key fuzzy pipeline steps are visible in the UI.
