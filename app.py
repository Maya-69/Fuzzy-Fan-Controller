import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


def triangular_mf(x, a, b, c):
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_vals = np.atleast_1d(x_arr)

    if not (a <= b <= c):
        raise ValueError("Expected a <= b <= c for triangular membership function")

    mu = np.zeros_like(x_vals, dtype=float)

    if a == b and b == c:
        mu[x_vals == a] = 1.0
    else:
        if a != b:
            rising = (x_vals > a) & (x_vals < b)
            mu[rising] = (x_vals[rising] - a) / (b - a)
        else:
            mu[x_vals == b] = 1.0

        mu[x_vals == b] = 1.0

        if b != c:
            falling = (x_vals > b) & (x_vals < c)
            mu[falling] = (c - x_vals[falling]) / (c - b)
        else:
            mu[x_vals == b] = 1.0

    mu = np.clip(mu, 0.0, 1.0)
    return float(mu[0]) if scalar_input else mu


def trapezoidal_mf(x, a, b, c, d):
    x_arr = np.asarray(x, dtype=float)
    scalar_input = x_arr.ndim == 0
    x_vals = np.atleast_1d(x_arr)

    if not (a <= b <= c <= d):
        raise ValueError("Expected a <= b <= c <= d for trapezoidal membership function")

    mu = np.zeros_like(x_vals, dtype=float)

    if a == b == c == d:
        mu[x_vals == a] = 1.0
    else:
        if a != b:
            rising = (x_vals > a) & (x_vals < b)
            mu[rising] = (x_vals[rising] - a) / (b - a)

        plateau = (x_vals >= b) & (x_vals <= c)
        mu[plateau] = 1.0

        if c != d:
            falling = (x_vals > c) & (x_vals < d)
            mu[falling] = (d - x_vals[falling]) / (d - c)

    mu = np.clip(mu, 0.0, 1.0)
    return float(mu[0]) if scalar_input else mu


def plot_membership_with_marker(title, x_vals, fuzzy_sets, current_value, memberships):
    fig, ax = plt.subplots(figsize=(5.8, 2.9))
    for label, mu in fuzzy_sets.items():
        ax.plot(x_vals, mu, label=label)
        point_mu = memberships.get(label, 0.0)
        ax.scatter([current_value], [point_mu], s=35)
        ax.hlines(point_mu, x_vals[0], current_value, linestyle=":", linewidth=1)

    ax.axvline(current_value, color="black", linestyle="--", linewidth=1.2, label="Current value")
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Membership")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def fuzzify_values(temperature, crowd):
    temp_x = np.linspace(0, 50, 501)
    crowd_x = np.linspace(0, 10, 501)
    speed_x = np.linspace(0, 100, 501)

    temperature_memberships = {
        "Cold": trapezoidal_mf(temperature, 0, 0, 15, 25),
        "Moderate": triangular_mf(temperature, 18, 25, 32),
        "Hot": trapezoidal_mf(temperature, 28, 35, 50, 50),
    }

    crowd_memberships = {
        "Empty": trapezoidal_mf(crowd, 0, 0, 2, 4),
        "Normal": triangular_mf(crowd, 3, 5, 7),
        "Crowded": trapezoidal_mf(crowd, 6, 8, 10, 10),
    }

    speed_sets = {
        "Low": trapezoidal_mf(speed_x, 0, 0, 20, 40),
        "Medium": triangular_mf(speed_x, 30, 50, 70),
        "High": trapezoidal_mf(speed_x, 60, 80, 100, 100),
    }

    temperature_curves = {
        "Cold": trapezoidal_mf(temp_x, 0, 0, 15, 25),
        "Moderate": triangular_mf(temp_x, 18, 25, 32),
        "Hot": trapezoidal_mf(temp_x, 28, 35, 50, 50),
    }

    crowd_curves = {
        "Empty": trapezoidal_mf(crowd_x, 0, 0, 2, 4),
        "Normal": triangular_mf(crowd_x, 3, 5, 7),
        "Crowded": trapezoidal_mf(crowd_x, 6, 8, 10, 10),
    }

    universes = {
        "temperature": temp_x,
        "crowd": crowd_x,
        "speed": speed_x,
    }

    return temperature_memberships, crowd_memberships, speed_sets, temperature_curves, crowd_curves, universes


def render_membership_values(title, value, memberships):
    st.subheader(title)
    st.write(f"Crisp value: {value}")
    for label, mu in memberships.items():
        st.write(f"{label} = {mu:.3f}")


def compute_rule_strength(temp_m, crowd_m, temp_label, crowd_label, operator):
    if operator == "AND":
        return min(temp_m[temp_label], crowd_m[crowd_label])
    if operator == "OR":
        return max(temp_m[temp_label], crowd_m[crowd_label])
    raise ValueError("Operator must be AND or OR")


def get_rules_for_mode(mode):
    if mode == "Energy Saving":
        return [
            ("R1", "Hot", "AND", "Crowded", "High"),
            ("R2", "Hot", "AND", "Normal", "High"),
            ("R3", "Hot", "AND", "Empty", "High"),
            ("R4", "Cold", "AND", "Empty", "Low"),
            ("R5", "Moderate", "AND", "Normal", "Low"),
            ("R6", "Moderate", "AND", "Crowded", "Medium"),
            ("R7", "Hot", "OR", "Crowded", "High"),
        ]
    if mode == "Aggressive":
        return [
            ("R1", "Hot", "AND", "Crowded", "High"),
            ("R2", "Hot", "AND", "Normal", "High"),
            ("R3", "Hot", "AND", "Empty", "High"),
            ("R4", "Cold", "AND", "Empty", "Low"),
            ("R5", "Moderate", "AND", "Normal", "Medium"),
            ("R6", "Moderate", "AND", "Crowded", "High"),
            ("R7", "Hot", "OR", "Crowded", "High"),
        ]
    return [
        ("R1", "Hot", "AND", "Crowded", "High"),
        ("R2", "Hot", "AND", "Normal", "High"),
        ("R3", "Hot", "AND", "Empty", "High"),
        ("R4", "Cold", "AND", "Empty", "Low"),
        ("R5", "Moderate", "AND", "Normal", "Medium"),
        ("R6", "Moderate", "AND", "Crowded", "High"),
        ("R7", "Hot", "OR", "Crowded", "High"),
    ]


def run_mamdani_inference(temp_m, crowd_m, mode):
    rules = get_rules_for_mode(mode)

    results = []
    for name, temp_label, operator, crowd_label, output_label in rules:
        strength = compute_rule_strength(temp_m, crowd_m, temp_label, crowd_label, operator)
        results.append(
            {
                "rule": name,
                "description": f"IF Temp is {temp_label} {operator} Crowd is {crowd_label} THEN Speed is {output_label}",
                "output": output_label,
                "strength": strength,
            }
        )
    return results


def render_rule_results(rule_results):
    st.subheader("Rule Activation Strength")
    for item in rule_results:
        st.write(f"{item['rule']}: {item['description']}")
        st.write(f"Activation = {item['strength']:.3f}")
        st.write(f"Output set = {item['output']}")

    fired = [item for item in rule_results if item["strength"] > 0.0]
    st.subheader("Fired Rules")
    if fired:
        for item in fired:
            st.write(f"{item['rule']} fired with strength {item['strength']:.3f} -> {item['output']}")
    else:
        st.write("No rules fired")


def build_fan_svg_html(speed_value):
    speed_clamped = float(np.clip(speed_value, 0.0, 100.0))
    spin_duration = 4.0 - (speed_clamped / 100.0) * 3.4
    return f"""
    <div class='fan-section'>
        <svg class='fan-svg' viewBox='0 0 320 320' aria-label='fan illustration'>
            <defs>
                <radialGradient id='fanGlow' cx='50%' cy='50%' r='60%'>
                    <stop offset='0%' stop-color='#f8fafc'/>
                    <stop offset='100%' stop-color='#e8f1ff'/>
                </radialGradient>
                <linearGradient id='bladeGrad' x1='0%' y1='0%' x2='100%' y2='100%'>
                    <stop offset='0%' stop-color='#b8f7c6'/>
                    <stop offset='100%' stop-color='#35b46f'/>
                </linearGradient>
                <filter id='softShadow' x='-20%' y='-20%' width='140%' height='140%'>
                    <feDropShadow dx='0' dy='5' stdDeviation='6' flood-color='#0f172a' flood-opacity='0.16'/>
                </filter>
            </defs>
            <circle cx='160' cy='160' r='132' fill='url(#fanGlow)' stroke='#bfd7ff' stroke-width='2' stroke-dasharray='4 7'/>
            <g class='fan-rotor' style='animation-duration:{spin_duration:.2f}s;' filter='url(#softShadow)'>
                <path class='fan-blade' d='M160 160 C135 82, 96 54, 74 76 C49 101, 83 140, 160 160 Z' fill='url(#bladeGrad)'/>
                <path class='fan-blade' d='M160 160 C238 135, 266 96, 244 74 C219 49, 180 83, 160 160 Z' fill='url(#bladeGrad)'/>
                <path class='fan-blade' d='M160 160 C185 238, 224 266, 246 244 C271 219, 237 180, 160 160 Z' fill='url(#bladeGrad)'/>
                <path class='fan-blade' d='M160 160 C82 185, 54 224, 76 246 C101 271, 140 237, 160 160 Z' fill='url(#bladeGrad)'/>
                <circle cx='160' cy='160' r='26' fill='#1d4ed8'/>
                <circle cx='160' cy='160' r='10' fill='#dbeafe'/>
            </g>
        </svg>
        <div class='fan-speed-text'>Current fan speed {speed_clamped:.2f}</div>
    </div>
    """


def aggregate_outputs(rule_results, speed_sets):
    clipped_sets = []
    for item in rule_results:
        clipped = np.minimum(item["strength"], speed_sets[item["output"]])
        clipped_sets.append(clipped)

    if not clipped_sets:
        raise ValueError("No rule outputs to aggregate")

    aggregated = np.maximum.reduce(clipped_sets)
    return aggregated, clipped_sets


def compute_centroid(x_vals, mu_vals):
    denominator = np.sum(mu_vals)
    if denominator == 0:
        return 0.0
    numerator = np.sum(mu_vals * x_vals)
    return float(numerator / denominator)


def plot_aggregated_output(speed_x, aggregated_mu, centroid):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(speed_x, aggregated_mu, color="#1f77b4", linewidth=2, label="Aggregated output")
    ax.fill_between(speed_x, 0, aggregated_mu, color="#1f77b4", alpha=0.25)
    ax.axvline(centroid, color="red", linestyle="--", linewidth=1.5, label=f"Centroid = {centroid:.2f}")

    if len(aggregated_mu) > 0:
        idx = int(np.argmin(np.abs(speed_x - centroid)))
        centroid_mu = float(aggregated_mu[idx])
        ax.hlines(centroid_mu, speed_x[0], centroid, color="red", linestyle=":", linewidth=1)

    ax.set_title("Aggregated Fan Speed Output")
    ax.set_xlabel("Fan Speed")
    ax.set_ylabel("Membership")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def main():
    # this is a comment
    st.set_page_config(page_title="Fuzzy Fan Controller", layout="wide")
    st.markdown(
        """
        <style>
            .stApp {
                background: #ffffff;
                color: #000000;
            }
            .block-container {
                padding-top: 1rem;
                color: #000000;
            }
            .stApp h1,
            .stApp h2,
            .stApp h3,
            .stApp h4,
            .stApp h5,
            .stApp h6,
            .stApp p,
            .stApp div,
            .stApp span,
            .stApp label,
            .stApp li,
            .stApp button {
                color: #000000;
            }
            .panel-card {
                background: #ffffff;
                border: 1px solid #e2e8f0;
                border-radius: 14px;
                padding: 14px;
                box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
                margin-bottom: 12px;
            }
            .step-card {
                background: #ffffff;
                border: 1px solid #dbeafe;
                border-left: 4px solid #2563eb;
                border-radius: 12px;
                padding: 10px 12px;
                margin: 8px 0;
                font-size: 0.95rem;
            }
            .fan-section {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 8px 0 2px 0;
                margin-top: 0;
            }
            .fan-svg {
                width: 260px;
                height: 260px;
                display: block;
            }
            .fan-rotor {
                transform-origin: center center;
                animation-name: spinRotor;
                animation-timing-function: linear;
                animation-iteration-count: infinite;
            }
            .fan-speed-text {
                margin-top: 10px;
                font-weight: 600;
                color: #000000;
                text-align: center;
            }
            @keyframes spinRotor {
                from { transform: rotate(0deg); }
                to { transform: rotate(360deg); }
            }
            div[data-baseweb="select"] > div {
                background-color: #ffffff !important;
                border-color: #d1d5db !important;
                color: #000000 !important;
            }
            div[data-baseweb="select"] [role="combobox"] {
                background-color: #ffffff !important;
            }
            div[data-baseweb="select"] [aria-expanded="true"] {
                background-color: #ffffff !important;
            }
            div[data-baseweb="select"] [data-baseweb="popover"] {
                background-color: #ffffff !important;
            }
            div[data-baseweb="select"] ul,
            div[data-baseweb="select"] li {
                background-color: #ffffff !important;
            }
            div[data-baseweb="select"] * {
                color: #000000 !important;
            }
            .stSlider [data-baseweb="slider"] > div > div {
                background: #ffffff;
            }
            .stSlider [data-baseweb="slider"] {
                background: #ffffff;
                border-radius: 999px;
            }
            .stExpander {
                background: #ffffff;
                border-radius: 12px;
                border: 1px solid #e5e7eb;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Fuzzy Fan Controller")
    st.write("Compact fuzzy fan controller with manual fuzzy logic")

    left_col, center_col, right_col = st.columns([1.25, 0.95, 1.0], gap="medium")

    with right_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.subheader("Controls")
        mode = st.selectbox("Rule mode", ["Energy Saving", "Balanced", "Aggressive"], index=1)
        temperature = st.slider("Temperature", min_value=0.0, max_value=50.0, value=30.0, step=0.5)
        crowd = st.slider("Crowd", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        st.markdown("</div>", unsafe_allow_html=True)

    temp_m, crowd_m, speed_sets, temp_curves, crowd_curves, universes = fuzzify_values(temperature, crowd)
    rule_results = run_mamdani_inference(temp_m, crowd_m, mode)
    aggregated_mu, _ = aggregate_outputs(rule_results, speed_sets)
    final_speed = compute_centroid(universes["speed"], aggregated_mu)

    with center_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        st.subheader("Fan Output")
        st.markdown(build_fan_svg_html(final_speed), unsafe_allow_html=True)
        st.progress(int(np.clip(final_speed, 0.0, 100.0)))
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='step-card'>1. Fuzzification: Temp {temperature:.1f}, Crowd {crowd:.1f}</div>
            <div class='step-card'>2. Rules & Inference: {mode} mode, {len(rule_results)} rules</div>
            <div class='step-card'>3. Aggregation: max composition complete</div>
            <div class='step-card'>4. Defuzzification: centroid gives {final_speed:.2f}</div>
            """,
            unsafe_allow_html=True,
        )

    with left_col:
        st.markdown("<div class='panel-card'>", unsafe_allow_html=True)
        with st.expander("1. Fuzzification", expanded=True):
            render_membership_values("Temperature Membership Values", temperature, temp_m)
            temp_fig = plot_membership_with_marker(
                "Temperature Sets",
                universes["temperature"],
                temp_curves,
                temperature,
                temp_m,
            )
            st.pyplot(temp_fig, width="content")
            plt.close(temp_fig)
            render_membership_values("Crowd Membership Values", crowd, crowd_m)
            crowd_fig = plot_membership_with_marker(
                "Crowd Sets",
                universes["crowd"],
                crowd_curves,
                crowd,
                crowd_m,
            )
            st.pyplot(crowd_fig, width="content")
            plt.close(crowd_fig)

        with st.expander("2. Rules & Inference", expanded=False):
            render_rule_results(rule_results)

        with st.expander("3. Aggregation", expanded=False):
            st.write("Rules are combined with max composition across clipped output sets.")
            agg_fig = plot_aggregated_output(universes["speed"], aggregated_mu, final_speed)
            st.pyplot(agg_fig, width="content")
            plt.close(agg_fig)

        with st.expander("4. Defuzzification", expanded=False):
            st.write(f"Centroid method output: {final_speed:.2f}")
            st.write("The value above is the crisp fan speed used by the center fan animation.")
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()