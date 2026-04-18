#!/home/alanxw/.local/share/mamba/envs/graphics/bin/python
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import streamlit as st


@dataclass
class Cue:
    name: str
    alpha: float = 0.3
    V: float = 0.0


@dataclass
class Trial:
    cues_present: List[str]
    us_present: bool = True
    lambda_val: float = 1.0


@dataclass
class SimResult:
    trial_labels: List[str]
    histories: Dict[str, List[float]] = field(default_factory=dict)


def run_simulation(cues: Dict[str, Cue], trials: List[Trial], beta: float) -> SimResult:
    result = SimResult(trial_labels=[])
    for name in cues:
        result.histories[name] = []

    for trial in trials:
        lam = trial.lambda_val if trial.us_present else 0.0
        V_total = sum(cues[c].V for c in trial.cues_present if c in cues)
        prediction_error = lam - V_total

        for cue_name in trial.cues_present:
            if cue_name in cues:
                cue = cues[cue_name]
                delta = cue.alpha * beta * prediction_error
                cue.V += delta

        label_parts = "+".join(trial.cues_present)
        us_label = "+" if trial.us_present else "-"
        result.trial_labels.append(f"{label_parts}{us_label}")

        for name, cue in cues.items():
            result.histories[name].append(cue.V)

    return result


def make_acquisition(n_trials=25, alpha=0.3, beta=0.5):
    cues = {"CS": Cue("CS", alpha=alpha)}
    trials = [Trial(cues_present=["CS"], us_present=True)] * n_trials
    return cues, trials, beta


def make_extinction(n_acq=20, n_ext=20, alpha=0.3, beta=0.5):
    cues = {"CS": Cue("CS", alpha=alpha)}
    trials = [Trial(cues_present=["CS"], us_present=True)] * n_acq + [
        Trial(cues_present=["CS"], us_present=False)
    ] * n_ext
    return cues, trials, beta


def make_blocking(n_phase1=15, n_phase2=15, alpha_a=0.4, alpha_b=0.4, beta=0.5):
    cues = {"A": Cue("A", alpha=alpha_a), "B": Cue("B", alpha=alpha_b)}
    trials = [Trial(cues_present=["A"], us_present=True)] * n_phase1 + [
        Trial(cues_present=["A", "B"], us_present=True)
    ] * n_phase2
    return cues, trials, beta


def make_overshadowing(n_trials=25, alpha_a=0.6, alpha_b=0.2, beta=0.5):
    cues = {
        "A (salient)": Cue("A (salient)", alpha=alpha_a),
        "B (weak)": Cue("B (weak)", alpha=alpha_b),
    }
    trials = [
        Trial(cues_present=["A (salient)", "B (weak)"], us_present=True)
    ] * n_trials
    return cues, trials, beta


def make_conditioned_inhibition(n_acq=20, n_inh=20, alpha_a=0.4, alpha_x=0.4, beta=0.5):
    cues = {"A": Cue("A", alpha=alpha_a), "X": Cue("X", alpha=alpha_x)}
    trials = [Trial(cues_present=["A"], us_present=True)] * n_acq + [
        Trial(cues_present=["A", "X"], us_present=False)
    ] * n_inh
    return cues, trials, beta


PRESETS = {
    "Simple Acquisition": make_acquisition,
    "Acquisition + Extinction": make_extinction,
    "Blocking": make_blocking,
    "Overshadowing": make_overshadowing,
    "Conditioned Inhibition": make_conditioned_inhibition,
}

COLORS = [
    "#2176AE",
    "#E05263",
    "#57A773",
    "#F2A541",
    "#8B5CF6",
    "#06B6D4",
    "#F472B6",
    "#A3E635",
]

DEFAULT_TRIAL_TEXT = {
    "Simple Acquisition": "25 CS +",
    "Acquisition + Extinction": "20 CS +\n20 CS -",
    "Blocking": "15 A +\n15 A+B +",
    "Overshadowing": "25 A (salient)+B (weak) +",
    "Conditioned Inhibition": "20 A +\n20 A+X -",
}

DEFAULT_CUES = {
    "Simple Acquisition": [("CS", 0.3)],
    "Acquisition + Extinction": [("CS", 0.3)],
    "Blocking": [("A", 0.4), ("B", 0.4)],
    "Overshadowing": [("A (salient)", 0.6), ("B (weak)", 0.2)],
    "Conditioned Inhibition": [("A", 0.4), ("X", 0.4)],
}

DEFAULT_BETA = 0.5


def parse_trials(
    cue_defs: List[Tuple[str, float]], trial_text: str, beta: float
) -> Tuple[Dict[str, Cue], List[Trial], float]:
    cues: Dict[str, Cue] = {}
    for name, alpha in cue_defs:
        name = name.strip()
        if name:
            cues[name] = Cue(name=name, alpha=alpha)

    if not cues:
        raise ValueError("Add at least one cue.")

    trials: List[Trial] = []
    raw = trial_text.strip()
    if not raw:
        raise ValueError("Enter at least one trial phase.")

    for line_no, line in enumerate(raw.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            raise ValueError(
                f"Line {line_no}: expected '<n> <cues> <+/->', got: {line}"
            )
        try:
            n = int(parts[0])
        except ValueError:
            raise ValueError(
                f"Line {line_no}: first value must be an integer, got '{parts[0]}'"
            )
        us_str = parts[-1]
        if us_str not in ("+", "-"):
            raise ValueError(
                f"Line {line_no}: last value must be '+' or '-', got '{us_str}'"
            )
        cue_str = " ".join(parts[1:-1])
        cue_names = [c.strip() for c in cue_str.split("+") if c.strip()]

        for cn in cue_names:
            if cn not in cues:
                raise ValueError(
                    f"Line {line_no}: cue '{cn}' not defined. "
                    f"Available cues: {list(cues.keys())}"
                )

        for _ in range(n):
            trials.append(Trial(cues_present=cue_names, us_present=(us_str == "+")))

    if not trials:
        raise ValueError("No valid trials parsed.")

    return cues, trials, beta


def make_plot(result: SimResult):
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(1, len(result.trial_labels) + 1))

    for idx, (name, hist) in enumerate(result.histories.items()):
        color = COLORS[idx % len(COLORS)]
        ax.plot(x, hist, label=name, color=color, linewidth=2)

    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Associative Strength (V)", fontsize=11)
    ax.set_title("Rescorla-Wagner Simulation", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    prev = None
    for i, lbl in enumerate(result.trial_labels):
        if prev is not None and lbl != prev:
            ax.axvline(i + 0.5, color="#888", linewidth=1, linestyle=":")
        prev = lbl

    fig.tight_layout()
    return fig


st.set_page_config(page_title="Rescorla-Wagner Simulator", layout="wide")
st.title("Rescorla-Wagner Model Simulator")

st.latex(r"\Delta V_i = \alpha_i \cdot \beta \cdot (\lambda - \Sigma V)")
st.caption(
    "**α** = CS salience · **β** = US learning rate · "
    "**λ** = max conditioning (1 if US present, 0 if not) · "
    "**ΣV** = total prediction from all cues present"
)

with st.sidebar:
    st.header("Parameters")

    preset = st.selectbox("Preset Scenario", list(PRESETS.keys()))

    # Reset widget state when preset changes
    if "prev_preset" not in st.session_state:
        st.session_state.prev_preset = preset
    if st.session_state.prev_preset != preset:
        st.session_state.prev_preset = preset
        defaults = DEFAULT_CUES[preset]
        for i in range(8):
            st.session_state.pop(f"cue_name_{i}", None)
            st.session_state.pop(f"cue_alpha_{i}", None)
        st.session_state.pop("num_cues", None)
        st.session_state.pop("trial_text", None)
        st.rerun()

    beta = st.slider("β (US learning rate)", 0.01, 1.0, DEFAULT_BETA, 0.01)

    st.subheader("Cues")
    defaults = DEFAULT_CUES[preset]
    num_cues = st.number_input(
        "Number of cues", min_value=1, max_value=8, value=len(defaults),
        key="num_cues",
    )

    cue_defs: List[Tuple[str, float]] = []
    for i in range(int(num_cues)):
        col1, col2 = st.columns([2, 1])
        default_name = defaults[i][0] if i < len(defaults) else f"Cue{i + 1}"
        default_alpha = defaults[i][1] if i < len(defaults) else 0.3
        with col1:
            name = st.text_input(
                f"Cue {i + 1} name", value=default_name, key=f"cue_name_{i}"
            )
        with col2:
            alpha = st.slider(
                f"α {i + 1}", 0.01, 1.0, default_alpha, 0.01, key=f"cue_alpha_{i}"
            )
        cue_defs.append((name, alpha))

    st.subheader("Trial Sequence")
    st.caption(
        "One phase per line: `<n> <cue1+cue2...> <+/->`  \n"
        "Example: `20 A+B +` (20 trials, A & B, US present)"
    )
    trial_text = st.text_area(
        "Trials", value=DEFAULT_TRIAL_TEXT[preset], height=150, key="trial_text"
    )

run = st.button("Run Simulation", type="primary", use_container_width=True)

if run:
    try:
        cues, trials, beta_val = parse_trials(cue_defs, trial_text, beta)
        result = run_simulation(cues, trials, beta_val)
        fig = make_plot(result)
        st.pyplot(fig)
        plt.close(fig)

        st.subheader("Final Associative Strengths")
        summary = {name: f"{hist[-1]:.4f}" for name, hist in result.histories.items()}
        st.table(summary)

    except ValueError as e:
        st.error(str(e))
else:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Associative Strength (V)", fontsize=11)
    ax.set_title("Rescorla-Wagner Simulation", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
