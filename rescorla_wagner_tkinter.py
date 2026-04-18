#!/home/alanxw/.local/share/mamba/envs/graphics/bin/python
import tkinter as tk
from tkinter import ttk, messagebox
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


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

    for t_idx, trial in enumerate(trials):
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


# PRESET SCENARIOS ------------------------------------------------------------
# all preset functions return: cues dict, trials list, beta value, and a description string
def make_acquisition(n_trials=25, alpha=0.3, beta=0.5):
    cues = {"CS": Cue("CS", alpha=alpha)}
    trials = [Trial(cues_present=["CS"], us_present=True)] * n_trials
    return cues, trials, beta, "Simple Acquisition"


def make_extinction(n_acq=20, n_ext=20, alpha=0.3, beta=0.5):
    cues = {"CS": Cue("CS", alpha=alpha)}
    trials = [Trial(cues_present=["CS"], us_present=True)] * n_acq + [
        Trial(cues_present=["CS"], us_present=False)
    ] * n_ext
    return cues, trials, beta, "Acquisition then Extinction"


def make_blocking(n_phase1=15, n_phase2=15, alpha_a=0.4, alpha_b=0.4, beta=0.5):
    cues = {
        "A": Cue("A", alpha=alpha_a),
        "B": Cue("B", alpha=alpha_b),
    }
    trials = [Trial(cues_present=["A"], us_present=True)] * n_phase1 + [
        Trial(cues_present=["A", "B"], us_present=True)
    ] * n_phase2
    return cues, trials, beta, "Blocking (A+ then AB+)"


def make_overshadowing(n_trials=25, alpha_a=0.6, alpha_b=0.2, beta=0.5):
    cues = {
        "A (salient)": Cue("A (salient)", alpha=alpha_a),
        "B (weak)": Cue("B (weak)", alpha=alpha_b),
    }
    trials = [
        Trial(cues_present=["A (salient)", "B (weak)"], us_present=True)
    ] * n_trials
    return cues, trials, beta, "Overshadowing (AB+, alpha_A > alpha_B)"


def make_conditioned_inhibition(n_acq=20, n_inh=20, alpha_a=0.4, alpha_x=0.4, beta=0.5):
    cues = {
        "A": Cue("A", alpha=alpha_a),
        "X": Cue("X", alpha=alpha_x),
    }
    trials = [Trial(cues_present=["A"], us_present=True)] * n_acq + [
        Trial(cues_present=["A", "X"], us_present=False)
    ] * n_inh
    return cues, trials, beta, "Conditioned Inhibition (A+ then AX-)"


PRESETS = {
    "Simple Acquisition": make_acquisition,
    "Acquisition + Extinction": make_extinction,
    "Blocking": make_blocking,
    "Overshadowing": make_overshadowing,
    "Conditioned Inhibition": make_conditioned_inhibition,
}

# Colors for the graphig
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


# MAIN TK INTERFACE CLASS
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Rescorla-Wagner Model Simulator")
        self.geometry("1100x780")
        self.minsize(900, 650)

        # top controls -------------------------------------------------
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Label(top, text="Preset Scenario:").pack(side="left")
        self.preset_var = tk.StringVar(value="Simple Acquisition")
        preset_menu = ttk.Combobox(
            top,
            textvariable=self.preset_var,
            values=list(PRESETS.keys()),
            state="readonly",
            width=30,
        )
        preset_menu.pack(side="left", padx=6)
        preset_menu.bind("<<ComboboxSelected>>", self._load_preset)

        ttk.Button(top, text="Run Simulation", command=self._run).pack(
            side="left", padx=12
        )
        ttk.Button(top, text="Clear", command=self._clear).pack(side="left")

        pane = ttk.PanedWindow(self, orient="horizontal")
        pane.pack(fill="both", expand=True, padx=6, pady=4)

        left = ttk.Frame(pane, padding=6)
        pane.add(left, weight=1)

        bf = ttk.LabelFrame(left, text="US Learning Rate (beta)", padding=6)
        bf.pack(fill="x", pady=4)
        self.beta_var = tk.DoubleVar(value=0.5)
        self._make_slider(bf, self.beta_var, 0.01, 1.0)

        cf = ttk.LabelFrame(left, text="Cues (CS stimuli)", padding=6)
        cf.pack(fill="both", expand=True, pady=4)

        btn_row = ttk.Frame(cf)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="+ Add Cue", command=self._add_cue_row).pack(
            side="left"
        )
        ttk.Button(btn_row, text="- Remove Last", command=self._remove_cue_row).pack(
            side="left", padx=4
        )

        self.cue_canvas = tk.Canvas(cf, highlightthickness=0)
        self.cue_inner = ttk.Frame(self.cue_canvas)
        sb = ttk.Scrollbar(cf, orient="vertical", command=self.cue_canvas.yview)
        self.cue_canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.cue_canvas.pack(fill="both", expand=True, pady=4)
        self.cue_canvas.create_window((0, 0), window=self.cue_inner, anchor="nw")
        self.cue_inner.bind(
            "<Configure>",
            lambda e: self.cue_canvas.configure(
                scrollregion=self.cue_canvas.bbox("all")
            ),
        )

        self.cue_rows: List[Dict] = []

        tf = ttk.LabelFrame(left, text="Trial Sequence", padding=6)
        tf.pack(fill="both", expand=True, pady=4)
        ttk.Label(
            tf,
            text=(
                "One phase per line:  <n_trials> <cue1+cue2...> <+/->\n"
                "Example:  20 A+B +     (20 trials, cues A & B, US present)\n"
                "          15 A -       (15 trials, cue A alone, no US)"
            ),
            justify="left",
            font=("Consolas", 9),
        ).pack(anchor="w")
        self.trial_text = tk.Text(tf, height=6, width=40, font=("Consolas", 10))
        self.trial_text.pack(fill="both", expand=True, pady=4)

        right = ttk.Frame(pane, padding=4)
        pane.add(right, weight=3)

        self.fig = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.fmt_xdata = lambda x: f"{x:.1f}"
        self.ax.fmt_ydata = lambda y: f"{y:.4f}"
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, right)
        toolbar.update()

        # info section at the bottom
        info = ttk.LabelFrame(self, text="Model Equation", padding=6)
        info.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Label(
            info,
            text=(
                "Rescorla-Wagner rule:   dV_i  =  alpha_i * beta * (lambda - sumV)        "
                "where  alpha = CS salience,  beta = US learning rate,  "
                "lambda = max conditioning (1 if US, 0 if not),  sumV = total prediction"
            ),
        ).pack(anchor="w")

        self._clear()
        self._load_preset()

    def _add_cue_row(self, name="", alpha=0.3):
        row = ttk.Frame(self.cue_inner)
        row.pack(fill="x", pady=2)
        name_var = tk.StringVar(value=name)
        alpha_var = tk.DoubleVar(value=alpha)
        ttk.Label(row, text="Name:").pack(side="left")
        ttk.Entry(row, textvariable=name_var, width=12).pack(side="left", padx=2)
        ttk.Label(row, text="  alpha:").pack(side="left")
        self._make_slider(row, alpha_var, 0.01, 1.0, length=120)
        self.cue_rows.append(dict(name_var=name_var, alpha_var=alpha_var, frame=row))

    def _remove_cue_row(self):
        if self.cue_rows:
            info = self.cue_rows.pop()
            info["frame"].destroy()

    def _clear_cue_rows(self):
        while self.cue_rows:
            self._remove_cue_row()

    @staticmethod
    def _make_slider(parent, var, from_, to, length=160):
        # create a horizontal slider with a label showing the current value
        frame = ttk.Frame(parent)
        frame.pack(side="left", padx=4)
        label = ttk.Label(frame, text=f"{var.get():.2f}", width=5)
        scale = ttk.Scale(
            frame,
            variable=var,
            from_=from_,
            to=to,
            length=length,
            command=lambda v: label.configure(text=f"{float(v):.2f}"),
        )
        scale.pack(side="left")
        label.pack(side="left")

    def _load_preset(self, event=None):
        name = self.preset_var.get()
        factory = PRESETS.get(name)
        if factory is None:
            return
        cues, trials, beta, _ = factory()

        self.beta_var.set(beta)
        self._clear_cue_rows()
        for cue in cues.values():
            self._add_cue_row(name=cue.name, alpha=cue.alpha)

        self.trial_text.delete("1.0", "end")
        phases: List[Tuple[int, List[str], bool]] = []
        for t in trials:
            key = (tuple(t.cues_present), t.us_present)
            if phases and (tuple(phases[-1][1]), phases[-1][2]) == key:
                phases[-1] = (phases[-1][0] + 1, phases[-1][1], phases[-1][2])
            else:
                phases.append((1, list(t.cues_present), t.us_present))
        for count, cs, us in phases:
            cs_str = "+".join(cs)
            us_str = "+" if us else "-"
            self.trial_text.insert("end", f"{count} {cs_str} {us_str}\n")

    def _parse_trials(self) -> Tuple[Dict[str, Cue], List[Trial], float]:
        beta = self.beta_var.get()

        cues: Dict[str, Cue] = {}
        for row in self.cue_rows:
            n = row["name_var"].get().strip()
            if not n:
                continue
            cues[n] = Cue(name=n, alpha=row["alpha_var"].get())

        if not cues:
            raise ValueError("Add at least one cue.")

        trials: List[Trial] = []
        raw = self.trial_text.get("1.0", "end").strip()
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

    def _run(self):
        try:
            cues, trials, beta = self._parse_trials()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        result = run_simulation(cues, trials, beta)
        self._plot(result)

    def _plot(self, result: SimResult):
        self.ax.clear()
        x = list(range(1, len(result.trial_labels) + 1))

        for idx, (name, hist) in enumerate(result.histories.items()):
            color = COLORS[idx % len(COLORS)]
            self.ax.plot(x, hist, label=name, color=color, linewidth=2)

        self.ax.set_xlabel("Trial", fontsize=11)
        self.ax.set_ylabel("Associative Strength (V)", fontsize=11)
        self.ax.set_title("Rescorla-Wagner Simulation", fontsize=13, fontweight="bold")
        self.ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        self.ax.legend(fontsize=10)
        self.ax.grid(True, alpha=0.3)

        prev = None
        for i, lbl in enumerate(result.trial_labels):
            if prev is not None and lbl != prev:
                self.ax.axvline(i + 0.5, color="#888", linewidth=1, linestyle=":")
            prev = lbl

        self.fig.tight_layout()
        self.canvas.draw()

    def _clear(self):
        self.ax.clear()
        self.ax.set_xlabel("Trial")
        self.ax.set_ylabel("Associative Strength (V)")
        self.ax.set_title("Rescorla-Wagner Simulation")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()


if __name__ == "__main__":
    app = App()
    app.mainloop()
