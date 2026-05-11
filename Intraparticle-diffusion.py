"""
============================================================
INTRAPARTICLE DIFFUSION MODEL (WEBER–MORRIS)
============================================================

Model:
    qt = kWM * t^(1/2) + C

Criteria:
    Only points with qt/qe < 0.3 are used (initial diffusion stage)

Outputs:
    - Linear regression parameters
    - R² and RMSE
    - Plot for each temperature
    - Excel summary file

Author: [Dr. Francisca L. Aranda]
============================================================
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


# ------------------------------------------------------------
# 1. EXPERIMENTAL DATA
# ------------------------------------------------------------
t = np.array([ ], dtype=float)

data = {
    "20°C": np.array([ ]),
    "30°C": np.array([ ]),
    "40°C": np.array([ ]),
    "50°C": np.array([ ])
}


# ------------------------------------------------------------
# 2. WEBER–MORRIS MODEL FUNCTION
# ------------------------------------------------------------
def weber_morris_fit(t, qt, label):
    qe = qt[-1]
    sqrt_t = np.sqrt(t)

    ratio = qt / qe
    mask = (t > 0) & (ratio < 0.3)

    x = sqrt_t[mask]
    y = qt[mask]

    if len(x) < 2:
        print(f"[WARNING] {label}: insufficient points for fitting")
        return None, None

    # Linear regression
    reg = linregress(x, y)

    kWM = reg.slope
    C = reg.intercept
    r2 = reg.rvalue**2

    y_pred = kWM * x + C
    rmse = np.sqrt(np.mean((y - y_pred)**2))

    results = {
        "Temperature": label,
        "qe (mg/g)": qe,
        "Points used": len(x),
        "kWM": kWM,
        "C": C,
        "R²": r2,
        "RMSE": rmse
    }

    # Data table
    table = pd.DataFrame({
        "t (min)": t,
        "sqrt(t)": sqrt_t,
        "qt (mg/g)": qt,
        "qt/qe": ratio,
        "Used": mask
    })

    return results, table


# ------------------------------------------------------------
# 3. PLOTTING FUNCTION
# ------------------------------------------------------------
def plot_weber_morris(t, qt, results, table, label):
    sqrt_t = np.sqrt(t)

    used = table["Used"].values

    plt.figure(figsize=(6,4))
    plt.scatter(sqrt_t, qt, label="Experimental data")
    plt.scatter(sqrt_t[used], qt[used], label="Selected points")

    x_line = np.linspace(min(sqrt_t[used]), max(sqrt_t[used]), 100)
    y_line = results["kWM"] * x_line + results["C"]

    plt.plot(x_line, y_line, label="Linear fit")

    plt.xlabel(r"$t^{1/2}$ (min$^{1/2}$)")
    plt.ylabel(r"$q_t$ (mg g$^{-1}$)")
    plt.title(f"Weber–Morris Model ({label})")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ------------------------------------------------------------
# 4. RUN ANALYSIS
# ------------------------------------------------------------
all_results = []
tables = {}

for label, qt in data.items():
    print("="*60)
    print(f"Processing {label}")

    results, table = weber_morris_fit(t, qt, label)

    if results:
        all_results.append(results)
        tables[label] = table

        plot_weber_morris(t, qt, results, table, label)

        print(pd.DataFrame([results]).T)


# ------------------------------------------------------------
# 5. SAVE RESULTS
# ------------------------------------------------------------
if all_results:
    summary = pd.DataFrame(all_results)

    with pd.ExcelWriter("weber_morris_results.xlsx") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)

        for label, table in tables.items():
            sheet = label.replace("°", "")
            table.to_excel(writer, sheet_name=sheet, index=False)

    print("\n✔ Results saved: weber_morris_results.xlsx")