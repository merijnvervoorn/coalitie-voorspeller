# 📊 Coalition Predictions and Ideological Divergence

This group project is part of the Bachelor in Computational Social Science. We are investigating how ideological divergence shapes coalition formation in the Netherlands' multiparty democracy, using a mix of speech data and historical coalition patterns.

---

## 🔍 Overview

This project explores how **ideological differences influence coalition formation** in the Netherlands. We analyze Tweede Kamer speeches alongside historical government formation data and seat distributions to model coalition viability.

Our approach combines **machine learning, rule-based modeling, and historical analysis** to simulate coalitions that are not only numerically viable, but also ideologically and institutionally feasible.

We pay special attention to:

* The role of ideological distance between parties
* The influence of Eerste Kamer (Senate) alignment
* The historical frequency of similar coalitions
* Known incompatibilities between parties

---

## 🎯 Research Question

**How do ideological differences influence coalition formation in the Netherlands’ multiparty system?**

---

## 💾 Data Sources

* **Political Speech Dataset:** Speeches from the Tweede Kamer (2014–2024), sourced from [opendata.tweedekamer.nl](https://opendata.tweedekamer.nl/)
* **Post X Society / DemocratieMonitor:** Provided structured speech data, seat distributions, and feedback
* **Historical Cabinet Data:** Coalition compositions from 1918 to 2023
* **Seat Distributions:** Both Tweede Kamer (100 and 150 seats) and Eerste Kamer (50 and 75 seats) distributions

---

## 🧠 Modeling Approach

We simulate and rank realistic majority coalitions based on several weighted factors:

### 1. **Historical Frequency Score**

Coalitions are scored based on how often similar combinations of parties (including historical equivalents like *PvdA + GL*) occurred in past Dutch cabinets. We account for:

* **Partial lineage matches** (e.g., *JA21* treated as a split from *FvD*)
* **Scaled weighting** by party seat size using a logarithmic function
* **Overlap normalization** across coalition size

### 2. **Ideological Distance Penalty**

We assign each party a value on a manually defined ideological scale from **–5 (far-left)** to **+5 (far-right)**. Each coalition receives a penalty based on:

* **Mean pairwise ideological distance**
* Greater distance = lower compatibility

### 3. **Eerste Kamer (EK) Alignment Score**

We calculate how well each coalition is represented in the *Eerste Kamer*, based on:

* Actual EK seat distributions for the given year
* Whether the coalition would reach a **38-seat majority** in the EK
* Scaled EK alignment contributes positively to the final score

### 4. **Penalties for Overcomplexity**

To reduce unrealistic or inefficient coalitions:

* **Party count penalty:** Coalitions with >4 parties are penalized
* **Surplus seat penalty:** Coalitions with >90 TK seats are slightly penalized to prefer minimal-majority formations

### 5. **Exclusion of Unrealistic Combinations**

Known ideologically incompatible or adversarial party pairs (e.g., *PVV* + *BIJ1*, *FvD* + *Volt*) are excluded from the simulation entirely.

---

## 📊 Final Coalition Score Calculation

Each viable coalition receives a **final score between 0 and 100**, calculated as:

```
Final Score = (
    Historical Score × 2
    – Ideological Distance × 2
    + EK Alignment × 0.25
    – Party Count Penalty × 2
    – Surplus Penalty
)
→ Scaled to a 0–100 range
```

This ranking helps identify the **most realistic and ideologically feasible majority coalitions** given a seat distribution in the Tweede Kamer and EK.

---

## 🤝 Information

- ManifestoBERTa codes: https://manifesto-project.wzb.eu/coding_schemes/mp_v4
- Files from .gitignore: http://merijn.cc/cssci-tweedekamer

---

## 🛠️ How to use?

1. Clone the repository:
   ```bash
   git clone
   ```

2. Navigate to the project directory:
   ```bash
   cd cssci-political-speech
   ```

3. open coalition-output.ipynb in Jupyter Notebook or your preferred IDE.

4. Fill in the seat distribution you want to check and the year the model should look at for the Eerste Kamer distributions.
    - make sure to remove the year you are looking at from the dataset in coalition-calculations.py

5. Run the notebook cell by cell to see the coalition predictions.
    - The notebook will show the coalition predictions for the given seat distribution and year, along with the historical frequency score, ideological distance penalty, EK alignment score, and final score.

---

## 📁 Project Structure

```
cssci-political-speech/
├── data/                       # All datasets
│   ├── api/                    # API-related scripts and output
│   ├── cabinets/               # Historical Dutch cabinet data
│   ├── opendata-tk/            # Raw and processed Tweede Kamer speech data
│   │   └── opendata-notebooks/ # Original notebooks
│   │
│   └── zetelverdeling/         # Historical seat distribution data
│       └── zetel-data/
│
├── methods/                    # Modeling and analysis scripts
├── results/                    # Visualizations, outputs, and summaries
├── coalition-calculations.py   # Coalition calculation scripts
├── coalition-output.ipynb      # Coalition prediction
└── README.md                   # This file
```

