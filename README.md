## 📊 Coalition Predictions: Modeling Ideological Viability in Dutch Politics

This project, part of the Bachelor in Computational Social Science, investigates how ideological divergence shapes coalition formation in the Netherlands' multiparty democracy. We analyze Tweede Kamer speeches, historical coalition patterns, and seat distributions to rank realistic majority coalitions using a weighted scoring system.

Our model combines machine learning, rule-based logic, and historical data to evaluate coalitions based on ideological fit, policy alignment, historical precedent, and institutional viability. We focus especially on:

* Ideological distance between parties
* Historical frequency of coalition combinations
* Known incompatibilities and public rejections
* Institutional feasibility (e.g., Eerste Kamer alignment)


---

## 🧠 Coalition Model Components

#### 1. Historical Coalition Score

Rewards overlap with past governments, based on:

* Frequency in past coalitions
* Party size (log-scaled for diminishing returns)
* Lineage mapping (e.g., GL/PvdA → GL + PvdA; NSC ← CDA)
* Partial matches via lineage count as 50%

#### 2. Ideological Distance

Penalizes ideological spread between parties, using:

* 2D axes: Left–Right, Progressive–Conservative
* 4D axes: Economic, Cultural, Globalist–Nationalist, Libertarian–Authoritarian
* Based on average pairwise Euclidean distance

#### 3. Topic Compatibility

Assesses programmatic alignment using the mean Jensen–Shannon Divergence (JSD) between parties’ topic distributions. Higher divergence = lower compatibility.

#### 4. Eerste Kamer Alignment

Checks for majority (38+ seats) in the Senate using expanded lineage. Full majority is rewarded; partial presence is proportionally scored.

#### 5. Exclusion Filters

Certain combinations are ruled out entirely due to irreconcilable differences (e.g., PVV + Volt, FvD + BIJ1). These are filtered before scoring.

#### 6. Complexity Penalties

To discourage large or excessive coalitions:

* +2 penalty per party beyond 4
* Minor penalty for surplus seats over 76
  Encourages minimal winning coalitions.

---

### 🧪 Final Score Formula

```python
Final Score = (
    + Historical Score × 2
    – Ideological Distance × 2
    + EK Alignment × 0.25
    – JSD × 10
    – Party Count Penalty × 2
    – Surplus Seat Penalty
)
```

All scores are normalized to a 0–100 scale using fixed bounds (4.5; -3). 

---


## 🤝 Information

- ManifestoBERTa codes: https://manifesto-project.wzb.eu/coding_schemes/mp_v4
- Files from .gitignore: http://merijn.cc/cssci-tweedekamer

---

## 💾 Data Sources

* **Political Speech Dataset:** Speeches from the Tweede Kamer (2014–2024), sourced from [opendata.tweedekamer.nl](https://opendata.tweedekamer.nl/)
* **Post X Society / DemocratieMonitor:** Provided structured speech data, seat distributions, and feedback
* **Historical Cabinet Data:** Coalition compositions from 1918 to 2023
* **Seat Distributions:** Both Tweede Kamer (100 and 150 seats) and Eerste Kamer (50 and 75 seats) distributions

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

3. open /rule-based_model/coalition-output.ipynb in Jupyter Notebook or your preferred IDE.

4. Fill in the seat distribution you want to check and the year the model should look at for the Eerste Kamer distributions.
    - make sure to remove the year you are looking at from the dataset in coalition-calculations.py

5. Run the notebook cell by cell to see the coalition predictions.
    - The notebook will show the coalition predictions for the given seat distribution and year, along with the historical frequency score, ideological distance penalty, EK alignment score, and final score.

---

## 📁 Project Structure

```
cssci-political-speech/
├── data
│   ├── api
│   │   ├── data                                 # API-related data
│   │   └── pdf                                  # All PDFs from the API
│   ├── cabinets                                 # Historical Dutch cabinet data
│   ├── opendata-tk                              # Raw and processed Tweede Kamer speech data
│   │   ├── DemocratieMonitor_speeches_2014-2024 # Processed speeches from DemocratieMonitor
│   │   └── opendata-notebooks                   # Original notebooks from opendata.tweedekamer.nl
│   └── zetelverdeling                           # Historical seat distribution data
│       ├── eerste-zetels                        # Raw Eerste Kamer seat distributions
│       └── zetel-data                           # Tweede and Eerste Kamer seat distributions
│           └── verkiezingsuitslag               # Election results and seat distributions
├── methods                                      # Jensen-Shannon divergence and ideology score calculations
├── old-weekly_goals                             # Weekly goals trying out different methods
├── party_speeches                               # All speeches sorted by party
├── roberta                                      # RoBERTa model for topic modeling
├── rule-based_model                             # Rule-based model for coalition predictions
└── txt                                          # All speeches in txt format
```
