# 📊 Ideological Divergence and Dutch Democracy

This group project is part of the Bachelor in Computational Social Science. We are using a comprehensive database of speeches from the Tweede Kamer (the Dutch House of Representatives) to analyze how ideological divergence affects the state of democracy in the Netherlands.



## 🔍 Overview

This project leverages machine learning and AI modeling to explore the dynamics of political discourse in the Netherlands. By analyzing speech data, we aim to understand the implications of increasing inter- and intra-partisan ideological divergence on democratic processes.


## 🎯 Research Question

Draft: **What implications (for democracy) does the increase of inter- and intra-partisan ideological divergence have?**


## 💾 Data Source

- **Political Speech Dataset:** Speeches from the Tweede Kamer. Fetched from [opendata.tweedekamer.nl](https://opendata.tweedekamer.nl/)
- **Post X Society:** The creators of DemocratieMonitor offered the structured speech dataset per year between 2014 and 2024.


## 🤝 Stakeholders

- **DemocratieMonitor:** Our primary stakeholder, providing key insights and contextual feedback.
- Other academic and policy-making bodies that might be interested in the project.


## 🪛 Setup

- **Download Speech Files:** Some datasets are too big for Github, download them here: [go.merijnvervoorn.com/tk-download1](https://go.merijnvervoorn.com/tk-download1) (includes pdf's of all speeches, csv with speeches from 2014-2024, and original notebook pdf's 2024)

*Note: Gitignore is setup so these files won't sync, please use the provided project structure and don't change file names*


## 🔧 Requirements

- **Programming Language:** Python 3 (or your preferred data analysis tools)
- **Key Libraries:** scikit-learn, pandas, nltk, matplotlib, and others as needed.
- **Environment:** A standard Python development environment (Jupyter Notebook is recommended for exploration).

*Note: Detailed installation instructions and dependencies will be added as the project develops.*


## 📁 Project Structure
```
cssci-political-speech/
├── data/                       # Main folder for all datasets
│   ├── api/                    # Files relating to using the OData and SyncFeed API
│   │   └── data/               # Output files from the APIs
│   ├── cabinets/               # Data on all historical Dutch cabinets
│   ├── opendata-tk/            # Raw and processed speech data from the Tweede Kamer
│   │   └── opendata-notebooks/ # Original Jupyter notebooks from Tweede Kamer data
│   └── zetelverdeling/         # Files relating to historical seat distributions in Tweede Kamer
│       └── zetel-data/         # The data and plots regarding historical seat distribution
├── methods/                    # Different methods we might use for models
├── results/                    # Eventual visualizations, model outputs, and analysis summaries
└── README.md                   # This file
```

