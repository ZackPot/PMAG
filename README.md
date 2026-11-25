# PMAG: Protein Metal Affinity GINE

**PMAG** (Protein Metal Affinity GINE) is a pytorch machine learning project directed at predicting the experimental LgK1 (first stepwise stability constant) based on a molecule's SMILES string. Molecules are represented as pytorch graph objects with node and edge features, giving the model a higher level understanding of the molecular structure allowing it to predict with a level of higher accuracy. 

---

## Features

- **Molecular Graphs:** Chemical structures are represented as graphs
- **Graph Isomorphism Network with Edge Features (GINE):** Uses message passing to predict metal affinity from node and edge features
- **Chunk generation:** Uses multiprocessing to accelerate the generation of chunks of graph objects by using the CPU in parallel

---

## Project Structure

```text
PMAG/
├── data/            # Processed Chunks and Raw Data
├── models/          # Saved Model Weights
├── sample.ipynb     # Jupyter Notebook for Data Processing and Model Training
├── workers.py       # Multiprocessing Script for Chunk Generation
└── README.md        # Documentation
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ZackPot/PMAG
   cd PMAG
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  
   ```

3. **Install dependencies:**
   ```bash
   pip install rdkit torch torch-geometric jupyter numpy pandas
   ```

## Usage

1. **Prepare the Dataset:**
   Run workers.py directed at the train_input.csv to generate chunks

2. **Run the Example Notebook:**
   Launch Jupyter and open `sample.ipynb` to train the graph neural network


---

## Sources

- [RDKit](https://rdkit.org) — Biochemistry framework for Feature Extraction
- [LibreTexts Chemistry](https://libretexts.org) — Background Reference for Metal Affinity metrics
