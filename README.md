# EMT-N-RMT

A simulation tool for the core circuit in Epithelial-Mesenchymal Transition (EMT) modeling, using numerical relay methods.

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

1. Install uv:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and navigate to the project directory.

3. Install dependencies:
   ```bash
   uv sync
   ```

## Generating Bifurcation Diagram

1. Run the core circuit simulation to generate the data:
   ```bash
   uv run python src/core_circuit.py
   ```
   This will create `data/core_circuit_output.txt` with the simulation data.

2. Generate the bifurcation diagram:
   ```bash
   uv run python src/plot_bifurcation.py
   ```
   This will create `figs/core_circuit_bifurcation.png` with the plotted diagram.