import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
path = Path('data/core_circuit_output.txt')
data = np.loadtxt(path)
S = data[:,0]
mZEB = data[:,1]
plt.figure(figsize=(6,4))
plt.plot(S, mZEB, '.', markersize=1, label='Forward sweep')
plt.xlabel('External signal S')
plt.ylabel('mZEB steady state')
plt.title('Bifurcation diagram: Core circuit')
plt.tight_layout()
output_fig = Path('figs/core_circuit_bifurcation.png')
output_fig.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_fig, dpi=300)