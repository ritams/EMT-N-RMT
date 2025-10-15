import math
from pathlib import Path

import numpy as np

NUM_POINTS = 100
STATE_SIZE = 7
DT = 0.1
T_END = 1000.0

L = np.array([1.0, 0.6, 0.3, 0.1, 0.05, 0.05, 0.05], dtype=np.float64)
GAMMA_MRNA = np.array([0.0, 0.04, 0.2, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
GAMMA_MIRNA = np.array([0.0, 0.005, 0.05, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
COMB_6 = np.array([math.comb(6, i) for i in range(7)], dtype=np.float64)
COMB_2 = np.array([math.comb(2, i) for i in range(3)], dtype=np.float64)
INDICES_7 = np.arange(7, dtype=np.float64)
INDICES_3 = np.arange(3, dtype=np.float64)

G_MIR34 = 1.35e3
G_MSNAIL = 90.0
G_SNAIL = 0.1e3
G_MIR200 = 2.1e3
G_MZEB = 11.0
G_ZEB = 0.1e3

K_MIR34 = 0.05
K_MSNAIL = 0.5
K_SNAIL = 0.125
K_MIR200 = 0.05
K_MZEB = 0.5
K_ZEB = 0.1

T_MIR34_SNAIL = 300e3
T_MSNAIL_SNAIL = 200e3
T_MIR34_ZEB = 600e3
T_MIR34 = 10e3
T_MSNAIL_I = 50e3
T_MIR200_ZEB = 220e3
T_MIR200_SNAIL = 180e3
T_MZEB_ZEB = 25e3
T_MZEB_SNAIL = 180e3
T_MIR200 = 10e3

N_MIR34_SNAIL = 1
N_MIR34_ZEB = 1
N_MSNAIL_SNAIL = 1
N_MSNAIL_I = 1
N_MIR200_ZEB = 3
N_MIR200_SNAIL = 2
N_MZEB_ZEB = 2
N_MZEB_SNAIL = 2

L_MIR34_SNAIL = 0.1
L_MSNAIL_SNAIL = 0.1
L_MIR34_ZEB = 0.2
L_MSNAIL_I = 10.0
L_MIR200_ZEB = 0.1
L_MIR200_SNAIL = 0.1
L_MZEB_ZEB = 7.5
L_MZEB_SNAIL = 10.0


def _hill(x: float, threshold: float, hill_n: int, leak: float) -> float:
    base = 1.0 / (1.0 + (x / threshold) ** hill_n)
    return base + leak * (1.0 - base)


def snail_zeb_mir200_mir34_system(state: np.ndarray, deriv: np.ndarray) -> None:
    fac_mir200_num = state[0] / T_MIR200
    fac_mir200 = np.power(fac_mir200_num, INDICES_7) / np.power(1.0 + fac_mir200_num, 6)

    degrad_mir200 = np.sum(GAMMA_MIRNA * COMB_6 * INDICES_7 * fac_mir200)
    degrad_mzeb = np.sum(GAMMA_MRNA * COMB_6 * fac_mir200)
    trans_mzeb = np.sum(L * COMB_6 * fac_mir200)

    fac_mir34_num = state[5] / T_MIR34
    fac_mir34 = np.power(fac_mir34_num, INDICES_3) / np.power(1.0 + fac_mir34_num, 2)

    degrad_mir34 = np.sum(GAMMA_MIRNA[:3] * COMB_2 * INDICES_3 * fac_mir34)
    degrad_msna = np.sum(GAMMA_MRNA[:3] * COMB_2 * fac_mir34)
    trans_msna = np.sum(L[:3] * COMB_2 * fac_mir34)

    h_mir200_zeb = _hill(state[2], T_MIR200_ZEB, N_MIR200_ZEB, L_MIR200_ZEB)
    h_mir200_sna = _hill(state[3], T_MIR200_SNAIL, N_MIR200_SNAIL, L_MIR200_SNAIL)
    h_mzeb_zeb = _hill(state[2], T_MZEB_ZEB, N_MZEB_ZEB, L_MZEB_ZEB)
    h_mzeb_sna = _hill(state[3], T_MZEB_SNAIL, N_MZEB_SNAIL, L_MZEB_SNAIL)
    h_mir34_sna = _hill(state[3], T_MIR34_SNAIL, N_MIR34_SNAIL, L_MIR34_SNAIL)
    h_mir34_zeb = _hill(state[2], T_MIR34_ZEB, N_MIR34_ZEB, L_MIR34_ZEB)
    h_msna_sna = _hill(state[3], T_MSNAIL_SNAIL, N_MSNAIL_SNAIL, L_MSNAIL_SNAIL)
    h_msna_i = _hill(state[6], T_MSNAIL_I, N_MSNAIL_I, L_MSNAIL_I)

    deriv[0] = G_MIR200 * h_mir200_zeb * h_mir200_sna - state[1] * degrad_mir200 - K_MIR200 * state[0]
    deriv[1] = G_MZEB * h_mzeb_zeb * h_mzeb_sna - state[1] * degrad_mzeb - K_MZEB * state[1]
    deriv[2] = G_ZEB * state[1] * trans_mzeb - K_ZEB * state[2]

    
    deriv[3] = G_SNAIL * state[4] * trans_msna - K_SNAIL * state[3]
    deriv[4] = G_MSNAIL * h_msna_i * h_msna_sna - state[4] * degrad_msna - K_MSNAIL * state[4]
    deriv[5] = G_MIR34 * h_mir34_zeb * h_mir34_sna - state[4] * degrad_mir34 - K_MIR34 * state[5]
    deriv[6] = 0.0


def rk4_integrate(system, state: np.ndarray, t0: float, t_end: float, dt: float, steady_tol: float | None = None) -> None:
    steps = int((t_end - t0) / dt)
    k1 = np.empty_like(state)
    k2 = np.empty_like(state)
    k3 = np.empty_like(state)
    k4 = np.empty_like(state)
    temp = np.empty_like(state)

    for _ in range(steps):
        system(state, k1)
        temp[:] = state + 0.5 * dt * k1
        system(temp, k2)
        temp[:] = state + 0.5 * dt * k2
        system(temp, k3)
        temp[:] = state + dt * k3
        system(temp, k4)
        delta = (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        state += delta
        if steady_tol is not None and np.max(np.abs(delta)) < steady_tol:
            break


def get_bifurcation(start_val: float, end_val: float, num_points: int = NUM_POINTS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    control = np.linspace(start_val, end_val, num_points, endpoint=False, dtype=np.float64)
    snail = np.empty(num_points, dtype=np.float64)
    mzeb = np.empty(num_points, dtype=np.float64)
    state = np.zeros(STATE_SIZE, dtype=np.float64)

    for idx, control_value in enumerate(control):
        print(f"{idx}/{num_points}")
        state[6] = control_value
        rk4_integrate(snail_zeb_mir200_mir34_system, state, 0.0, T_END, DT, steady_tol=1e-6)
        mzeb[idx] = state[1]
        snail[idx] = state[3]

    return control, snail, mzeb


def write_output(path: Path, snail: np.ndarray, mzeb: np.ndarray) -> None:
    data = np.column_stack((snail, mzeb))
    np.savetxt(path, data, fmt="%.10g")


def run_default() -> None:
    segments = [
        get_bifurcation(20e3, 120e3),
        get_bifurcation(120e3, 20e3),
        get_bifurcation(65e3, 20e3),
    ]

    output_path = Path("data/core_circuit_output.txt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for _, snail, mzeb in segments:
            for s_val, m_val in zip(snail, mzeb):
                handle.write(f"{s_val:.10g} {m_val:.10g}\n")


if __name__ == "__main__":
    run_default()
