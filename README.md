# HHO-WOA Based Encryption Performance Optimization

This project is an AI-driven system that uses a hybrid of **Harris Hawks Optimization (HHO)** and the **Whale Optimization Algorithm (WOA)** to optimize the **performance–security balance** of cryptographic algorithms.

The system evaluates symmetric ciphers such as **AES, ChaCha20, 3DES, Blowfish, CAST5** and modes like **CBC, GCM, CTR**. Based on the **real-time system state**, it selects the best configuration:

- Cipher algorithm  
- Mode of operation  
- Key length  
- Buffer size  

---

## Features

- **Hybrid Metaheuristic Approach**  
  Combines HHO’s strong **exploration** capability with WOA’s effective **exploitation** mechanism.

- **Comparative Optimization**  
  Runs **HHO-WOA**, **Differential Evolution (DE)**, and **Particle Swarm Optimization (PSO)** on the same cost function in parallel; convergence curves are compared in a single plot.

- **Robust Optimization**  
  Uses a **Median / IQR**-based Robust Scaler instead of Min-Max normalization to reduce the impact of OS-induced jitter and outlier measurements.

- **Multi-Objective Optimization**  
  Minimizes a weighted cost function that includes:  
  - Performance: time, CPU usage, RAM consumption  
  - Security: compliance with NIST cryptographic standards  

---

## Dynamic Adaptation and Result Variability

This system is **non-deterministic**, **stochastic**, and **dynamic**. Different runs may yield different results (e.g., one run may pick `AES-CTR`, another may pick `ChaCha20`). This is not an error—it is a natural outcome of the system’s **adaptive capability**.

### Why Results Can Change

1. **Dynamic System Resources**  
   CPU load, cache state, and RAM usage change within milliseconds. The optimizer chooses the solution that best fits the **current hardware conditions**.

2. **Nature of Metaheuristic Algorithms**  
   The search space is explored with randomness. It is mathematically expected for the optimizer to switch among strong candidates (global optima) whose performance is very close.

---

## Installation

You need **Python 3.x** and the project dependencies.

1. Clone/download the project.  
2. Navigate to the project directory in your terminal.  
3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

The optimization process measures system resources in real time and can yield different results on each run.

### Start the Optimization

Run the following command to execute HHO-WOA, DE, and PSO on the same search space and cost function:

```bash
python optimizer.py
```

During execution the system:

- Measures CPU and RAM usage in real time  
- Benchmarks selected algorithms and modes  
- Finds the configuration that minimizes the multi-objective cost function  
- Produces convergence curves for all three algorithms in `robust_result.png`  
- Reports the best configuration (algo/mode/key/buffer) and continuous parameters (`data_size_mb`, `repeats`) in the console  

---

## Re-Running and Evaluation

Because the system is stochastic:

- Re-running the same command can produce different but **near-optimal** results.  
- This is not instability; it is a direct result of **adaptive optimization**.

For a fair assessment:  
- Run multiple times.  
- Compare averages or medians of the results.

---

## Parameter Customization

You can adjust optimization parameters inside `optimizer.py`:

- Population size  
- Maximum iterations  
- Performance / security weights  
- Continuous parameters:  
  - `data_size_mb` (1–8 MB chunked data)  
  - `repeats` (1–10 benchmark repetitions)  

This allows configurations that are:  
- Performance-oriented  
- Security-oriented  
- Balanced  

---

## Note

The goal is not to pick a single **fastest** or **most secure** cipher in isolation; it is to find the **best-balanced solution** for the current system load and security requirements.

---

## Additional Study: Michalewicz Benchmark (d=50, m=10)

- HHO-WOA, DE, and PSO minimize the 50-dimensional Michalewicz function (m=10, 0 ≤ x_i ≤ π).  
- Population: 60, iterations: 1000 (configurable).  
- Output: convergence curves in `michalewicz_result.png`; console prints best scores and example solution components.

Run with:

```bash
python michalewicz_benchmark.py
```
