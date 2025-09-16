# SD-Desafio03 – Distributed Systems (Challenge 03)

This project implements and compares different versions of the **SAXPY** operation (`y = a*x + y`), evaluating performance on **sequential CPU**, **parallel CPU (OpenMP)**, and **GPU (CUDA)**.  
The goal is to study **parallelism**, **computational performance**, and **communication overheads** across different architectures.

---

## 📌 Project Structure

- `saxpy_seq.cpp` → Sequential implementation in C++.
- `saxpy_omp.cpp` → Parallel implementation using **OpenMP**.
- `saxpy_cuda.cu` → Implementation in **CUDA**.
- `benchmark.py` → Python script that:
  - Runs all three versions for different input sizes.
  - Extracts metrics from the programs.
  - Generates comparative charts (time, GFLOP/s, speedup, and CUDA *breakdown*).
  - Saves results in JSON for future reference.

---

## ⚙️ Prerequisites

### Compilers and Tools
- **GCC** with C++17 and OpenMP support  
- **NVIDIA CUDA Toolkit**  
- **Python 3.8+** with libraries:
  - `matplotlib`
  - `numpy`
  - `json`
  - `datetime`

### Compilation
```bash
# Compile sequential version
g++ -O3 -std=c++17 saxpy_seq.cpp -o saxpy_seq

# Compile OpenMP version
g++ -O3 -std=c++17 -fopenmp saxpy_omp.cpp -o saxpy_omp

# Compile CUDA version
nvcc -O3 -std=c++17 saxpy_cuda.cu -o saxpy_cuda
```

---

## ▶️ Running Benchmarks

The `benchmark.py` script automates testing:

```bash
python3 benchmark.py
```

It runs each version for different problem sizes (in millions of elements):

* **5M, 10M, 50M, 100M, and 500M**

---

## 📊 Collected Metrics

### Sequential / OpenMP

* **Time (ms)**
* **GFLOP/s**
* **Checksum** (numerical sanity)

### CUDA

* **H2D (ms)** → Host → Device transfer
* **Kernel (ms)** → kernel execution time
* **D2H (ms)** → Device → Host transfer
* **Total (ms)** → sum of times
* **GFLOP/s (Kernel)**
* **GFLOP/s (Total)** (including transfers)
* **Max Error** (numerical consistency)

---

## 📈 Generated Charts

Results are saved in `plots/`:

* `tempo_execucao.png` → Time vs Problem Size
* `desempenho.png` → GFLOP/s vs Size
* `speedup.png` → Speedup vs sequential version
* `breakdown_cuda.png` → CUDA execution time *breakdown*
* `results.json` → Complete data in structured format

---

## 📚 Studied Concepts

* Differences between **sequential execution**, **CPU parallelism (OpenMP)**, and **massive GPU parallelism (CUDA)**.
* Importance of **memory transfer overhead** on GPU.
* Measuring **speedup** and **scalability** across architectures.
* Numerical sanity (checksums and max error).

---