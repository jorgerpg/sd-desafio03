#!/usr/bin/env python3
import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime

def run_command(cmd, env=None):
    """Executa um comando e retorna a saída"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
        return result.stdout, result.stderr, result.returncode
    except Exception as e:
        return "", str(e), 1

def parse_output(output, program):
    """Analisa a saída dos programas e extrai métricas"""
    metrics = {}
    
    if program == "seq":
        # Parse para saxpy_seq
        time_match = re.search(r"Tempo: ([\d.]+) ms", output)
        gflops_match = re.search(r"\| ([\d.]+) GFLOP/s", output)
        checksum_match = re.search(r"Checksum\(10\): ([\d.e+-]+)", output)
        
        if time_match:
            metrics["time_ms"] = float(time_match.group(1))
        if gflops_match:
            metrics["gflops"] = float(gflops_match.group(1))
        if checksum_match:
            metrics["checksum"] = float(checksum_match.group(1))
            
    elif program == "omp":
        # Parse para saxpy_omp
        time_match = re.search(r"Tempo: ([\d.]+) ms", output)
        gflops_match = re.search(r"\| ([\d.]+) GFLOP/s", output)
        checksum_match = re.search(r"Checksum\(10\): ([\d.e+-]+)", output)
        
        if time_match:
            metrics["time_ms"] = float(time_match.group(1))
        if gflops_match:
            metrics["gflops"] = float(gflops_match.group(1))
        if checksum_match:
            metrics["checksum"] = float(checksum_match.group(1))
            
    elif program == "cuda":
        # Parse para saxpy_cuda
        h2d_match = re.search(r"H2D: ([\d.]+) ms", output)
        kernel_match = re.search(r"K: ([\d.]+) ms", output)
        d2h_match = re.search(r"D2H: ([\d.]+) ms", output)
        total_match = re.search(r"Total: ([\d.]+) ms", output)
        gflops_match = re.search(r"Kernel: ([\d.]+) GFLOP/s", output)
        error_match = re.search(r"Max \|err\| \(amostra 1k\): ([\d.e+-]+)", output)
        
        if h2d_match:
            metrics["h2d_ms"] = float(h2d_match.group(1))
        if kernel_match:
            metrics["kernel_ms"] = float(kernel_match.group(1))
        if d2h_match:
            metrics["d2h_ms"] = float(d2h_match.group(1))
        if total_match:
            metrics["time_ms"] = float(total_match.group(1))
        if gflops_match:
            metrics["gflops"] = float(gflops_match.group(1))
        if error_match:
            metrics["error"] = float(error_match.group(1))
    
    return metrics

def run_benchmarks(n_values, num_threads=8):
    """Executa os benchmarks para diferentes valores de N"""
    results = {
        "seq": {"times": [], "gflops": [], "checksums": []},
        "omp": {"times": [], "gflops": [], "checksums": []},
        "cuda": {"times": [], "gflops": [], "h2d": [], "kernel": [], "d2h": [], "errors": []}
    }
    
    for n in n_values:
        print(f"Executando para N={n}")
        
        # Executar versão sequencial
        stdout, stderr, returncode = run_command(f"./saxpy_seq {n}")
        if returncode == 0:
            metrics = parse_output(stdout, "seq")
            results["seq"]["times"].append(metrics.get("time_ms", 0))
            results["seq"]["gflops"].append(metrics.get("gflops", 0))
            results["seq"]["checksums"].append(metrics.get("checksum", 0))
        
        # Executar versão OpenMP
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(num_threads)
        stdout, stderr, returncode = run_command(f"./saxpy_omp {n}", env=env)
        if returncode == 0:
            metrics = parse_output(stdout, "omp")
            results["omp"]["times"].append(metrics.get("time_ms", 0))
            results["omp"]["gflops"].append(metrics.get("gflops", 0))
            results["omp"]["checksums"].append(metrics.get("checksum", 0))
        
        # Executar versão CUDA
        stdout, stderr, returncode = run_command(f"./saxpy_cuda {n} 1")
        if returncode == 0:
            metrics = parse_output(stdout, "cuda")
            results["cuda"]["times"].append(metrics.get("time_ms", 0))
            results["cuda"]["gflops"].append(metrics.get("gflops", 0))
            results["cuda"]["h2d"].append(metrics.get("h2d_ms", 0))
            results["cuda"]["kernel"].append(metrics.get("kernel_ms", 0))
            results["cuda"]["d2h"].append(metrics.get("d2h_ms", 0))
            results["cuda"]["errors"].append(metrics.get("error", 0))
    
    return results

def plot_results(n_values, results, output_dir="plots"):
    """Gera gráficos a partir dos resultados"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Converter N para milhões para melhor legibilidade
    n_millions = [n / 1e6 for n in n_values]
    
    # Gráfico de Tempo de Execução
    plt.figure(figsize=(10, 6))
    plt.plot(n_millions, results["seq"]["times"], 'o-', label="Sequencial")
    plt.plot(n_millions, results["omp"]["times"], 's-', label=f"OpenMP ({num_threads} threads)")
    plt.plot(n_millions, results["cuda"]["times"], '^-', label="CUDA")
    plt.xlabel("Tamanho do Problema (Milhões de elementos)")
    plt.ylabel("Tempo de Execução (ms)")
    plt.title("Tempo de Execução vs Tamanho do Problema")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/tempo_execucao.png", dpi=300, bbox_inches='tight')
    
    # Gráfico de GFLOPS
    plt.figure(figsize=(10, 6))
    plt.plot(n_millions, results["seq"]["gflops"], 'o-', label="Sequencial")
    plt.plot(n_millions, results["omp"]["gflops"], 's-', label=f"OpenMP ({num_threads} threads)")
    plt.plot(n_millions, results["cuda"]["gflops"], '^-', label="CUDA (Kernel)")
    
    # Calcular GFLOPS total para CUDA (considerando transferências)
    cuda_total_gflops = []
    for i, n in enumerate(n_values):
        flops = 2.0 * n  # 1 multiplicação + 1 adição por elemento
        time_s = results["cuda"]["times"][i] / 1000.0
        if time_s > 0:
            cuda_total_gflops.append((flops / 1e9) / time_s)
        else:
            cuda_total_gflops.append(0)
    
    plt.plot(n_millions, cuda_total_gflops, 'd-', label="CUDA (Total)")
    plt.xlabel("Tamanho do Problema (Milhões de elementos)")
    plt.ylabel("Desempenho (GFLOP/s)")
    plt.title("Desempenho vs Tamanho do Problema")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/desempenho.png", dpi=300, bbox_inches='tight')
    
    # Gráfico de Speedup
    plt.figure(figsize=(10, 6))
    speedup_omp = [results["seq"]["times"][i] / results["omp"]["times"][i] for i in range(len(n_values))]
    speedup_cuda = [results["seq"]["times"][i] / results["cuda"]["times"][i] for i in range(len(n_values))]
    
    plt.plot(n_millions, speedup_omp, 's-', label=f"OpenMP ({num_threads} threads)")
    plt.plot(n_millions, speedup_cuda, '^-', label="CUDA")
    plt.xlabel("Tamanho do Problema (Milhões de elementos)")
    plt.ylabel("Speedup (vs Sequencial)")
    plt.title("Speedup vs Tamanho do Problema")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{output_dir}/speedup.png", dpi=300, bbox_inches='tight')
    
    # Gráfico de Breakdown do Tempo CUDA
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(n_values))
    for i in range(len(n_values)):
        plt.bar(i, results["cuda"]["h2d"][i], bottom=bottom[i], label="H2D" if i == 0 else "", color='lightblue')
        bottom[i] += results["cuda"]["h2d"][i]
        plt.bar(i, results["cuda"]["kernel"][i], bottom=bottom[i], label="Kernel" if i == 0 else "", color='lightgreen')
        bottom[i] += results["cuda"]["kernel"][i]
        plt.bar(i, results["cuda"]["d2h"][i], bottom=bottom[i], label="D2H" if i == 0 else "", color='lightcoral')
    
    plt.xticks(range(len(n_millions)), [f"{n}M" for n in n_millions])
    plt.xlabel("Tamanho do Problema")
    plt.ylabel("Tempo (ms)")
    plt.title("Breakdown do Tempo CUDA")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.savefig(f"{output_dir}/breakdown_cuda.png", dpi=300, bbox_inches='tight')
    
    # Salvar resultados em JSON para referência futura
    with open(f"{output_dir}/results.json", "w") as f:
        json.dump({
            "n_values": n_values,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    plt.show()

if __name__ == "__main__":
    # Valores de N para testar (em milhões)
    n_values = [5, 10, 50, 100, 500]  # Em milhões
    n_values = [int(n * 1e6) for n in n_values]  # Converter para valores reais
    
    num_threads = 8  # Número de threads para OpenMP
    
    print("Executando benchmarks SAXPY...")
    results = run_benchmarks(n_values, num_threads)
    
    print("Gerando gráficos...")
    plot_results(n_values, results)
    
    print("Concluído! Gráficos salvos na pasta 'plots'")