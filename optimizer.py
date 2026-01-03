import numpy as np
import math
import random
import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes
from crypto_engine import CryptoBenchmark

# Tekrarlanabilirlik
random.seed(42)
np.random.seed(42)

class RobustOptimizer:
    def __init__(self, pop_size=20, max_iter=30, max_data_mb=8):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.engine = CryptoBenchmark()

        # Sürekli parametre sınırları
        self.min_data_mb = 1
        self.max_data_mb = max_data_mb
        self.min_repeats = 1
        self.max_repeats = 10

        print(f">>> {self.max_data_mb} MB maksimum test verisi hazırlanıyor (dinamik dilimleme)...")
        self.base_data = get_random_bytes(self.max_data_mb * 1024 * 1024)

        # Parametreler: Algo(0-4), Mod(0-2), Key(0-2), Buffer(1KB-64KB), data_size_mb(1-8), repeats(1-10)
        # Not: Optimizer sürekli uzayda çalışır, Engine bunu ayrık uzaya (Discrete) yuvarlar.
        self.lb = [0, 0, 0, 1024, self.min_data_mb, self.min_repeats] # Lower Bound / Alt Sınır
        self.ub = [4.49, 2.49, 2.49, 65536, self.max_data_mb, self.max_repeats] # Upper Bound / Üst Sınır
        self.dim = len(self.lb)

        self.curves = {}  # Yakınsama eğrilerini algoritma adına göre tutar

        # Robust Scaling için Baseline Analizi (N=40)
        print(">>> Baseline Analizi (Robust Scaling - Median/IQR)...")
        self.stats = self._collect_robust_stats(samples=40)
        # İstatistik çıktısı çok kalabalık olduğu için logdan kaldırıldı, arka planda hesaplanıyor.

    def _slice_data(self, data_mb):
        size_mb = max(self.min_data_mb, min(self.max_data_mb, int(round(data_mb))))
        return self.base_data[: size_mb * 1024 * 1024]

    def _collect_robust_stats(self, samples):
        """
        Ortamın istatistiksel dağılımını (Medyan ve IQR) çıkarır.
        Min-Max yerine bu kullanılır çünkü outlier'lara (anlık takılmalara) karşı dirençlidir.
        """
        history = {"Time": [], "CPU": [], "Mem": []}

        for _ in range(samples):
            rand_x = [random.uniform(self.lb[j], self.ub[j]) for j in range(self.dim)]
            data_mb = rand_x[4] if len(rand_x) > 4 else self.min_data_mb
            data = self._slice_data(data_mb)
            res = self.engine._run_single_test(rand_x[:4], data)
            if res:
                history["Time"].append(res["Time"])
                history["CPU"].append(res["CPU_Time"])
                history["Mem"].append(res["Memory"])

        stats = {}
        for key in history:
            data = np.array(history[key])
            stats[key + "_Med"] = np.median(data)
            # IQR = Q3 - Q1
            q75, q25 = np.percentile(data, [75, 25])
            stats[key + "_IQR"] = q75 - q25
            # Sıfıra bölme hatası önlemi
            if stats[key + "_IQR"] == 0:
                stats[key + "_IQR"] = 1.0

        return stats

    def robust_scale(self, val, key): # Puanlama metodu
        """
        Robust Scaler Formülü: (X - Median) / IQR
        Sonuçlar genelde -1 ile 1 arasında toplanır ama dışına çıkabilir.
        Biz bunu sigmoid ile 0-1 arasına sıkıştırabiliriz veya basit clamp kullanabiliriz.
        Burada basit 0-1 clamping kullanacağız.
        """
        median = self.stats[key + "_Med"]
        iqr = self.stats[key + "_IQR"]
        
        # Scaling
        scaled = (val - median) / iqr
        
        # Negatif ve Pozitif uçları yumuşatıp 0-1 arasına çekelim (Sigmoid benzeri)
        # Basit Min-Max yerine, IQR bazlı bir relative skorlama yapıyoruz.
        # 0.5 (Medyan) merkezli bir skorlama:
        norm = 0.5 + (scaled / 4.0) # +/- 2 IQR aralığını kapsar
        return min(max(norm, 0.0), 1.0)

    def calculate_fitness(self, x): # Hakem
        algo_params = x[:4]
        data_mb = max(self.min_data_mb, min(self.max_data_mb, int(round(x[4]))))
        repeats = max(self.min_repeats, min(self.max_repeats, int(round(x[5]))))

        data = self._slice_data(data_mb)
        res = self.engine.benchmark_with_stats(algo_params, data, repeats=repeats)
        if res is None:
            return self.engine.PENALTY

        # Robust Normalization
        n_Time = self.robust_scale(res["Time"], "Time")
        n_CPU = self.robust_scale(res["CPU_Time"], "CPU")
        n_Mem = self.robust_scale(res["Memory"], "Mem")
        n_Sec = res["Security"]

        # Weighted Sum Approach to Multi-Objective Optimization / Ağırlıklı Toplam (Fonksiyon)
        cost = (self.engine.wT * n_Time) + \
               (self.engine.wCPU * n_CPU) + \
               (self.engine.wM * n_Mem) - \
               (self.engine.wS * n_Sec)
        return cost

    def _init_population(self):
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(self.lb[j], self.ub[j])
        return population

    def run_hho_woa(self):
        population = self._init_population()
        rabbit_pos = np.zeros(self.dim)
        rabbit_score = float("inf")
        curve = []

        print("-" * 80)
        print(f"{'Iter':<5} | {'Cost':<10} | {'Algoritma':<10} | {'Mod'}")
        print("-" * 80)

        for t in range(self.max_iter):
            # 1. Fitness Update
            for i in range(self.pop_size):
                population[i, :] = np.clip(population[i, :], self.lb, self.ub)
                fitness = self.calculate_fitness(population[i, :])

                if fitness < rabbit_score:
                    rabbit_score = fitness
                    rabbit_pos = population[i, :].copy()

            curve.append(rabbit_score)

            # 2. HHO-WOA Hibrit Mantığı (Matematiksel Düzeltme)
            # Lineer azalma ile exploration -> exploitation geçişi yapıyoruz.
            alpha = 1 - (t / self.max_iter) # 1'den 0'a iner
            E1 = 2 * alpha

            for i in range(self.pop_size):
                if random.random() < alpha:
                    # --- HHO FAZI (Daha yüksek exploration şansı) ---
                    E0 = 2 * random.random() - 1
                    E = 2 * E0 * E1

                    if abs(E) >= 1: # Exploration
                        q = random.random()
                        rand_idx = random.randint(0, self.pop_size-1)
                        if q < 0.5:
                            population[i, :] = population[rand_idx, :] - random.random() * abs(population[rand_idx, :] - 2 * random.random() * population[i, :])
                        else:
                            population[i, :] = (rabbit_pos - population.mean(0)) - random.random() * ((np.array(self.ub) - np.array(self.lb)) * random.random() + np.array(self.lb))
                    else:
                        # Exploitation
                        # HHO Soft/Hard Besiege
                        population[i, :] = (rabbit_pos - population[i, :]) - E * abs(rabbit_pos - population[i, :])
                else:
                    # --- WOA FAZI (Daha yüksek exploitation şansı) ---
                    # Spiral Attack
                    distance = abs(rabbit_pos - population[i, :])
                    b = 1
                    l = (random.random() * 2) - 1
                    population[i, :] = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + rabbit_pos

            # Loglama
            algo_map = ["AES", "ChaCha", "3DES", "Blow", "CAST"]
            mod_map = ["CBC", "GCM/AEAD", "CTR"]

            a_idx = int(round(rabbit_pos[0]))
            m_idx = int(round(rabbit_pos[1]))
            a_name = algo_map[min(a_idx, 4)]

            # Log Tutarlılığı: Engine tarafındaki zorlamaları loga yansıt
            if a_name == "ChaCha":
                m_name = "AEAD"
            elif a_name in ["3DES", "Blow", "CAST"]:
                m_name = "CBC"
            else:
                m_name = mod_map[min(m_idx, 2)]

            print(f"{t+1:<5} | {rabbit_score:.5f}    | {a_name:<10} | {m_name}")

        self.curves["HHO-WOA"] = curve
        return rabbit_pos, rabbit_score

    def run_de(self, F=0.5, CR=0.9):
        population = self._init_population()
        fitness = np.array([self.calculate_fitness(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_pos = population[best_idx].copy()
        best_score = fitness[best_idx]
        curve = []

        print("-" * 80)
        print(">>> DE başlıyor...")
        for t in range(self.max_iter):
            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[random.sample(idxs, 3)]
                mutant = a + F * (b - c)
                mutant = np.clip(mutant, self.lb, self.ub)

                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f_trial = self.calculate_fitness(trial)
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < best_score:
                        best_score = f_trial
                        best_pos = trial.copy()
            curve.append(best_score)
            print(f"DE iter {t+1}/{self.max_iter} | best {best_score:.5f}")

        self.curves["DE"] = curve
        return best_pos, best_score

    def run_pso(self, w=0.7, c1=1.5, c2=1.5):
        population = self._init_population()
        velocity = np.zeros_like(population)
        fitness = np.array([self.calculate_fitness(ind) for ind in population])

        pbest_pos = population.copy()
        pbest_score = fitness.copy()
        best_idx = np.argmin(fitness)
        gbest_pos = population[best_idx].copy()
        gbest_score = fitness[best_idx]
        curve = []

        print("-" * 80)
        print(">>> PSO başlıyor...")
        for t in range(self.max_iter):
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            velocity = (w * velocity) + c1 * r1 * (pbest_pos - population) + c2 * r2 * (gbest_pos - population)
            population = population + velocity
            population = np.clip(population, self.lb, self.ub)

            for i in range(self.pop_size):
                f_val = self.calculate_fitness(population[i])
                if f_val < pbest_score[i]:
                    pbest_score[i] = f_val
                    pbest_pos[i] = population[i].copy()
                    if f_val < gbest_score:
                        gbest_score = f_val
                        gbest_pos = population[i].copy()
            curve.append(gbest_score)
            print(f"PSO iter {t+1}/{self.max_iter} | best {gbest_score:.5f}")

        self.curves["PSO"] = curve
        return gbest_pos, gbest_score

    def decode_solution(self, x):
        algo_map = {0: "AES", 1: "ChaCha20", 2: "3DES", 3: "Blowfish", 4: "CAST5"}
        mod_map = {0: "CBC", 1: "GCM/AEAD", 2: "CTR"}
        key_map = {0: "128-bit / Min", 1: "192-bit / Mid", 2: "256-bit / Max"}

        algo_id = int(round(x[0]))
        mod_id = int(round(x[1]))
        key_id = int(round(x[2]))
        buffer_bytes = int(round(x[3]))
        data_mb = max(self.min_data_mb, min(self.max_data_mb, int(round(x[4]))))
        repeats = max(self.min_repeats, min(self.max_repeats, int(round(x[5]))))

        algo_name = algo_map.get(min(algo_id, 4), "Unknown")
        mod_name = mod_map.get(min(mod_id, 2), "Unknown")

        # Engine Override Kontrolü (rapor tutarlılığı için)
        if algo_name == "ChaCha20":
            mod_name = "AEAD"
        elif algo_name in ["3DES", "Blowfish", "CAST5"]:
            mod_name = "CBC"

        key_name = "256-bit (Sabit)" if algo_name == "ChaCha20" else key_map.get(min(key_id, 2), "Unknown")

        return {
            "algo": algo_name,
            "mode": mod_name,
            "key": key_name,
            "buffer_bytes": buffer_bytes,
            "buffer_kb": buffer_bytes / 1024,
            "data_mb": data_mb,
            "repeats": repeats
        }

    def print_solution(self, label, x, score):
        info = self.decode_solution(x)
        print(f"[{label}] En İyi Skor: {score:.5f}")
        print(f"  Algoritma      : {info['algo']}")
        print(f"  Çalışma Modu   : {info['mode']}")
        print(f"  Anahtar        : {info['key']}")
        print(f"  Buffer Boyutu  : {info['buffer_bytes']} bytes ({info['buffer_kb']:.2f} KB)")
        print(f"  Veri Boyutu    : {info['data_mb']} MB")
        print(f"  Tekrar (repeats): {info['repeats']}")

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        for name, curve in self.curves.items():
            plt.plot(curve, linewidth=2, label=name)
        plt.title('Robust Optimizasyon Yakınsaması', fontsize=14)
        plt.xlabel('İterasyon', fontsize=12)
        plt.ylabel('Maliyet (Robust Scaled)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.savefig('robust_result.png')
        print(">>> Grafik kaydedildi: robust_result.png")

if __name__ == "__main__":
    print("\n" + "="*70)
    print(">>> HHO-WOA / DE / PSO Kriptografik Optimizasyon Başlatılıyor <<<")
    print("="*70)

    optimizer = RobustOptimizer(pop_size=20, max_iter=30, max_data_mb=8)

    hho_x, hho_score = optimizer.run_hho_woa()
    de_x, de_score = optimizer.run_de()
    pso_x, pso_score = optimizer.run_pso()

    optimizer.plot_results()

    # --- SONUÇ RAPORU ---
    print("\n" + "="*70)
    print("            🏆 GLOBAL OPTIMUM SONUÇ RAPORU (Karşılaştırmalı) 🏆")
    print("="*70)
    optimizer.print_solution("HHO-WOA", hho_x, hho_score)
    print("-" * 60)
    optimizer.print_solution("DE", de_x, de_score)
    print("-" * 60)
    optimizer.print_solution("PSO", pso_x, pso_score)
    print("-" * 60)
    print(">>> Analiz Tamamlandı.")