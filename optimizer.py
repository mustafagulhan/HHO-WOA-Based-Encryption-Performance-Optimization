import numpy as np
import math
import random
import matplotlib.pyplot as plt
from Crypto.Random import get_random_bytes
from crypto_engine import CryptoBenchmarkV3

# Tekrarlanabilirlik için Seed (Bilimsel şart)
random.seed(42)
np.random.seed(42)

class ScientificOptimizer:
    def __init__(self, pop_size=10, max_iter=20, data_size_mb=10):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.engine = CryptoBenchmarkV3()
        
        # Gerçek Veri (Deterministik olması için seed sonrası üretildi)
        print(f">>> {data_size_mb} MB test verisi hazırlanıyor...")
        self.test_data = get_random_bytes(data_size_mb * 1024 * 1024)
        
        # Sınırlar: Algo, Mod, Key, Buffer, Thread
        self.lb = [0, 0, 0, 1024, 1]
        self.ub = [4.49, 2.49, 2.49, 65536, 8]
        self.dim = 5
        
        self.convergence_curve = []
        
        # --- PHASE 0: BASELINE COLLECTION (WARM-UP) ---
        # Fitness normalizasyonu için ortamın limitlerini öğrenelim.
        print(">>> Baseline ölçümleri alınıyor (Max değer tespiti)...")
        self.history_max = self._collect_baseline(samples=15)
        print(f">>> Baseline Max Değerler: {self.history_max}")

    def _collect_baseline(self, samples):
        """Rastgele örneklerle ortamın max Time, CPU ve Mem değerlerini bulur."""
        maxes = {"Time": 0.1, "CPU": 0.1, "Mem": 0.1} # 0 olmasın diye minik değer
        
        for _ in range(samples):
            # Rastgele bir kromozom üret
            rand_x = [random.uniform(self.lb[j], self.ub[j]) for j in range(self.dim)]
            # Isınma turu olmadan tekli ölçüm yeterli baseline için
            res = self.engine._single_run(rand_x, self.test_data)
            
            if res:
                if res["Time"] > maxes["Time"]: maxes["Time"] = res["Time"]
                if res["CPU_Time"] > maxes["CPU"]: maxes["CPU"] = res["CPU_Time"]
                if res["Memory"] > maxes["Mem"]: maxes["Mem"] = res["Memory"]
        
        # Biraz marj bırakalım (%10)
        maxes["Time"] *= 1.1
        maxes["CPU"] *= 1.1
        maxes["Mem"] *= 1.1
        return maxes

    def calculate_fitness(self, x):
        # 3 Tekrarlı Medyan ölçüm
        res = self.engine.benchmark_with_repeats(x, self.test_data, repeats=3)
        if res is None:
            return self.engine.PENALTY
        
        # Sabit Baseline ile Normalizasyon
        n_Time = min(res["Time"] / self.history_max["Time"], 1.0)
        n_CPU = min(res["CPU_Time"] / self.history_max["CPU"], 1.0)
        n_Mem = min(res["Memory"] / self.history_max["Mem"], 1.0)
        n_Sec = res["Security"]
        
        cost = (self.engine.wT * n_Time) + \
               (self.engine.wCPU * n_CPU) + \
               (self.engine.wM * n_Mem) - \
               (self.engine.wS * n_Sec)
        return cost

    def optimize(self):
        population = np.zeros((self.pop_size, self.dim))
        # Başlangıç
        for i in range(self.pop_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(self.lb[j], self.ub[j])

        rabbit_pos = np.zeros(self.dim)
        rabbit_score = float("inf")

        print("-" * 65)
        print(f"{'Iter':<5} | {'Cost':<10} | {'Algoritma'} | {'Mod'}")
        print("-" * 65)

        for t in range(self.max_iter):
            # 1. Fitness Update
            for i in range(self.pop_size):
                population[i, :] = np.clip(population[i, :], self.lb, self.ub)
                fitness = self.calculate_fitness(population[i, :])
                
                if fitness < rabbit_score:
                    rabbit_score = fitness
                    rabbit_pos = population[i, :].copy()
            
            self.convergence_curve.append(rabbit_score)
            
            # 2. ADAPTİF HYBRID STRATEJİSİ
            # Başta HHO ağırlıklı (Keşif), Sonda WOA ağırlıklı (Sömürü)
            # Alpha zamanla azalır: 0.9 -> 0.1
            alpha = 0.9 * (1 - (t / self.max_iter)) 
            
            # HHO Enerjisi
            E1 = 2 * (1 - (t / self.max_iter))
            
            for i in range(self.pop_size):
                # Olasılık kontrolü: Alpha yüksekse HHO çalıştır
                if random.random() < alpha + 0.1: # +0.1 baz HHO şansı
                    # --- HHO MANTIĞI ---
                    E0 = 2 * random.random() - 1
                    E = 2 * E0 * E1
                    
                    if abs(E) >= 1: # Exploration
                        q = random.random()
                        rand_hawk = population[random.randint(0, self.pop_size-1), :]
                        if q < 0.5:
                            population[i, :] = rand_hawk - random.random() * abs(rand_hawk - 2 * random.random() * population[i, :])
                        else:
                            population[i, :] = (rabbit_pos - population.mean(0)) - random.random() * ((np.array(self.ub) - np.array(self.lb)) * random.random() + np.array(self.lb))
                    else: # Exploitation (Soft Besiege Basitleştirilmiş)
                        population[i, :] = (rabbit_pos - population[i, :]) - E * abs(rabbit_pos - population[i, :])
                
                else:
                    # --- WOA MANTIĞI (Spiral Update) ---
                    distance = abs(rabbit_pos - population[i, :])
                    b = 1
                    l = (random.random() * 2) - 1
                    population[i, :] = distance * math.exp(b * l) * math.cos(2 * math.pi * l) + rabbit_pos

            # Loglama
            algo_map = ["AES", "ChaCha", "3DES", "Blow", "CAST"]
            mod_map = ["CBC", "GCM/STR", "CTR"]
            
            a_idx = int(round(rabbit_pos[0]))
            m_idx = int(round(rabbit_pos[1]))
            # Index koruması
            a_name = algo_map[min(a_idx, 4)]
            m_name = mod_map[min(m_idx, 2)]
            
            print(f"{t+1:<5} | {rabbit_score:.5f}    | {a_name:<9} | {m_name}")

        return rabbit_pos, rabbit_score

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'b-o', linewidth=2, label="Adaptive HHO-WOA")
        plt.title('Bilimsel Optimizasyon Yakınsama Grafiği', fontsize=14)
        plt.xlabel('İterasyon', fontsize=12)
        plt.ylabel('Maliyet (Cost)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.savefig('scientific_result.png')
        print(">>> Grafik kaydedildi: scientific_result.png")

if __name__ == "__main__":
    # 10 MB veri, 3 tekrar = Her fitness hesabı ~30MB işlem demektir.
    # Toplam süre biraz uzayabilir ama sonuçlar KESİN olur.
    optimizer = ScientificOptimizer(pop_size=8, max_iter=15, data_size_mb=10)
    best_x, best_score = optimizer.optimize()
    optimizer.plot_results()
    
    print("\n>>> GLOBAL OPTIMUM <<<")
    print(f"Skor: {best_score}")
    print(f"Parametreler: {best_x}")