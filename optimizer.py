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
    def __init__(self, pop_size=20, max_iter=30, data_size_mb=2):
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.engine = CryptoBenchmark()
        
        print(f">>> {data_size_mb} MB test verisi hazırlanıyor...")
        self.test_data = get_random_bytes(data_size_mb * 1024 * 1024)
        
        # Parametreler: Algo(0-4), Mod(0-2), Key(0-2), Buffer(1KB-64KB)
        # Not: Optimizer sürekli uzayda çalışır, Engine bunu ayrık uzaya (Discrete) yuvarlar.
        # Bu yöntem literatürde "Rounded-Continuous Metaheuristics for Discrete Problems" olarak geçer.
        self.lb = [0, 0, 0, 1024] # Lower Bound / Alt Sınır
        self.ub = [4.49, 2.49, 2.49, 65536] # Upper Bound / Üst Sınır
        self.dim = 4 
        
        self.convergence_curve = [] # Her iterasyonda (döngüde) bulduğun en iyi sonucu buraya kaydeder
        
        # Robust Scaling için Baseline Analizi (N=40)
        print(">>> Baseline Analizi (Robust Scaling - Median/IQR)...")
        self.stats = self._collect_robust_stats(samples=40)
        # İstatistik çıktısı çok kalabalık olduğu için logdan kaldırıldı, arka planda hesaplanıyor.

    def _collect_robust_stats(self, samples):
        """
        Ortamın istatistiksel dağılımını (Medyan ve IQR) çıkarır.
        Min-Max yerine bu kullanılır çünkü outlier'lara (anlık takılmalara) karşı dirençlidir.
        """
        history = {"Time": [], "CPU": [], "Mem": []}
        
        for _ in range(samples):
            rand_x = [random.uniform(self.lb[j], self.ub[j]) for j in range(self.dim)]
            res = self.engine._run_single_test(rand_x, self.test_data)
            if res:
                history["Time"].append(res["Time"])
                history["CPU"].append(res["CPU_Time"])
                history["Mem"].append(res["Memory"])
        
        stats = {}
        for key in history:
            data = np.array(history[key])
            stats[key + "_Med"] = np.median(data)
            # IQR = Q3 - Q1
            q75, q25 = np.percentile(data, [75 ,25])
            stats[key + "_IQR"] = q75 - q25
            # Sıfıra bölme hatası önlemi
            if stats[key + "_IQR"] == 0: stats[key + "_IQR"] = 1.0
            
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
        res = self.engine.benchmark_with_stats(x, self.test_data, repeats=5)
        if res is None: return self.engine.PENALTY
        
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

    def optimize(self):
        population = np.zeros((self.pop_size, self.dim))
        for i in range(self.pop_size):
            for j in range(self.dim):
                population[i, j] = random.uniform(self.lb[j], self.ub[j])

        rabbit_pos = np.zeros(self.dim)
        rabbit_score = float("inf")

        print("-" * 65)
        print(f"{'Iter':<5} | {'Cost':<10} | {'Algoritma':<10} | {'Mod'}")
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

        return rabbit_pos, rabbit_score

    def plot_results(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.convergence_curve, 'purple', linewidth=2, label="Robust HHO-WOA")
        plt.title('Robust Optimizasyon Yakınsaması', fontsize=14)
        plt.xlabel('İterasyon', fontsize=12)
        plt.ylabel('Maliyet (Robust Scaled)', fontsize=12)
        plt.grid(True)
        plt.legend()
        plt.savefig('robust_result.png')
        print(">>> Grafik kaydedildi: robust_result.png")

if __name__ == "__main__":
    # N=40 Baseline + 5 Repeats + 20 Pop + 30 Iter
    print("\n" + "="*60)
    print(">>> HHO-WOA Kriptografik Optimizasyon Başlatılıyor <<<")
    print("="*60)
    
    # 2 MB veri ile testi başlatıyoruz
    optimizer = RobustOptimizer(pop_size=20, max_iter=30, data_size_mb=2)
    best_x, best_score = optimizer.optimize()
    
    # Grafiği çiz
    optimizer.plot_results()
    
    # --- SONUÇLARI ANLAMLANDIRMA VE RAPORLAMA ---
    print("\n" + "="*60)
    print("            🏆 GLOBAL OPTIMUM SONUÇ RAPORU 🏆")
    print("="*60)
    
    # Mapping Sözlükleri (Motor ile aynı)
    algo_map = {0: "AES", 1: "ChaCha20", 2: "3DES", 3: "Blowfish", 4: "CAST5"}
    mod_map = {0: "CBC", 1: "GCM/AEAD", 2: "CTR"}
    key_map = {0: "128-bit / Min", 1: "192-bit / Mid", 2: "256-bit / Max"}
    
    # Ham değerleri yuvarla ve eşleştir
    final_algo_id = int(round(best_x[0]))
    final_mod_id = int(round(best_x[1]))
    final_key_id = int(round(best_x[2]))
    final_buffer = int(round(best_x[3]))
    
    # İsimleri al (Index hatası olmaması için min/max koruması)
    algo_name = algo_map.get(min(final_algo_id, 4), "Unknown")
    mod_name = mod_map.get(min(final_mod_id, 2), "Unknown")
    
    # Engine Override Kontrolü (Raporun doğru çıkması için)
    if algo_name == "ChaCha20":
        mod_name = "AEAD"
    elif algo_name in ["3DES", "Blowfish", "CAST5"]:
        mod_name = "CBC"

    # ChaCha20 özel durumu (Key sabittir)
    key_name = "256-bit (Sabit)" if algo_name == "ChaCha20" else key_map.get(min(final_key_id, 2), "Unknown")
    
    print(f"✅ En İyi Skor (Cost)     : {best_score:.5f}")
    print("-" * 60)
    print(f"🔹 Seçilen Algoritma     : {algo_name}")
    print(f"🔹 Çalışma Modu          : {mod_name}")
    print(f"🔹 Anahtar Uzunluğu      : {key_name}")
    print(f"🔹 Buffer (Chunk) Boyutu : {final_buffer} bytes ({final_buffer/1024:.2f} KB)")
    print("-" * 60)
    print(">>> Analiz Tamamlandı.")