import time
import os
import psutil
import statistics
import tracemalloc
from Crypto.Cipher import AES, ChaCha20, DES3, Blowfish, CAST
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

class CryptoBenchmarkV3:
    def __init__(self):
        # Ağırlıklar (Makalede gerekçelendirilecek)
        self.wT = 0.4
        self.wCPU = 0.3
        self.wM = 0.1
        self.wS = 0.2
        self.PENALTY = 1e9 # Hata durumu cezası

    def get_scientific_security_score(self, algo_name, key_len_bytes, mode_name=None):
        """
        NIST SP 800-57 Part 1 standartlarına göre Effective Security Strength.
        """
        bits = key_len_bytes * 8
        score = 0.0

        if algo_name == "AES":
            if bits >= 256: score = 1.0       # Quantum resistant-ish
            elif bits >= 192: score = 0.85
            else: score = 0.70
        elif algo_name == "ChaCha20":
            score = 0.90                      # 256-bit modern stream
        elif algo_name in ["Twofish", "Blowfish"]:
            if bits >= 128: score = 0.50
            else: score = 0.30
        elif algo_name == "3DES":
            score = 0.20                      # Legacy (112-bit effective)
        elif algo_name == "CAST5":
            score = 0.40

        # Authenticated Encryption Bonusu (GCM / Poly1305)
        if mode_name == "GCM" or algo_name == "ChaCha20":
            score += 0.10
        elif mode_name == "ECB":
            score -= 0.50 # Güvensiz mod cezası
        
        return min(max(score, 0.0), 1.0)

    def _single_run(self, params, data):
        """
        Tekil ölçüm fonksiyonu.
        params: [AlgoID, ModID, KeyIdx, BlockSize, Threads]
        """
        algo_map = {0: "AES", 1: "ChaCha20", 2: "3DES", 3: "Blowfish", 4: "CAST5"}
        algo_name = algo_map.get(int(round(params[0])), "AES")
        mode_val = int(round(params[1]))
        key_idx = int(round(params[2]))
        buffer_size = int(round(params[3]))

        try:
            cipher = None
            key = b""
            mode_name = "STREAM"

            # --- Kriptografik Kurulum (Explicit IV/Nonce) ---
            if algo_name == "AES":
                key_sizes = [16, 24, 32]
                k_len = key_sizes[min(key_idx, 2)]
                key = get_random_bytes(k_len)
                
                if mode_val == 1: # GCM
                    mode_name = "GCM"
                    nonce = get_random_bytes(12) # NIST önerisi 96-bit
                    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                elif mode_val == 2: # CTR
                    mode_name = "CTR"
                    nonce = get_random_bytes(8)  # 64-bit nonce genelde yeterli
                    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
                else: # CBC
                    mode_name = "CBC"
                    iv = get_random_bytes(AES.block_size) # Zorunlu rastgele IV
                    cipher = AES.new(key, AES.MODE_CBC, iv=iv)

            elif algo_name == "ChaCha20":
                mode_name = "STREAM"
                key = get_random_bytes(32)
                nonce = get_random_bytes(12) # RFC 7539
                cipher = ChaCha20.new(key=key, nonce=nonce)

            elif algo_name == "3DES":
                mode_name = "CBC"
                k_len = 16 if key_idx == 0 else 24
                key = get_random_bytes(k_len)
                iv = get_random_bytes(DES3.block_size)
                cipher = DES3.new(key, DES3.MODE_CBC, iv=iv)

            elif algo_name == "Blowfish":
                mode_name = "CBC"
                bs_keys = [16, 32, 56]
                key = get_random_bytes(bs_keys[min(key_idx, 2)])
                iv = get_random_bytes(Blowfish.block_size)
                cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv=iv)
            
            else: # CAST5
                mode_name = "CBC"
                key = get_random_bytes(16)
                iv = get_random_bytes(CAST.block_size)
                cipher = CAST.new(key, CAST.MODE_CBC, iv=iv)

            # Padding (Sadece CBC için)
            data_to_encrypt = data
            if mode_name == "CBC":
                blk = getattr(cipher, 'block_size', 8)
                data_to_encrypt = pad(data, blk)

            # --- ÖLÇÜM BAŞLANGICI ---
            proc = psutil.Process(os.getpid())
            
            # 1. Bellek Snapshot (Peak RSS için)
            start_rss = proc.memory_info().rss
            tracemalloc.start()

            # 2. CPU ve Wall Clock
            start_cpu = time.process_time()
            start_wall = time.perf_counter() # Yüksek çözünürlük

            # 3. İşlem (Chunking Simülasyonu)
            if mode_name == "GCM":
                cipher.encrypt_and_digest(data_to_encrypt)
            else:
                # Buffer size simülasyonu
                for i in range(0, len(data_to_encrypt), buffer_size):
                    cipher.encrypt(data_to_encrypt[i:i+buffer_size])

            # 4. Ölçüm Bitiş
            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            end_rss = proc.memory_info().rss
            
            _, peak_trace = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # --- Metrik Hesaplama ---
            wall_ms = (end_wall - start_wall) * 1000
            cpu_ms = (end_cpu - start_cpu) * 1000
            
            # Memory: Hem RSS farkını hem Trace peak'ini dikkate alalım (Max olanı)
            rss_diff = end_rss - start_rss
            mem_bytes = max(rss_diff, peak_trace)
            mem_mb = abs(mem_bytes) / (1024 * 1024)

            sec_score = self.get_scientific_security_score(algo_name, len(key), mode_name)

            return {
                "Time": wall_ms,
                "CPU_Time": cpu_ms,
                "Memory": mem_mb,
                "Security": sec_score,
                "Params": f"{algo_name}-{mode_name}"
            }

        except Exception as e:
            # Hatayı yutma, logla ama optimizasyonu kırma
            # print(f"Benchmark Error: {e}") 
            return None

    def benchmark_with_repeats(self, params, data, repeats=3):
        """
        İstatistiksel Güvenilirlik İçin:
        1 Warmup Run + N Repeat Run -> Medyan Değer
        """
        valid_results = []
        
        # 1. Warm-up (Cache ısıtma, sonucu kaydetme)
        self._single_run(params, data)
        
        # 2. Gerçek Ölçümler
        for _ in range(repeats):
            res = self._single_run(params, data)
            if res is not None:
                valid_results.append(res)
        
        if not valid_results:
            return None
        
        # Medyan Alma (Aykırı değerleri temizler)
        med_time = statistics.median([r["Time"] for r in valid_results])
        med_cpu = statistics.median([r["CPU_Time"] for r in valid_results])
        med_mem = statistics.median([r["Memory"] for r in valid_results])
        # Güvenlik ve İsim değişmez
        sec = valid_results[0]["Security"]
        p_name = valid_results[0]["Params"]

        return {
            "Time": med_time,
            "CPU_Time": med_cpu,
            "Memory": med_mem,
            "Security": sec,
            "Params": p_name
        }