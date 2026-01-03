import time
import statistics
import tracemalloc
from Crypto.Cipher import AES, ChaCha20_Poly1305, DES3, Blowfish, CAST
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad

class CryptoBenchmark:
    def __init__(self):
        # Ağırlıklar: Multi-Objective Weighted Sum yaklaşımı
        # Bu ağırlıklar literatürdeki "Balanced Security-Performance" çalışmalarına atıftır.
        self.wT = 0.35
        self.wCPU = 0.25
        self.wM = 0.05
        self.wS = 0.35
        self.PENALTY = 1e9 

    def get_scientific_security_score(self, algo_name, key_len_bytes, mode_name=None):
        """
        Güvenlik Puanlama Metodolojisi:
        NIST SP 800-57 Part 1 Rev 5 ve ECRYPT-CSA önerilerine dayalı 'Ordinal Ranking'.
        
        Sıralama (Güçlüden Zayıfa):
        1.00: AES-256 (Post-Quantum direnci yüksek)
        0.95: ChaCha20-Poly1305 (256-bit, Modern, AEAD)
        0.85: AES-192
        0.70: AES-128 (Günümüz standardı)
        0.50: Blowfish/CAST5 (64-bit blok boyutu zafiyeti - Sweet32 saldırısı riski)
        0.20: 3DES (NIST tarafından 'Disallowed' statüsünde - Legacy)
        """
        bits = key_len_bytes * 8
        score = 0.0

        if algo_name == "AES":
            if bits >= 256: score = 1.00
            elif bits >= 192: score = 0.85
            else: score = 0.70
            
        elif algo_name == "ChaCha20":
            # ChaCha20 anahtarı sabittir (256-bit). Key parametresi burada etkisizdir.
            score = 0.95
            
        elif algo_name in ["Blowfish", "CAST5"]:
            # 64-bit blok boyutu cezası (Sweet32)
            score = 0.50
            
        elif algo_name == "3DES":
            # Legacy ceza
            score = 0.20

        # AEAD ve Mod Bonusu/Cezası
        if mode_name == "GCM" or algo_name == "ChaCha20":
            score += 0.05 # Authenticated Encryption (Integrity) bonusu
        
        return min(max(score, 0.0), 1.0)

    def _run_single_test(self, params, data):
        """
        Tekil Test Koşucusu.
        params: [AlgoID, ModID, KeyIdx, BufferSize]
        Not: Sürekli (Continuous) optimizer çıktıları, burada ayrık (Discrete) değerlere 
        'Rounding' yöntemiyle map edilmektedir.
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

            # --- Kripto Kurulum ---
            if algo_name == "AES":
                key_sizes = [16, 24, 32]
                k_len = key_sizes[min(key_idx, 2)]
                key = get_random_bytes(k_len)
                
                if mode_val == 1: # GCM
                    mode_name = "GCM"
                    nonce = get_random_bytes(12) 
                    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
                    # GCM için AAD (Simülasyon)
                    cipher.update(b"HeaderData")
                    
                elif mode_val == 2: # CTR
                    mode_name = "CTR"
                    # CTR Modu: 64-bit Nonce + 64-bit Counter (PyCryptodome Default)
                    # Counter overflow riski 2^64 bloktan sonra vardır, bu test için güvenlidir.
                    nonce = get_random_bytes(8)
                    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
                else: # CBC
                    mode_name = "CBC"
                    iv = get_random_bytes(16)
                    cipher = AES.new(key, AES.MODE_CBC, iv=iv)

            elif algo_name == "ChaCha20":
                mode_name = "AEAD"
                # ChaCha20 Key 256-bit sabittir. optimizer'ın key_idx'i burada yok sayılır.
                key = get_random_bytes(32) 
                nonce = get_random_bytes(12)
                cipher = ChaCha20_Poly1305.new(key=key, nonce=nonce)
                cipher.update(b"HeaderData")

            elif algo_name == "3DES":
                mode_name = "CBC"
                k_len = 16 if key_idx == 0 else 24
                key = get_random_bytes(k_len)
                iv = get_random_bytes(8)
                cipher = DES3.new(key, DES3.MODE_CBC, iv=iv)

            elif algo_name == "Blowfish":
                mode_name = "CBC"
                bs_keys = [16, 32, 56]
                key = get_random_bytes(bs_keys[min(key_idx, 2)])
                iv = get_random_bytes(8)
                cipher = Blowfish.new(key, Blowfish.MODE_CBC, iv=iv)
            
            else: # CAST5
                mode_name = "CBC"
                key = get_random_bytes(16)
                iv = get_random_bytes(8) 
                cipher = CAST.new(key, CAST.MODE_CBC, iv=iv)

            # --- Buffer Alignment (CBC Fix) ---
            blk_size = getattr(cipher, 'block_size', 1)
            if blk_size > 1:
                remainder = buffer_size % blk_size
                buffer_size -= remainder 
                if buffer_size == 0: buffer_size = blk_size

            # Padding
            data_proc = data
            if mode_name == "CBC":
                data_proc = pad(data, blk_size)

            # --- ÖLÇÜM ---
            # Sınırlılık Notu: tracemalloc Python heap'ini ölçer. C-level allocation (OpenSSL vb.)
            # tam yansımayabilir ancak algoritmik karmaşıklık (Overhead) için bir proxy olarak kullanılır.
            tracemalloc.start() 
            start_cpu = time.process_time()
            start_wall = time.perf_counter()

            total_len = len(data_proc)
            
            # Şifreleme Döngüsü
            if mode_name in ["GCM", "AEAD"]:
                for i in range(0, total_len, buffer_size):
                    chunk = data_proc[i:i+buffer_size]
                    cipher.encrypt(chunk)
                tag = cipher.digest() 
            else:
                for i in range(0, total_len, buffer_size):
                    chunk = data_proc[i:i+buffer_size]
                    cipher.encrypt(chunk)

            end_wall = time.perf_counter()
            end_cpu = time.process_time()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Metrikler
            wall_ms = (end_wall - start_wall) * 1000
            cpu_ms = (end_cpu - start_cpu) * 1000
            mem_mb = peak_mem / (1024 * 1024)
            
            sec_score = self.get_scientific_security_score(algo_name, len(key), mode_name)

            return {
                "Time": wall_ms,
                "CPU_Time": cpu_ms,
                "Memory": mem_mb,
                "Security": sec_score,
                "Params": f"{algo_name}-{mode_name}"
            }

        except Exception as e:
            return None

    def benchmark_with_stats(self, params, data, repeats=5):
        valid_results = []
        # Isınma turu
        self._run_single_test(params, data)
        
        for _ in range(repeats):
            res = self._run_single_test(params, data)
            if res: valid_results.append(res)
        
        if not valid_results: return None
        
        # İstatistiksel Düzeltme: Tüm metriklerin medyanı alınır.
        # Güvenlik skoru deterministik olsa da kod tutarlılığı için listeden alınır.
        return {
            "Time": statistics.median([r["Time"] for r in valid_results]),
            "CPU_Time": statistics.median([r["CPU_Time"] for r in valid_results]),
            "Memory": statistics.median([r["Memory"] for r in valid_results]),
            "Security": statistics.median([r["Security"] for r in valid_results]), 
            "Params": valid_results[0]["Params"]
        }