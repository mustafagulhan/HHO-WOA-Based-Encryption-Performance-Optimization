import numpy as np
import time
import collections
import tensorflow as tf
from scipy.signal import butter, filtfilt, convolve
import matplotlib.pyplot as plt
import os

# --- KUTUPHANE KONTROLU ---
try:
    from ppg_lib import max30102
except ImportError:
    print("[HATA] 'ppg_lib' bulunamadi.")
    exit()

# ==========================================
# AYARLAR (ENGINEERING SPECS)
# ==========================================
MODEL_FILENAME = 'lstm_arrhythmia_model.h5'
AI_SEQUENCE_LEN = 10  
VIEW_WINDOW = 400     

# 1. FILTRE (Motion Artifact ve Baseline Wander icin optimize)
LOW_CUT = 0.5
HIGH_CUT = 8.0
FILTER_ORDER = 2
FS_ESTIMATE = 100.0 # Sadece filtre katsayilari ve window size hesabi icin.

# 2. ZAMANLAMA VE REFRACTORY (KRITIK AYARLAR)
# Fizyolojik olarak insan kalbi max 200-220 atar. 
# 0.25s (250ms) altindaki her tepe gurultudur/yankidir.
REFRACTORY_PERIOD = 0.25 

# 3. GLOBAL BUFFERLAR
raw_buffer = collections.deque(maxlen=VIEW_WINDOW) 
rr_buffer = collections.deque(maxlen=AI_SEQUENCE_LEN) 

# SQI BUFFER
signal_metric_buffer = collections.deque(maxlen=50)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y

def true_elgendi_detector(signal_data, fs):
    """
    GERCEK ELGENDI IMPLEMTASYONU (BLOCKS OF INTEREST)
    1. Squaring
    2. MA_Peak (Kisa) vs MA_Beat (Uzun)
    3. Bloklari bul (Candidates)
    4. Her blogun icindeki MAX degeri sec (Peak)
    """
    signal = np.array(signal_data)
    
    # 1. Squaring (Negatifleri yok et, tepeleri sivrilt)
    squared = np.power(signal, 2)
    
    # 2. Hareketli Ortalamalar (Moving Averages)
    w1 = int(0.12 * fs) # ~120ms (QRS/Systolic genisligi)
    w2 = int(0.60 * fs) # ~600ms (Beat genisligi)
    
    # Window size 0 olmasin (Guvenlik)
    if w1 < 1: w1 = 1
    if w2 < 1: w2 = 1
    
    ma_peak = convolve(squared, np.ones(w1)/w1, mode='same')
    ma_beat = convolve(squared, np.ones(w2)/w2, mode='same')
    
    # 3. Blocks of Interest (Ilgi Bloklari)
    # Kisa ortalama, uzun ortalamanin uzerine nerede cikiyor?
    # Beta offset = Mean * 0.02 (Literatür standardi)
    threshold = ma_beat + (np.mean(squared) * 0.02)
    blocks = ma_peak > threshold
    
    # 4. Bloklar icinden Peak Secimi
    peaks = []
    
    # Bloklarin baslangic ve bitislerini bul
    # Diff alarak 0->1 (baslangic) ve 1->0 (bitis) gecislerini buluyoruz
    edges = np.diff(blocks.astype(int))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]
    
    # Kenar durumlari (Eger blok en basta basliyorsa veya en sonda bitiyorsa)
    if blocks[0]: starts = np.insert(starts, 0, 0)
    if blocks[-1]: ends = np.append(ends, len(blocks)-1)
    
    # Her blogu gez
    for s, e in zip(starts, ends):
        # Blok cok kucukse atla (Gurultu)
        if e - s < 3: continue
        
        # Bu araliktaki orijinal sinyalin (veya squared sinyalin) en yuksek noktasini bul
        # Local Maxima
        search_window = signal[s:e+1]
        if len(search_window) == 0: continue
        
        local_max_idx = np.argmax(search_window)
        peaks.append(s + local_max_idx) # Global indeksi kaydet
        
    return np.array(peaks)

def calculate_advanced_sqi(signal_segment):
    """
    GELISTIRILMIS SINYAL KALITE INDEKSI (SQI)
    1. Dynamic Range (Genlik farki cok azsa parmak yoktur)
    2. Clipping (Tavan/Taban yapma)
    3. Noise (Asiri gurultu)
    """
    if len(signal_segment) < 10: return False
    
    sig_min = np.min(signal_segment)
    sig_max = np.max(signal_segment)
    dynamic_range = sig_max - sig_min
    
    # 1. Parmak Yok / Sinyal Yok
    if dynamic_range < 10: return False 
    
    # 2. Clipping Kontrolu (Sensor saturasyonu)
    # MAX30102 max degeri genelde bellidir ama burada goreceli bakalim.
    # Eger sinyalin %20'si ayni max degerdeyse clipping vardir.
    
    # 3. Asiri Gurultu (Std Dev kontrolu)
    std_dev = np.std(signal_segment)
    if std_dev > 1000: return False # Cok yuksek gurultu
    
    return True

def init_system():
    print("="*60)
    print("   ARITMI ANALIZ SISTEMI (v6.0 - Engineering Edition)")
    print("   Fixes: True Elgendi, Time-Domain Refractory, Logic Bugs")
    print("="*60)
    
    if not os.path.exists(MODEL_FILENAME):
        print(f"[HATA] {MODEL_FILENAME} yok.")
        exit()
    
    print("[INIT] Model Yukleniyor...")
    model = tf.keras.models.load_model(MODEL_FILENAME, compile=False)
    
    print("[INIT] Sensor Baglaniyor...")
    try:
        sensor = max30102.MAX30102()
    except:
        print("[HATA] Sensor bulunamadi!")
        exit()
        
    print("[INIT] Grafik Arayuzu...")
    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Dark Mode
    fig.patch.set_facecolor('#121212') 
    ax.set_facecolor('#121212')
    line, = ax.plot([], [], color='#00ffcc', linewidth=1.5) 
    
    ax.set_xlim(0, VIEW_WINDOW)
    ax.set_ylim(-3, 3) 
    ax.grid(True, alpha=0.1, color='white', linestyle=':')
    
    ax.tick_params(colors='gray')
    for spine in ax.spines.values(): spine.set_edgecolor('#333333')

    # Status Bar
    info_box = dict(boxstyle='round', facecolor='#222222', alpha=0.9, edgecolor='#444444')
    txt_bpm = ax.text(0.02, 0.92, "NABIZ: --", transform=ax.transAxes, fontsize=16, color='white', weight='bold', bbox=info_box)
    txt_status = ax.text(0.02, 0.80, "DURUM: Hazir", transform=ax.transAxes, fontsize=12, color='gray', bbox=info_box)
    
    return model, sensor, fig, ax, line, txt_bpm, txt_status

def main():
    model, sensor, fig, ax, line, txt_bpm, txt_status = init_system()
    
    # LOGIC STATE
    last_beat_time = 0 # Epoch time
    
    print("\n>>> ANALIZ BASLIYOR <<<\n")
    
    try:
        while True:
            # I2C Veri Okuma
            num = sensor.available()
            if num > 0:
                data_chunk = []
                for _ in range(num):
                    red, ir = sensor.read_fifo()
                    if ir > 50000: data_chunk.append(ir)
                
                if not data_chunk:
                    plt.pause(0.001)
                    continue

                raw_buffer.extend(data_chunk)
                
                # Ekrani Guncelleme Dongusu
                if len(raw_buffer) == VIEW_WINDOW:
                    
                    # 1. HAM FILTRELEME (Algoritma icin)
                    # Normalizasyon yapmiyoruz! Algoritma ham genlikleri gormeli.
                    sig_array = list(raw_buffer)
                    filtered_sig = butter_bandpass_filter(sig_array, LOW_CUT, HIGH_CUT, FS_ESTIMATE, FILTER_ORDER)
                    
                    # SQI KONTROLU
                    if calculate_advanced_sqi(filtered_sig):
                        
                        # 2. TRUE ELGENDI DETECTOR
                        peaks = true_elgendi_detector(filtered_sig, FS_ESTIMATE)
                        
                        # Inverted Check (MI icin) - Eger tepe yoksa tersine bak
                        if len(peaks) == 0:
                            peaks = true_elgendi_detector(-1 * filtered_sig, FS_ESTIMATE)

                        # 3. ZAMAN BAZLI BEAT SECIMI (LOGIC FIX)
                        if len(peaks) > 0:
                            # Son bulunan tepe pencerenin sonunda mi? (Yeni veri mi?)
                            last_peak_idx = peaks[-1]
                            
                            # Pencerenin son %10'u icindeyse yeni atis kabul et
                            if last_peak_idx > VIEW_WINDOW - 20:
                                
                                now = time.time()
                                
                                # ILK ATIS KONTROLU (BUG FIX)
                                if last_beat_time == 0:
                                    last_beat_time = now
                                else:
                                    # REFRACTORY PERIOD KONTROLU (FS Bagimsiz)
                                    time_diff = now - last_beat_time
                                    
                                    if time_diff > REFRACTORY_PERIOD:
                                        # Gecerli RR Araligi (Fizyolojik limit: 0.3s - 1.5s)
                                        if 0.3 < time_diff < 1.5:
                                            rr_buffer.append(time_diff)
                                            last_beat_time = now # Zaman damgasini guncelle
                                            
                                            # --- AI INFERENCE ---
                                            if len(rr_buffer) == AI_SEQUENCE_LEN:
                                                input_seq = np.array(rr_buffer).reshape(1, AI_SEQUENCE_LEN, 1)
                                                risk = model.predict(input_seq, verbose=0)[0][0]
                                                
                                                bpm = int(60.0 / np.mean(rr_buffer))
                                                txt_bpm.set_text(f"NABIZ: {bpm} BPM")
                                                
                                                # GRAY ZONE LOGIC
                                                if risk < 0.4:
                                                    txt_status.set_text(f"NORMAL ({risk:.2f})")
                                                    txt_status.set_color('#00ffcc')
                                                    line.set_color('#00ffcc')
                                                elif risk > 0.6:
                                                    txt_status.set_text(f"RİTİM DÜZENSİZLİĞİ ({risk:.2f})")
                                                    txt_status.set_color('#ff4444')
                                                    line.set_color('#ff4444')
                                                else:
                                                    txt_status.set_text(f"BELİRSİZ / ANALİZ EDİLİYOR ({risk:.2f})")
                                                    txt_status.set_color('orange')
                                                    line.set_color('orange')
                                    else:
                                        # Refractory icindeyse (gurultu/yanki) yoksay
                                        pass
                        
                        # 4. GORSELLESTIRME (NORMALIZED)
                        # Sadece ekran icin Z-Score yapiyoruz
                        disp_mean = np.mean(filtered_sig)
                        disp_std = np.std(filtered_sig)
                        if disp_std < 0.001: disp_std = 1
                        normalized_display = (filtered_sig - disp_mean) / disp_std
                        
                        line.set_ydata(normalized_display)
                        line.set_xdata(np.arange(len(normalized_display)))
                        
                    else:
                        # SQI KOTU
                        txt_status.set_text("DURUM: Sinyal Kalitesi Düşük / Parmak Oynadı")
                        txt_status.set_color('yellow')
                        line.set_color('#444444') # Gri

                    plt.draw()
                    plt.pause(0.001)
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[SISTEM] Kapatiliyor...")
        sensor.shutdown()
        plt.close()

if __name__ == "__main__":
    main()