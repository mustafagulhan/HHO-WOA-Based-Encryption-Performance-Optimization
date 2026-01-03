======================================================================
>>> HHO-WOA / DE / PSO Kriptografik Optimizasyon Başlatılıyor <<<
======================================================================
>>> 8 MB maksimum test verisi hazırlanıyor (dinamik dilimleme)...
>>> Baseline Analizi (Robust Scaling - Median/IQR)...
--------------------------------------------------------------------------------
Iter  | Cost       | Algoritma  | Mod
--------------------------------------------------------------------------------
1     | -0.10724    | AES        | GCM/AEAD
2     | -0.13167    | AES        | GCM/AEAD
3     | -0.13167    | AES        | GCM/AEAD
4     | -0.13211    | AES        | GCM/AEAD
5     | -0.13211    | AES        | GCM/AEAD
6     | -0.13243    | AES        | CTR
7     | -0.13245    | AES        | CTR
8     | -0.13245    | AES        | CTR
9     | -0.13245    | AES        | CTR
10    | -0.13245    | AES        | CTR
11    | -0.13245    | AES        | CTR
12    | -0.13247    | AES        | CTR
13    | -0.13266    | AES        | CTR
14    | -0.13266    | AES        | CTR
15    | -0.13266    | AES        | CTR
16    | -0.13266    | AES        | CTR
17    | -0.13301    | AES        | CTR
18    | -0.13301    | AES        | CTR
19    | -0.13301    | AES        | CTR
20    | -0.13315    | AES        | CTR
21    | -0.13337    | AES        | CTR
22    | -0.13337    | AES        | CTR
23    | -0.13411    | AES        | CTR
24    | -0.13437    | AES        | CTR
25    | -0.13437    | AES        | CTR
26    | -0.13449    | AES        | CTR
27    | -0.13449    | AES        | CTR
28    | -0.13449    | AES        | CTR
29    | -0.13449    | AES        | CTR
30    | -0.13449    | AES        | CTR
--------------------------------------------------------------------------------
>>> DE başlıyor...
DE iter 1/30 | best -0.11113
DE iter 2/30 | best -0.11113
DE iter 3/30 | best -0.11113
DE iter 4/30 | best -0.11113
DE iter 5/30 | best -0.11966
DE iter 6/30 | best -0.12875
DE iter 7/30 | best -0.13209
DE iter 8/30 | best -0.13209
DE iter 9/30 | best -0.13209
DE iter 10/30 | best -0.13209
DE iter 11/30 | best -0.13209
DE iter 12/30 | best -0.13362
DE iter 13/30 | best -0.13380
DE iter 14/30 | best -0.13380
DE iter 15/30 | best -0.13426
DE iter 16/30 | best -0.13426
DE iter 17/30 | best -0.13426
DE iter 18/30 | best -0.13426
DE iter 19/30 | best -0.13426
DE iter 20/30 | best -0.13426
DE iter 21/30 | best -0.13426
DE iter 22/30 | best -0.13426
DE iter 23/30 | best -0.13426
DE iter 24/30 | best -0.13435
DE iter 25/30 | best -0.13438
DE iter 26/30 | best -0.13438
DE iter 27/30 | best -0.13438
DE iter 28/30 | best -0.13438
DE iter 29/30 | best -0.13438
DE iter 30/30 | best -0.13438
--------------------------------------------------------------------------------
>>> PSO başlıyor...
PSO iter 1/30 | best -0.12232
PSO iter 2/30 | best -0.12768
PSO iter 3/30 | best -0.13137
PSO iter 4/30 | best -0.13443
PSO iter 5/30 | best -0.13472
PSO iter 6/30 | best -0.13473
PSO iter 7/30 | best -0.13473
PSO iter 8/30 | best -0.13473
PSO iter 9/30 | best -0.13473
PSO iter 10/30 | best -0.13477
PSO iter 11/30 | best -0.13484
PSO iter 12/30 | best -0.13484
PSO iter 13/30 | best -0.13484
PSO iter 14/30 | best -0.13484
PSO iter 15/30 | best -0.13503
PSO iter 16/30 | best -0.13503
PSO iter 17/30 | best -0.13503
PSO iter 18/30 | best -0.13510
PSO iter 19/30 | best -0.13510
PSO iter 20/30 | best -0.13510
PSO iter 21/30 | best -0.13510
PSO iter 22/30 | best -0.13510
PSO iter 23/30 | best -0.13510
PSO iter 24/30 | best -0.13510
PSO iter 25/30 | best -0.13510
PSO iter 26/30 | best -0.13510
PSO iter 27/30 | best -0.13512
PSO iter 28/30 | best -0.13512
PSO iter 29/30 | best -0.13512
PSO iter 30/30 | best -0.13512
>>> Grafik kaydedildi: robust_result.png

======================================================================
            🏆 GLOBAL OPTIMUM SONUÇ RAPORU (Karşılaştırmalı) 🏆
======================================================================
[HHO-WOA] En İyi Skor: -0.13449
  Algoritma      : AES
  Çalışma Modu   : CTR
  Anahtar        : 256-bit / Max
  Buffer Boyutu  : 11139 bytes (10.88 KB)
  Veri Boyutu    : 1 MB
  Tekrar (repeats): 1
------------------------------------------------------------
[DE] En İyi Skor: -0.13438
  Algoritma      : AES
  Çalışma Modu   : CTR
  Anahtar        : 256-bit / Max
  Buffer Boyutu  : 9069 bytes (8.86 KB)
  Veri Boyutu    : 1 MB
  Tekrar (repeats): 9
------------------------------------------------------------
[PSO] En İyi Skor: -0.13512
  Algoritma      : AES
  Çalışma Modu   : CTR
  Anahtar        : 256-bit / Max
  Buffer Boyutu  : 7464 bytes (7.29 KB)
  Veri Boyutu    : 1 MB
  Tekrar (repeats): 1
------------------------------------------------------------
>>> Analiz Tamamlandı.