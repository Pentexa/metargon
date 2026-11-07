# METARGON v0.11 — çalışma günlüğü / açık işler

**Tarih:** 8 Kasım 2026, gece (WIP checkpoint — 02:00)
**Dal:** `research/v11-noise` (main'e merge edilmedi)

## Bugün yapılanlar
1. Gürültü teşhisi (v0.10): v10 4.3× sahte olay inanıyor; doğrulanmış
   hipotezlerin ~%96'sı sahte; replay ~7k uygulama/bölüm ile amplifikasyon.
   → `results/v11/diagnostics_noise_v10.json`
2. v0.11 mekanizması (yalnızca kamu bilgisi, aile-agnostik):
   - Doğrulanmış olay akışı (`robust_belief.py`): 2 ardışık tutarlı okuma
     veya hareket kanıtı; aynı yönlü dedupe penceresi; tek bitlik
     hatalar düşürülüyor.
   - Kesinlik-ağırlıklı kanıt (`compositional_memory_v10.py`): bağlam
     üyeliği filtrelenmez, ağırlık = stable sensör oranı (min 0.25).
   - Replay modları: `on` (v10 mirası), `off`, `filtered` (önerilen),
     `weighted`.
   - Nonevent kapısı (kararsız kapıda çelişki sayımı yok), koruma
     iyileştirmesi (son olay kredisi kaldırıldı).
3. Yeni ajan: `metargon_v11.py` (MetargonV10 alt sınıfı; ablasyon
   bayrakları `-cf`, `-ct`, `-rp`, `-rw`).
4. Testler: robust belief 8/8, v11 ajanı 6/6, seçili v10 regresyonları
   geçti.
5. Ön ölçümler (aynı tohumlar, gürültü): v11 ~%90 başarı (v10 %76.7);
   sahte doğrulama 44.9 → 7.3/bölüm; replay 7171 → 482/bölüm.

## Yarın kalan işler
- [ ] **Resmi eşleştirilmiş koşular (YENİ KODLA):** önceki arka plan
      koşusu eski kodla başlamıştı → geçersiz. Şunları sıralı koş:
      1. `F: 7 koşul × 8 tohum × 100 bölüm` (`benchmark_v11.py --parallel`)
      2. `H: stable,novel × 8×100`
      3. `bellek örneklemi: stable,noisy × 2×20` (--json ayrı)
- [ ] `v11_postprocess.py` ile `benchmark_table_v11.md` ve `_h.md`
      tablolarını üret; `docs/RESEARCH_V11.md`'deki yer tutucuları
      doldur.
- [ ] Ablasyon tablosu (clean/noisy için -cf/-ct/-rp/-rw) + paired CI.
- [ ] Tam Python test paketi (`unittest discover -s python`).
- [ ] Raporu finalize et; commit + push; bağımsız PR aç (main'e
      merge yok).

## Notlar
- Geçici hata ayıklama script'leri (`dbg_*.py`, `smoke_v11.py`) bu
  commit'e dahil edilmedi.
- Bayes'e dokunulmadı (zayıflatma yasağı korundu).
- `results/v11/` yalnızca teşhis JSON'ları içeriyor; resmi summary'lar
  henüz yok.
