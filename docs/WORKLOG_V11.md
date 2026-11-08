# METARGON v0.11 — çalışma günlüğü / durum

**Dal:** `research/v11-noise` (main'e merge edilmedi)

## Sonuçlar (resmi, 8 tohum × 100 bölüm)

### F ailesi (7 koşul)
| koşul | v0.10 | v0.11 | Bayes | v11−v10 | v11−Bayes |
|---|---:|---:|---:|---:|---:|
| stable | 80.62 | 74.62 | 77.50 | −6.00 | −2.88 |
| shift | 86.88 | 86.25 | 80.62 | −0.63 | +5.62 |
| rapid_shift | 83.62 | 86.50 | 76.88 | +2.88 | +9.62 |
| iid | 67.75 | 68.25 | 79.00 | +0.50 | −10.75 |
| novel | 89.75 | 86.75 | 78.50 | −3.00 | +8.25 |
| **noisy_obs** | 45.12 | **84.88** | 73.12 | **+39.75** | **+11.75** |
| tight_budget | 82.62 | 85.88 | 64.88 | +3.25 | +21.00 |

### H ailesi (yapısal yenilik)
| koşul | v0.10 | v0.11 | Bayes | v11−v10 | v11−Bayes |
|---|---:|---:|---:|---:|---:|
| stable | 79.88 | 68.00 | 66.88 | −11.88 | +1.13 |
| novel | 79.88 | 74.25 | 68.12 | −5.62 | +6.13 |

### Gürültü ablasyonları (F, noisy_obs)
| varyant | başarı | vs v11 |
|---|---:|---:|
| v11 (filtered replay) | 84.88 | — |
| -cf (onay yok) | 73.12 | +11.75 |
| -ct (kesinlik yok) | 86.62 | −1.75 |
| -rp (replay yok) | 80.50 | +4.37 |
| -rw (replay ağırlıklı) | 79.62 | +5.25 |

Teşhis: sahte doğrulama 44.9 → 7.3/ep; replay 7171 → 482/ep; verified
47.5 → 9.0/seed-run; CPU noisy 151 s → 57 s; peak bellek 0.23 → 0.13 MB.

## Durum
- [x] Teşhis + mekanizma + testler (22/22 hızlı; tam paket arka planda)
- [x] Resmi F/H/bellek koşuları (exit 0) + tablolar + `RESEARCH_V11.md`
- [ ] Tam test paketi sonucu → commit + push → PR (token yok, kullanıcı açacak)
- [ ] İsteğe bağlı: -cf/-ct'nin F-stable'daki nötrlüğü not edildi (rapor §7)
