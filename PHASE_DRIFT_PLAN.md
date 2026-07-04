# Phase-Drift Programı — Yol Haritası (Faz 5+)

> Hipotez: ST-LRPS pozisyon hatasının baskın kısmı yörünge-şekli hatası değil,
> küçük bir ortalama tanjansiyel ivme bias'ının (⟨Δa_T⟩) Gauss varyasyonel
> denklemleri üzerinden ürettiği **along-track faz kaymasıdır**
> (⟨Δa_T⟩ → δa → δn → t² ile büyüyen along-track hata).
>
> Durum (2026-07-04): Faz 1–4 TAMAM. Teşhis zinciri
> `src/lunaris/surrogate/st_lrps/evaluation/phase_diagnostics.py` içinde;
> benchmark artık her senaryo için `phase_lag_final_s`,
> `phase_lag_slope_s_per_day`, `phase_corrected_rms_km`,
> `phase_explained_fraction` kolonlarını, `selected_*_phase_lag_all_models.png`
> ve `phase_diagnostics_summary.png` figürlerini üretiyor;
> `benchmark_validation.py` PHASE_COLUMNS kontratını denetliyor
> (eski artefaktlar `metrics: {phase: false}` ile açıkça muaf tutulur).

Sıra: **G0 → (karar) → Faz 5C → (karar) → Faz 5B? → Faz 6 → Faz 7.**
Her kapının kabul kriteri önceden sabittir; kriter tutmazsa sonraki faza
geçilmez ve negatif sonuç raporlanır (o da yayınlanabilir bir bulgudur).

---

## G0 — Gerçek koşu ile hipotez doğrulama (kapı, kod değil veri işi)

Amaç: Faz 1–4 altyapısını gerçek ST-LRPS modeliyle çalıştırıp önceden
sabitlenmiş kabul kriterini test etmek.

**G0a — Standart benchmark koşusu (mevcut altyapı, yeni kod yok)**
- `lunaris-benchmark` gpu-batch modu, gerçek ST-LRPS model dizini, sentetik
  değil (`paper_safe` disiplinine uygun; reproducible-benchmarks akışı).
- Öneri: ≥ 64 senaryo, 5 gün süre, mevcut truth cache varsa yeniden kullan.
- Çıktı: `scenario_results.csv` faz kolonları + özet figürler.

**G0b — Nedensel test + UQ hizalanması (küçük script/CLI işi, ~yarım gün)**
- Benchmark satır metrikleri Δa örneklemeleri içermez; seçili (best/repr/worst
  + birkaç rastgele) senaryo için cache'lenmiş truth trajesi üzerinde
  `Δa(t) = a_model(r_gt) − a_truth(r_gt)` değerlendirip
  `compute_phase_diagnostics(..., da_m_s2=...)` çağıran ince bir CLI yaz
  (örn. `python -m lunaris.surrogate.st_lrps.evaluation.phase_report`).
  Rapor: τ_pred vs τ overlay, tau_pred_measured_corr, tau_pred_final_ratio,
  δa_meas vs δa_pred.
- UQ hizalanması için: aynı başlangıç durumu etrafında bir batch ensemble
  koşusu → `uq_covariance.npz` → `load_uq_covariance_history` +
  `interpolate_covariance_to_times` + `compute_uq_alignment`.
  Rapor `UQ_ALIGNMENT_CAVEAT` metnini aynen taşımak zorunda.

**G0 kabul kriteri (hipotez DOĞRULANDI sayılır ⇔ üçü birden):**
1. `phase_explained_fraction` medyanı > 0.8 ve along-track MS payı baskın
   (along ≫ radial + cross),
2. `tau_pred_measured_corr` > 0.95 ve `tau_pred_final_ratio` ∈ [0.7, 1.3]
   (seçili senaryolarda),
3. ⟨Δa_T⟩ işareti/mertebesi senaryolar arasında tutarlı (5C'nin ön şartı).

Kriter tutmazsa: Faz 5 iptal; bulgular `scientific-reporting` disipliniyle
negatif sonuç olarak yazılır; radial/cross hata anatomisine dönülür.

---

## Faz 5C — Orbit-averaged tanjansiyel bias kalibrasyonu (öncelikli düzeltme)

Fiziksel, savunulabilir düzeltme: modelin sistematik tanjansiyel bias'ını
sabit/irtifa-bağımlı bir kalibrasyonla çıkarmak. "NN sihri" değil,
"küçük orbit-averaged bias uzun-ufuk faz kayması üretir; kalibre ediyoruz."

**5C-1 — Kalibrasyon ölçümü (yalnız TRAIN yörüngeleri)**
- TRAIN split senaryolarında truth trajeleri boyunca ⟨Δa_T⟩ ölç
  (`diagnose_tangential_bias`), irtifa bin'lerine ayır.
- Model seçimi (artan karmaşıklık, gerektiği kadarı): (a) tek sabit
  δa_T; (b) irtifaya lineer; (c) irtifa-binli lookup. Seçim kriteri:
  train-içi senaryolar arası tutarlılık (işaret aynı, varyasyon
  < ~%50). Tutarsızsa 5C burada DURUR (state-bağımlı bias → 5B'ye bak).
- Çıktı: kalibrasyon sabiti/tablosu + üretildiği koşunun hash'i (provenance).

**5C-2 — Runtime entegrasyonu (mimari kısıtlar netleşti, dikkat)**
- **Kısıt 1:** Runtime gravity provider pozisyon-only ve body-fixed
  (`acceleration_fixed(x_m)`); hız provider'a ulaşmıyor. Düzeltme provider
  İÇİNE giremez → hızın mevcut olduğu **propagasyon RHS katmanında** ayrı bir
  kuvvet terimi olarak eklenir (SRP/third-body deseni):
  `a_corr = a_total + δa_T(alt) · v̂` (inertial frame, v̂ = inertial hız yönü).
- **Kısıt 2:** Terim hız-bağımlı → **non-conservative**; `is_conservative`
  bayrağı False olmalı ki mevcut symplectic guard doğru tetiklensin
  (symplectic integratörlerle kombinasyonu guard reddetmeli).
- **Kısıt 3:** Import kontratları — `ST-LRPS runtime (inference path) stays
  light`: kalibrasyon uygulaması evaluation/training'e import açamaz;
  kalibrasyon değerleri config/artifact üzerinden veri olarak taşınır.
- Config: SimConfig SSOT akışına yeni alan (örn.
  `surrogate_tangential_bias_correction: {enabled: false, table: ...,
  calibration_provenance: ...}`); varsayılan KAPALI (fail-closed).
  Manifest/run_config echo'suna kalibrasyon sabiti + kaynak hash yazılır.
- Testler: birim (δa_T·v̂ doğru frame/işaret), symplectic guard reddi,
  provenance echo, kapalıyken bit-identik sonuç.

**5C-3 — Değerlendirme (TEST/OOD, düzeltme açık vs kapalı)**
- Aynı benchmark, iki koşu (on/off), aynı seed/senaryolar.
- Kabul: raw RMS medyanı anlamlı düşer (hedef ≥ 3×; gerçek hedef G0
  ölçümünden türetilir: faz payı %90 ise ~3–10× beklenir), radial/cross RMS
  kötüleşmez (< %10 artış), `phase_lag_final_s` medyan |τ| ~10× düşer.
- Inference'ta GT kullanılmadığının kanıtı: kalibrasyon yalnız TRAIN
  split'ten; TEST/OOD sonuçları st-lrps-evidence-audit kriterlerinden geçer.
- Rapor dili: "calibrated surrogate"; düzeltilmiş sonuç ana metrik olarak
  ancak kalibrasyon metodolojisi açıkça beyan edilerek sunulur.

Tahmini efor: 5C-1 ~1 gün (script + analiz), 5C-2 ~1–2 gün (force terimi +
config + testler), 5C-3 ~yarım gün + koşu süreleri.

---

## Faz 5B — Öğrenilmiş online faz-hızı düzeltmesi (KOŞULLU, varsayılan: yapma)

Yalnızca şu ikisi birden doğruysa açılır: (i) 5C-1 bias'ın state-bağımlı
olduğunu gösterdi (sabit/irtifa modeli tutarsız), (ii) 5C-3 hedefe ulaşamadı.
- Küçük `f_φ(r, v, alt, ...) → δa_T` modeli; yalnız TRAIN trajelerinde eğitilir.
- `experimental` bayrağıyla gelir; paper-safe modda reddedilir (fail-closed).
- Leakage yükü yüksek: eğitim/eval ayrımı, scaler'lar, artifact kontratı
  eksiksiz; st-lrps-evidence-audit'ten geçmeden hiçbir sayı raporlanmaz.
- 5A (GT'ye hizalayarak düzeltilmiş RMS) her zaman yalnız teşhis; asla
  headline metrik değil (kod docstring'leri bunu zaten şart koşuyor).

---

## Faz 6 — Phase-aware training loss (yeniden eğitim gerektirir)

Kök neden eğitimde çözülür; 5C semptom kalibrasyonudur. Sıra önemli:
**önce 5C değerlendirmesi bitmeli** — yeniden eğitilmiş model bias'ı zaten
azaltırsa 5C sabiti YENİDEN kalibre edilmeli (çifte düzeltme tuzağı).

**6-1 — Dataset şemasına auxiliary velocity (ön şart, en ucuz kazanım)**
- Kısıt: mevcut örnekler pozisyon→(U, a); hız yok → gerçek along-track loss şu
  an imkânsız (losses.py bu yüzden bilinçli "radial vs cross-radial" yapıyor).
- Üretici trajelerde hız zaten var → dataset'e opsiyonel `velocity` kolonu
  (contract sürümü artar, geriye uyumlu: alan yoksa eski davranış).
- Hız yalnız loss ağırlığı yönü için kullanılır, model GİRDİSİ DEĞİL
  (model pozisyon-only kalır; runtime kontratı değişmez, scaler'lar etkilenmez).
- dataset_validation + artifact contract güncellemesi + testler.

**6-2 — Gerçek RTN-ağırlıklı loss**
- Mevcut radial/cross altyapısının doğal uzantısı: örnek hızından T̂ üret,
  `along_loss_weight` ekle (config + CLI + config_summary + ramp davranışı
  direction-loss deseninde).
- Ablation matrisine yeni satır (A7: +along-weight); A6'ya karşı test/OOD
  RMSE-a VE faz metrikleriyle (benchmark kolonları) karşılaştırılır.

**6-3 — Bias-penalty loss (muhtemelen en hedefli terim)**
- Faz kaymasının nedeni pointwise hata değil, ORTALAMA tanjansiyel bias →
  doğrudan cezalandır: `L_bias = ⟨(Δa·v̂)⟩²_batch` (batch-ortalama işaretli
  tanjansiyel hata karesi). Ucuz, mevcut loss'a eklenir, λ ile.
- Dikkat: batch kompozisyonuna duyarlı (irtifa karışımı) —
  altitude-balanced örnekleme ile birlikte değerlendirilir.

**6-4 — Short-window propagation loss (ERTELENDİ — diffprop'a bağlı)**
- K-adım differentiable RK4 ile kısa propagasyon, along-track state hatası
  cezası. Doğru ama pahalı; roadmap'teki diffprop uzantısıyla birleşik ayrı
  bir deney olarak planlanır. Bu plan kapsamında yalnız tasarım notu.

**6-5 — Checkpoint seçimi**
- Hybrid val skoruna opsiyonel faz-proxy terimi (örn. val kümesinde işaretli
  tanjansiyel bias büyüklüğü). Varsayılan kapalı; ablationla gerekçelendir.

Kabul (Faz 6 toplam): yeniden eğitilmiş model, düzeltmesiz halde 5C'li eski
modelin benchmark faz metriklerine yaklaşır veya geçer; test/OOD accel RMSE
kötüleşmez; evidence-audit temiz.

---

## Faz 7 — Raporlama / UI fan-in (düşük öncelik)

- Results zone'a faz teşhis paneli (UI roadmap P1a ile birlikte):
  scenario_results faz kolonları + phase_diagnostics_summary.png.
- Paper figürleri: τ_pred vs τ overlay (nedensel kanıt figürü), raw-vs-corrected
  saçılım, UQ hizalanma zaman serisi (caveat metniyle) — scientific-figures
  disipliniyle.

---

## Riskler ve tuzaklar (uygulama sırasında hatırla)

1. **Eksantrik yörüngeler:** τ_pred zincirinin δn→δM→τ adımı near-circular;
   e > ~0.1 senaryolarda τ_pred sapması beklenir — G0b'de eksantriklik
   bin'lerine göre raporla, kriteri near-circular alt kümede uygula.
2. **Çifte düzeltme:** Faz 6 sonrası 5C sabiti eski modele göre kalibre
   kalmışsa düzeltme ters yönde bias üretir → her model artefaktı kendi
   kalibrasyonunu taşır (artifact manifest'e bağla), model↔kalibrasyon
   eşleşmesini runtime'da doğrula (hash).
3. **Symplectic guard:** δa_T·v̂ non-conservative; guard'ın bu kombinasyonu
   reddettiğini test et (sessiz enerji drifti üretme).
4. **Import kontratları:** düzeltme runtime'da, teşhis evaluation'da kalır;
   `lint-imports` her PR'da (10 kontrat). phase_diagnostics import-hafif
   kalmalı (torch/matplotlib/numba yok; `_ric_basis` lazy importu bu yüzden).
5. **Frame tuzağı:** ST-LRPS body-fixed, trajeler inertial; Δa
   değerlendirmesinde model ivmesi inertial'e döndürülmüş halde
   karşılaştırılmalı (benchmark zaten böyle yapıyor — G0b script'i aynı yolu
   kullansın, kendi frame dönüşümünü yazmasın).
6. **Validasyon geriye uyumluluğu:** eski benchmark artefaktları PHASE_COLUMNS
   nedeniyle fail eder; yeniden validasyonda `metrics: {phase: false}` açıkça
   verilir (sessiz muafiyet yok).

## Özet sıra ve kapılar

| Adım | İş | Kapı kriteri | Tahmini efor |
|---|---|---|---|
| G0a | Gerçek benchmark koşusu | fraction > 0.8, along baskın | koşu süresi |
| G0b | phase_report CLI + ensemble UQ | corr > 0.95, ratio 0.7–1.3, bias tutarlı | ~1 gün |
| 5C-1 | Train-split kalibrasyon ölçümü | bias tutarlı (işaret + <%50 varyasyon) | ~1 gün |
| 5C-2 | RHS force terimi + config + guard | testler + provenance | ~1–2 gün |
| 5C-3 | On/off TEST-OOD değerlendirmesi | RMS ≥3×↓, radial/cross ≤%10↑ | ~0.5 gün + koşu |
| 5B | (koşullu) öğrenilmiş düzeltme | 5C yetersiz VE bias state-bağımlı | ayrı plan |
| 6-1..6-3 | velocity kolonu + RTN/bias loss + A7 | audit temiz, faz metrikleri ↓ | ~3–5 gün |
| 6-4 | propagation loss | diffprop hazır olunca | ertelendi |
| 7 | UI/paper fan-in | — | düşük öncelik |
