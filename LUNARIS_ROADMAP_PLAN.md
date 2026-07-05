# LUNARIS / ST-LRPS Sistem Yol Haritası Planı

> Kaynak: dış sistem review'ı (2026-07-05). Bu dosya o review'ın maddelerini
> (R01–R31) repo gerçekleriyle hizalanmış, kapılı (gated) ve izlenebilir bir
> uygulama planına dönüştürür. Her sprint'in çıkış kapısı (G1–G6) önceden
> sabittir; kapı kriteri tutmadan sonraki sprint'e geçilmez.
>
> Durum güncelleme disiplini: bir madde bittiğinde bu dosyada `[ ]` → `[x]`
> yapılır ve maddeye kanıt (test adı / benchmark artefaktı / commit) eklenir.
> Plan değişiklikleri bu dosyada gerekçeli not olarak bırakılır; sessiz
> kapsam değişikliği yapılmaz.

---

## 0. Konumlandırma — değişmez ilkeler (tüm sprint'lerin anayasası)

İki rol, tek cümlelik sınırlarıyla:

1. **Classical SH path** — tekil yüksek doğruluklu propagasyon için referans
   CPU yolu; paper-safe validation, final frozen-orbit doğrulaması ve
   benchmark truth model YALNIZ bu yoldan üretilir.
2. **ST-LRPS path** — tekil yörüngeyi hızlandırma iddiası TAŞIMAZ; büyük
   ensemble / Monte Carlo / Sobol global search / frozen-orbit screening için
   GPU batch throughput hedefler. Final bilimsel iddialar classical SH ile
   doğrulanır.

Makale ana konumlandırma cümlesi (değiştirilecekse önce bu plan güncellenir):

> "Classical spherical harmonics remain the reference method for individual
> high-fidelity propagation. ST-LRPS targets the repeated-evaluation regime of
> large lunar trajectory ensembles, where high-degree lunar gravity
> evaluations dominate the computational cost."

Model kimliği: **ST-LRPS = SH baseline + learned residual scalar potential**;
`a = a_SH_baseline + grad(residual potential)` (autograd). Bunu bozan her
özellik (doğrudan ivme öğrenme dahil) main dışına çıkar.

---

## 1. Repo gerçekleri — review'ın üzerine oturduğu mevcut durum

Plan yazılırken doğrulanan/bilinen durumlar (review bunların bir kısmını
görmemiş; ilgili maddelerde "kısmen hazır" notu düşülmüştür):

- **Faz-drift teşhis zinciri BÜYÜK ORANDA HAZIR** (R20'yi etkiler):
  `phase_diagnostics.py`, benchmark CSV'de `phase_lag_final_s`,
  `phase_corrected_rms_km`, `phase_explained_fraction` kolonları ve figürleri
  üretiliyor; ayrıntı ve G0 kapısı [PHASE_DRIFT_PLAN.md](PHASE_DRIFT_PLAN.md)
  içinde. R20 "sıfırdan yapılacak iş" değil, entegrasyon/rapor işidir.
- **Capability SSOT zaten kuruldu** (R09'u etkiler): backend yetenek kaydı
  `src/lunaris/core/backend_capabilities.py` içinde (`gpu_st_lrps_potential`
  ve `gpu_st_lrps_direct` burada tanımlı). R09'un işi, `src/lunaris/batch/
  backend_policy.py` tarafında registry dışında kalan manuel liste/karar
  kalıntılarını temizlemek.
- **`is_conservative` bayrağı + symplectic non-conservative guard mevcut**
  (R01'in gerekçesini güçlendirir; force_direct bu bayrağı anlamsızlaştırır).
- **Terrain-aware impact altyapısı mevcut** (`impact_surface_mode` ile gated;
  CPU ground-truth + torch & numba terrain freeze). R12'nin işi yalnız
  paper-safe modda sphere fallback'i hard fail yapmak.
- **force_direct yüzeyi geniş**: `training/force_direct_cli.py`,
  `evaluation/force_direct_eval.py`, `runtime/force_model.py`,
  `shared/contracts.py`, `shared/capabilities.py`, `batch/engine.py`,
  `batch/backend_policy.py`, `core/propagation/…` dahil ~20 dosya. R01 bir
  "flag silme" değil, kontrollü bir amputasyondur.
- **ÇELİŞKİ NOTU:** Eski "ST-LRPS clarity remediation" planının Faz 2'si
  force_direct'i *first-class* yapmayı öngörüyordu. Bu review o kararı
  **tersine çevirir**: force_direct main'den çıkar. Bu plan onaylandığında
  clarity-remediation Faz 2 İPTAL sayılır (Faz 3–4'ün refactor/lock-in
  hedefleri R11 içinde yaşamaya devam eder).

---

## 2. Madde envanteri — öncelik / sprint / bağımlılık matrisi

| ID  | Başlık | Öncelik | Sprint | Bağımlılık | Durum |
|-----|--------|---------|--------|------------|-------|
| R01 | force_direct main'den kaldırılması | P0 | S1 | — | [x] |
| R02 | GPU ST-LRPS "gravity-only" beyanı | P0 | S1 | — | [x] |
| R03 | gpu_st_lrps_third_body backend | P0 | S4 | R07, R06, R11 | [x] |
| R04 | surrogate_assisted_frozen_search workflow | P0 | S5 | R03, R05, R23, R27 | [x] |
| R05 | Frozen / quasi-frozen sınıflandırma modülü | P0 | S5 | — | [x] |
| R06 | VRAM-aware chunking | P1 | S3 | — | [x] |
| R07 | Ortak batched RK4 + impact loop | P1 | S3 | — | [x] |
| R08 | Alive-sample compaction | P1 | S3 | R07 | [x] |
| R09 | Capability registry tek kaynak | P1 | S2 | — | [x] |
| R10 | Dtype provenance düzeltmesi | P1 | S2 | — | [x] |
| R11 | Canonical ST-LRPS runtime | P1 | S3 | R01 | [x] |
| R12 | Terrain fallback paper-safe hard fail | P1 | S2 | — | [x] |
| R13 | SH-only Numba fast-path RHS | P1 | S6a | — | [x] |
| R14 | RHS'de frame-rotation tekilleştirme | P1 | S6a | — | [x] |
| R15 | Fixed-step out-buffer optimizasyonu | P1 | S6a | R07 | [x] |
| R16 | Paper-safe benchmark matrisi | P2 | S6 | R13–R15, R17 | [~] |
| R17 | JIT warm-up protokolü | P2 | S6 | — | [~] |
| R18 | GMAT/STK parity protokolü | P2 | S6 | R13, R17 | [~] |
| R19 | Genişletilmiş validation metrikleri | P2 | S6 | — | [~] |
| R20 | Phase-drift analizi ana validation'da | P2 | S6 | PHASE_DRIFT G0 | [~] |
| R21 | Frozen candidate = classical SH validation kuralı | P2 | S5 | R04 | [x] |
| R22 | Local refinement modülü | P2 | S5 | R04, R05 | [x] |
| R23 | Summary-only output mode | P2 | S3 | — | [x] |
| R24 | run_epoch decomposition | P3 | paralel | — | [x] |
| R25 | Laplacian scaling tutarlılığı | P3 | paralel | — | [x] |
| R26 | Artifact contract sertleştirme | P3 | S2 | — | [x] |
| R27 | Domain guard frozen search'te zorunlu | P3 | S5 | R05 | [x] |
| R28 | print → logger | P4 | paralel | — | [~] |
| R29 | paper_safe / research_mode fail politikası | P4 | S1+S2 | — | [x] |
| R30 | Candidate family JSON raporu | P4 | S5 | R04, R05 | [x] |
| R31 | Frozen orbit plot/report generator | P4 | S5 | R30 | [x] |

---

## Sprint 1 — Scope cleanup (kimlik netleşmesi)

> **Sıralama notu (2026-07-05):** R01 `training/config.py`, training CLI'ları
> ve surrogate runtime'a dokunduğu için aktif eğitim sürerken beklendi;
> kullanıcı eğitimi durdurunca R01 tamamlandı (arşiv → fail-closed → temizlik
> → test/doc). G1 GEÇİLDİ.

### R01 — force_direct main'den kaldırılmalı

**Problem:** ST-LRPS'nin savunulabilir tarafı scalar residual potential
öğrenmesi. force_direct doğrudan ivme öğrenir, conservative-field garantisini
ve `is_conservative`/symplectic-guard zincirini bozar, makale anlatısını
ikiye böler.

**Karar:** Silme değil arşivleme — kod `experimental/force-direct-archive`
branch/tag'ine alınır; main runtime yalnız residual scalar potential /
`potential_autograd` model kind kabul eder.

**Yapılacaklar:**
- [x] Arşiv branch'i + tag oluştur (`experimental/force-direct-archive`);
      README'sine "neden arşivlendi" notu. *(2026-07-05: branch + tag
      `force-direct-archive-20260705` HEAD'de; branch'e FORCE_DIRECT_ARCHIVE_NOTE.md
      commit'lendi.)*
- [x] `core/backend_capabilities.py`: `gpu_st_lrps_direct` kaydını kaldır.
      *(kaldırıldı; REQUIRED_BACKEND_NAMES + registry temiz, yerine arşiv notu.)*
- [x] Main config şemasından force_direct seçeneklerini çıkar
      (`training/config.py`, SimConfig akışı). *(--runtime-model-kind choices
      artık yalnız potential_autograd; pyproject entry-points
      lunaris-train/eval-force-direct kaldırıldı.)*
- [x] Runtime loader force_direct artifact'i **fail-closed** reddetsin
      (açık hata mesajı: "archived in experimental/force-direct-archive").
      *(iki katman: ArtifactContract.validate() ARCHIVED_RUNTIME_MODEL_KINDS
      ile reddediyor; load_surrogate_force_model + gravity_provider ayrıca
      RuntimeError veriyor. Testli: test_runtime_adapter_force_direct.py,
      test_runtime_contract_validation.py, test_artifact_contract.py.)*
- [x] Kaldırma dalgası: `training/force_direct_cli.py`,
      `evaluation/force_direct_eval.py` **silindi**; `runtime/force_model.py`
      (DirectForceRuntime silindi), `shared/contracts.py`,
      `shared/capabilities.py`, `batch/backend_policy.py`+`engine.py`,
      `common/batch_defs.py`, `cli/entrypoints.py`+`batch_runner.py`,
      `networks/models.py`, `training/engine.py`+`compute_accounting.py`,
      `evaluation/{ablation,validation_suite,orbit_drift,runtime_benchmark}.py`,
      `runtime/profiling.py`, `analysis/ensemble/result_audit.py`,
      `core/propagation/{propagator,integrators/fixed_step}.py`,
      `ui/pages/batch_propagation_page.py` temizlendi. Symplectic guard artık
      `is_conservative` bayrağını okuyor (string-match yerine, gelecek-korumalı).
- [x] README + paper anlatısını "scalar residual potential surrogate"
      üzerine sabitle. *(README ST-LRPS glance + backend listesi güncel.)*
- [x] Eski force_direct HPC senaryosu (`st_lrps_force_direct_student_sweep.jsonl`)
      silindi; tools/hpc launcher + docs (HPC/profiling/CONFIG/VALIDATION_HYGIENE/
      BENCHMARK_RESULTS/backend_matrix/PUBLIC_API) güncellendi; roadmap/refactor
      plan docs'a süpersede notu.
- [x] ARCHITECTURE.md ve import-linter kontratlarını güncelle. *(ARCHITECTURE.md
      backend listesi + force_model bölümü güncel; import-linter kontratları
      force_direct'e atıf yapmıyordu, silinen modüller kontratları kırmadı —
      33 import-contract/architecture testi yeşil.)*

**Kabul kriterleri:**
- [x] `rg force_direct src/` yalnız fail-closed hata mesajı/arşiv notu
  noktalarında iz bırakır *(doğrulandı: kalan referanslar contracts.py
  ARCHIVED set + loader/guard reddi + arşiv yorumları).*
- [x] Test: force_direct artifact yüklenmeye çalışılınca anlamlı hata
  *(ArtifactContractError + RuntimeError; testli).*
- [x] Tam suite yeşil *(2368 passed, 9 skipped, 0 failed — 11 dk)*;
  entry-point inventory (`test_repo_hygiene::test_console_scripts_documented_in_public_api`)
  + PUBLIC_API senkron; ruff temiz.

### R02 — GPU ST-LRPS "gravity-only" beyanı

**Problem:** GPU ST-LRPS path full-dynamics propagator değil (third-body,
SRP, albedo, thermal, tides, relativity yok → CPU fallback/unsupported).
"Full-fidelity GPU propagator" izlenimi verilmemeli.

**Yapılacaklar:**
- [x] README: "GPU ST-LRPS currently supports lunar gravity surrogate
      propagation only." *(2026-07-05: README üst callout'a eklendi; aynı
      blokta "not a full-dynamics propagator" + recorded-fallback beyanı.)*
- [x] Paper metni: "The surrogate replaces high-degree lunar gravity
      evaluations; non-gravity perturbations are handled separately in
      validation or future hybrid backends." *(2026-07-05: bağlayıcı cümle
      README callout'unda; manuskript repo dışında, Bölüm 5 sözlüğü esas.)*
- [x] Benchmark raporlarında gravity-only ve full-dynamics sonuçları ayrı
      tablo; karışık tek tablo YASAK. *(2026-07-05: benchmark_pipeline.py
      `FORCE_MODEL_SCOPE="gravity_only"` — report.md'ye scope satırı,
      metrics_summary.json'a `force_model_scope` alanı; pipeline yapısal
      olarak gravity-only (compare_gravity_models hiçbir perturbasyon flag'i
      içermiyor). Testler: test_benchmark_pipeline_smoke,
      test_benchmark_paper_safe, test_benchmark_validation yeşil.)*

**Kabul:** Dokümanlarda GPU ST-LRPS için "full dynamics/fidelity" iması
kalmaz *(doğrulandı: `rg -i "full.(fidelity|dynamics)"` yalnız CPU yolu için
iz bırakıyor)*; benchmark rapor şablonu iki ayrı tablo üretir *(scope alanı
ile etiketli; full-dynamics tablosu R16 matrisiyle gelecek)*.

### R29a — paper_safe politika belgesi (Sprint 1 payı)

- [x] `docs/PAPER_SAFE_POLICY.md`: paper_safe=true iken hangi durumların
      hard fail olduğu tek listede (terrain, ephemeris, backend fallback,
      dtype mismatch, model kind mismatch, gravity file mismatch, domain
      guard). research_mode=true iken warning+fallback+metadata kuralı.
      (Uygulama Sprint 2'de, R29b.) *(2026-07-05: belge yazıldı — 9 koşulluk
      hard-fail tablosu + her koşulun mevcut enforcement durumu
      (Enforced/Partial/Pending, R12/R10/R26/R27/R29b eşlemesiyle);
      docs/README.md ve README doküman tablosuna bağlandı.)*

### G1 kapısı (Sprint 1 çıkışı) — ✅ GEÇİLDİ (2026-07-05)
1. [x] Main'de force_direct çalıştırılabilir yol yok; arşiv branch'i mevcut.
2. [x] README/doc claim'leri Bölüm 5'teki güvenli-claim sözlüğüyle uyumlu.
3. [x] Tam test suite yeşil (2368 passed, 9 skipped).

**Sprint 1 tamam (R01 + R02 + R29a).** Sprint 2'ye (backend correctness:
R09/R10/R12/R26/R29b) geçilebilir.

---

## Sprint 2 — Backend correctness (güven zemini)

### R09 — Backend capability registry tek kaynak

- [x] `batch/backend_policy.py` içindeki manuel unsupported-feature
      listelerini kaldır; tüm destek/fallback kararları
      `core/backend_capabilities.py` üzerinden (`unsupported_force_models
      (backend_name, flags)` tek API). *(2026-07-05: `_st_lrps_gpu_unsupported_features`
      elle listeyi `unsupported_force_models("gpu_st_lrps_potential", flags)`'e
      delege ediyor; başka elle force literal'i kalmadı.)*
- [x] Yeni force flag eklerken yalnız `backend_capabilities.py` değişsin —
      bunu koruyan kontrat testi. *(test_backend_capabilities.py:
      `test_backend_policy_holds_no_hardcoded_perturbation_flag_literals` +
      `test_st_lrps_gpu_unsupported_delegates_to_registry`.)*

**Kabul:** backend_policy.py force listesi tutmuyor; capability registry
dışında destek kararı veren kod yolu yok. *(doğrulandı; 54 test yeşil.)*

### R10 — Dtype provenance mismatch

- [x] `effective_dtype` tek yerden resolve edilsin (tek fonksiyon; policy ve
      runtime aynı kaynağı kullanır). *(2026-07-05: `resolve_effective_dtype`
      + `DtypeResolution` core/backend_capabilities.py'de tek kaynak;
      dtype_support registry'den okunuyor, desteklenmeyen istek `downgraded`
      + reason ile kaydediliyor.)*
- [x] Metadata alanları: `requested_dtype`, `effective_dtype`, `dtype_downgraded`
      + mevcut `model_dtype`/`state_dtype`/`device`/`backend`. *(BatchBackendPlan'a
      requested/effective/downgraded eklendi; __post_init__ tek-dtype return
      site'larını yansıtıyor; engine output metadata'sı bunları çıkarıyor.)*
- [ ] float32/float64 benchmark sonuçları ayrı raporlanır (R16 matrisine
      girdi). *(R16'ya ertelendi — benchmark matrisi işi.)*

**Kabul:** Hardcoded float32 varsayımı kalmaz *(ST-LRPS GPU planı artık
config torch_dtype'ı `resolve_effective_dtype` ile çözüyor)*; test config
float64 isterken provenance'ta float64 görür *(test_batch_gpu_policy::
test_policy_st_lrps_gpu_dtype_provenance_honors_config).*

### R12 — Terrain fallback paper-safe modda hard fail

- [x] Strict/paper-safe + terrain/topography payload yok ⇒ RuntimeError;
      research mode ⇒ warn + sphere fallback + metadata `terrain_fallback=
      sphere`. *(2026-07-05: `BatchPropagationEngine._build_propagator`'a
      politika katmanı: terrain istendi ama payload None ise `_fallback_forbidden()`
      iken RuntimeError, değilse warn + `self._terrain_fallback="sphere"`;
      output metadata'ya `terrain_fallback` alanı.)*
- [x] Mevcut `impact_surface_mode` gating'ine ekleme olarak uygulandı
      (`_resolve_topo_payload` None dönünce devreye girer).

**Not:** Batch config'te ayrı `paper_safe` alanı yok; engine'in mevcut strict
sinyali `sh_fallback_policy="error"` (+ `_fallback_forbidden` sarmalanmış
config'lerde paper_safe/strict_backend/benchmark_mode'u da onurlandırır).
Dedike `paper_safe` batch alanı R29b kapsamında değerlendirilecek.

**Kabul:** strict altında silent simplification yok; fallback metadata'da
görünür. İki yönlü test: test_terrain_paper_safe_fallback.py (fail +
fallback-kayıtlı), her ikisi yeşil.

### R26 — Artifact contract sertleştirme

- [x] Zorunlu alanlar: `model_kind, runtime_kind, body, gravity_model_name,
      gravity_model_hash, baseline_degree, residual_degree_range,
      train_altitude_min_km, train_altitude_max_km, scaler_source, x_scale,
      u_scale, dtype, architecture, parameter_count, training_data_hash,
      validation_data_hash, git_commit, created_at, training_config_hash,
      loss_config, domain_guard_policy`. *(2026-07-05:
      `contracts.PAPER_SAFE_REQUIRED_METADATA` — 22 mantıksal alan, her biri
      kanonik kaynağına eşlenmiş; üretici tarafı `build_resolved_config` artık
      `dtype`, `domain_guard_policy`, `loss_config`, `parameter_count` yazıyor;
      checkpoint payload'ı `parameter_count` taşıyor.)*
- [x] `paper_safe=true` iken eksik metadata ⇒ inference başlamaz.
      *(`load_surrogate_force_model(paper_safe=True)` →
      `require_paper_safe_metadata` ArtifactContractError; paper-safe benchmark
      `paper_safe_metadata_report_for_run` ile aynı kapıyı uygular, rapor
      manifest'e yazılır.)*
- [x] `tests/test_artifact_contract.py` genişletilir; eski artifact'ler için
      açık legacy-override yolu (research_mode'da, metadata kaydıyla).
      *(5 kontrat-seviyesi test + 3 load-path testi
      (test_runtime_contract_validation.py: accept/reject/legacy-override) +
      benchmark reddi (test_benchmark_paper_safe.py); research yüklemesi
      `paper_safe_metadata.legacy_override=True` + missing listesi kaydeder.
      Commit: 63e14c29.)*

### R29b — Silent fallback / broad exception temizliği (uygulama)

- [x] R29a politikasını koda dök: paper_safe iken missing ephemeris,
      backend fallback (allow_fallback=false), dtype/model-kind/gravity-file
      mismatch ⇒ hard fail; research_mode ⇒ warning + metadata.
      *(2026-07-05: `backend_policy.fallback_forbidden()` tek kaynak; ST-LRPS
      GPU→CPU, dtype downgrade ve classic-SH ikameleri planlama anında raise;
      `select_classic_sh_backend` policy='error'ı artık tüm dallarda onurlandırıyor;
      gravity-file kimliği `_gravity_file_identity_check` (match/mismatch/
      unverified manifest'te, mismatch raise); ephemeris zaten iki katmanda
      fail-closed (bootstrap + `_prepare_ephem` sıfır-tablo guard'ı). Testler:
      test_batch_gpu_policy.py R29b bloğu. Commit: e79d5d2e.)*
- [x] Broad `except Exception` taraması; daraltılamayanlar gerekçeli yorumla
      işaretlenir. *(Bilimsel yol (batch/core/physics/surrogate-runtime)
      sessiz `pass/continue` siteleri tek tek incelendi: sonuç-etkileyen 2 yer
      düzeltildi (adaptive degree tablosu bozuk satır uyarısı; impact-event
      çıkarımı başarısızlığı uyarısı), kalanlar `R29b-justified:` yorumu taşıyor;
      UI/plot/report handler'ları kategorik savunmacı olarak
      PAPER_SAFE_POLICY.md'de belgelendi.)*

### G2 kapısı — ✅ GEÇİLDİ (2026-07-05)
1. [x] paper_safe fail-closed davranışları testli (terrain:
   test_terrain_paper_safe_fallback.py; artifact:
   test_runtime_contract_validation.py + test_benchmark_paper_safe.py; dtype +
   fallback: test_batch_gpu_policy.py R29b bloğu).
2. [x] Capability registry dışında destek kararı yok (kontrat testi yeşil:
   test_backend_capabilities.py, 54+ test).
3. [x] Provenance şeması (dtype alanları dahil) benchmark manifest'inde görünür
   (`numerics.dtype_provenance` — requested/effective/downgraded per backend;
   test_st_lrps_params_summary_provenance.py).

**Sprint 2 tamam (R09 + R10 + R12 + R26 + R29b).** Sprint 3'e (batch
infrastructure: R07/R08/R06/R23/R11) geçilebilir.

---

## Sprint 3 — Batch infrastructure (100k'ya hazırlık)

### R07 — Ortak batched fixed-step RK4 + impact loop

**Problem:** `core/torch_sh_propagator.py` ve GPU ST-LRPS propagator'ı benzer
RK4/output-grid/alive-mask/impact/snapshot mantığını kopyalıyor (DRY ihlali;
impact bugfix'leri iki yerde yapılıyor).

- [x] Yeni modül `src/lunaris/core/batched_fixed_step.py`:
      `build_output_grid`, `rk4_step`, alive mask, impact detection
      (terrain/sphere mode), snapshot yazımı, callback/progress, result
      packaging, impact-pozisyon interpolasyonu. *(2026-07-05:
      `build_output_grid`, `rhs_batch`, `rk4_step`, `run_batched_fixed_step`
      ve `BatchedFixedStepResult`; callback modu açık validate edilir.)*
- [x] Backend'ler yalnız acceleration provider verir:
      `class BatchedAccelerationProvider: def acceleration(t, state_batch)
      -> accel_batch`. *(Torch-SH `_SHAccelerationProvider`, ST-LRPS
      `_STLRPSAccelerationProvider`.)*
- [x] Torch SH ve ST-LRPS bu loop'a taşınır; eski loop'lar silinir
      (compat shim gerekirse dinamik fold — ruff static re-export'ları
      siliyor, bilinen tuzak). *(Her iki backend de aynı
      `run_batched_fixed_step` yolundan geçiyor; eski Torch loop kopyaları
      kaldırıldı.)*
- [x] Taşınan Numba fonksiyonlarında `@njit` dekoratörlerinin düşmediğini
      doğrula (bilinen tuzak). *(R07 yalnız Torch fixed-step loop'unu ortakladı;
      Numba `@njit` fonksiyon taşınmadı. `rg "@njit"` ile core/dynamics
      dekoratörleri yerinde görüldü.)*

**Kabul:** İki backend bit-uyumlu (veya tolerans-içi) aynı sonuçları ortak
loop'tan üretir; impact davranış testleri tek parametrize suite olur.
`tests/test_batched_fixed_step.py` impact ve chunk invariant davranışını tek
parametrize suite olarak kapsar; Torch-SH bağlantısı
`tests/test_torch_sh_batch_propagator.py` kapsamındadır.

### R08 — Alive-sample compaction

- [x] `alive_indices` compact edilir; yalnız alive state'ler RK4/acceleration
      görür; impacted state'ler output'ta frozen/terminal tutulur.
      *(2026-07-05: `batched_fixed_step._propagate_chunk` — alive set snapshot
      aralığı başına bir kez compact edilir (snapshot başına tek bounded host
      sync); hot loop sabit şekilli, sync'siz mask mantığını korur. Satır
      bağımsız RK4/provider ⇒ bit-uyum korunur (torch-SH batch==individual /
      chunk-invariance testleri aynen yeşil).)*
- [x] Perf testi: batch'in %70'i impact ettiğinde adım maliyeti ~%70 düşer
      (toleranslı assert, ör. ≥%50 düşüş). *(Deterministik proxy: provider'ın
      değerlendirdiği satır sayısı — %70 t=0-impact senaryosunda tam olarak
      alive satırlar kadar, ≥%50 azalma assert'li;
      `test_compaction_reduces_step_cost_for_impacted_batch`,
      `test_compaction_all_impacted_skips_provider_entirely`,
      `test_compaction_preserves_mixed_impact_results`.)*

### R06 — VRAM-aware chunking

- [x] Runtime VRAM ölçümü + model dtype/param/activation tahmini ⇒ otomatik
      `chunk_size` (küçük GPU 2048–8192, orta 8192–32768, büyük
      32768–262144). *(2026-07-05: `batched_fixed_step.resolve_vram_aware_chunk_size`
      + `query_device_memory` tek kaynak; torch-SH Legendre-workspace tahmini
      ve ST-LRPS MLP-aktivasyon tahmini (`_estimate_bytes_per_sample`) ile her
      iki backend bu resolver'dan geçiyor; tek sample bütçeyi aşarsa preflight
      raise.)*
- [x] OOM yakalanırsa chunk_size yarıya düşer ve devam eder (recover testi).
      *(`run_batched_fixed_step` chunk döngüsü CUDA OOM'da yarılayıp retry
      eder, chunk=1'de re-raise; `test_oom_recovery_halves_chunk_and_matches_reference`
      (sonuç bit-uyumlu) + `test_oom_at_chunk_one_reraises`.)*
- [x] Gerçek chunk_size provenance'a yazılır; chunk sonuçları tek output'ta
      birleşir. *(metrics: `chunk_size_requested` / `chunk_size_effective` /
      `oom_recoveries`; her iki propagator diagnostics_snapshot'a taşıyor.)*

**Kabul:** 100k orbit screening bellek patlatmadan tamamlanır (smoke:
elde varsa gerçek GPU, yoksa küçük-VRAM simülasyonu ile).
*(Küçük-VRAM simülasyonu: resolver band/cap unit testleri + OOM recover
testi; gerçek-GPU 100k smoke'u G3 kapsamında koşulacak.)*

### R23 — Summary-only output mode

- [x] Screening modunda full trajectory saklanmaz; yalnız summary metrics:
      initial/final elements, e_min/max/range, h_peri_min/max/range,
      trend_e, trend_h_peri, omega_behavior, impact flag/time, domain exit
      flag/time, score, validation stage, backend metadata. *(2026-07-05:
      `lunaris/batch/summary.py` — `BATCH_SUMMARY_SCHEMA_VERSION=1`,
      `summarize_ensemble` (post-impact satırlar zarf/trend dışı; ekvatoralde
      periapsis boylamı fallback'i), `SCORE_DEFINITION` v1 belgeli; engine
      `output_mode="summary_only"` dalı `_SummaryOnlyWriter` ile arşivi ve
      (T,N,6) Y_all tahsisini atlar, batch başına özetleyip bloğu bırakır;
      backend metadata diagnostics'te.)*
- [x] Top-K için full state history + element history + diagnostic plot
      verisi saklanır. *(`TopKTrajectoryBuffer` streaming top-K (skor v1,
      impacted/invalid=inf hariç); result.Y yalnız top-K (T,K,6), seçilen
      indeks/skorlar diagnostics'te; element history top-K trajectory'lerden
      türetilebilir.)*

**Kabul:** 100k × N_step × 6 state bellek/disk maliyeti oluşmaz; summary
şeması versiyonlu ve testli. *(tests/test_batch_summary.py: şema/zarf/trend/
omega/inf-skor/merge/top-K/config validasyonu + CPU uçtan uca engine testi —
arşiv dosyası yazılmadığı ve yalnız top-K tutulduğu assert'li.)*

### R11 — Canonical ST-LRPS runtime

**Problem:** `surrogate/runtime/gravity_provider.py` ve
`surrogate/st_lrps/runtime/force_model.py` scaler/domain-guard/model-kind/
prediction sorumluluklarını çiftliyor.

- [x] Yeni `src/lunaris/surrogate/st_lrps/runtime/canonical_runtime.py`:
      artifact loading, model_kind validation, scaler loading, domain guard,
      potential prediction, autograd acceleration, residual+baseline
      composition, metadata/provenance. *(2026-07-05: force_model.py'nin
      tamamı (SurrogateForceModel + loader + frame/rotasyon yardımcıları +
      R26 paper-safe kapısı) canonical_runtime.py'ye taşındı; tensör-yolu
      domain guard'ı `enforce_altitude_envelope_torch` tek-kaynak fonksiyon
      olarak eklendi.)*
- [x] `gravity_provider.py` yalnız adapter olur; `force_model.py` R01
      sonrası sadeleşir/kaldırılır. *(gravity_provider tensör domain guard'ını
      canonical'a delege ediyor; force_model.py dinamik-fold compat shim
      (ruff --fix static re-export tuzağına karşı PEP 562 `__getattr__`).
      NOT — bilinçli istisna: gravity_provider'daki pre-contract legacy
      fallback yolu (kontratsız eski artifact'ler, RuntimeWarning'li) davranış
      koruması için tutuldu; canonical_runtime docstring'inde belgeli.
      Kaldırma/fail-closed kararı kullanıcıya soruldu.)*
- [x] Import kontratı korunur: ST-LRPS runtime (inference path) hafif kalır;
      evaluation/training'e import açılmaz. *(Kontratlar paket-seviyesi;
      import/architecture testleri yeşil (789 pass'lık st_lrps/surrogate/
      import/architecture seçkisi).)*

**Kabul:** scaler/domain/model-kind mantığı yalnız canonical_runtime.py'de;
grep ile doğrulanabilir; davranış regresyon testleri yeşil.
*(tests/test_canonical_runtime_module.py: shim-identity, dinamik-fold,
monkeypatch-through-shim, domain-guard tek-ev grep'i, envelope semantiği,
shim/canonical loader eşitliği.)*

### G3 kapısı — ✅ GEÇİLDİ (2026-07-05)
1. [x] Ortak loop iki backend'de kullanımda, davranış testleri yeşil
   (R07/R08; tests/test_batched_fixed_step.py + test_torch_sh_batch_propagator.py).
2. [x] 100k screening smoke (chunking + compaction + summary-only) çalışıyor
   (`test_g3_100k_screening_smoke_chunked_compacted_summary`: N=100k,
   chunk=8192, ~%33 t=0-impact compaction, summary v1 + top-K; CPU'da tek
   adımlık plumbing smoke — gerçek GPU'da aynı yol tam adım sayısıyla koşar).
3. [x] Canonical runtime tek kaynak; duplicate logic grep'i temiz
   (test_domain_guard_logic_single_home).

**Sprint 3 tamam (R07 + R08 + R06 + R23 + R11).** Sprint 4'e (R03
gpu_st_lrps_third_body) geçilebilir.

---

## Sprint 4 — Hybrid backend: gpu_st_lrps_third_body

### R03 — ST-LRPS + analytic third-body GPU backend

**Kapsam (destekler):** Moon point-mass baseline + lunar high-degree residual
ST-LRPS + Earth third-body + Sun third-body + vectorized ephemeris
interpolation + fixed-step batched propagation (R07 loop'u üzerinde).

**Kapsam dışı (bilinçli):** albedo facets, thermal IR facets, tides,
relativity, adaptive solve_ivp, karmaşık event stack.

```
a_total = a_Moon_pointmass
        + a_Moon_SH_residual_STLRPS
        + a_Earth_third_body
        + a_Sun_third_body
```

- [x] Backend kaydı `core/backend_capabilities.py`'ye eklenir (R09 gereği
      YALNIZ oraya).
- [x] Vectorized ephemeris interpolasyonu: GPU vec3 Catmull-Rom yolu mevcut
      (CPU ile eşleşmesi daha önce doğrulandı) — batch'e genelle.
- [x] Third-body formülasyonu CPU'daki Battin F(q) yoluyla tutarlı olmalı
      (mevcut CPU+GPU üçüncü-cisim testleriyle çapraz doğrulama).
- [x] Provenance: `lunar_gravity_backend=st_lrps`,
      `third_body_backend=analytic_vectorized`,
      `unsupported_forces=[albedo, thermal, tides, relativity, ...]`,
      `effective_dtype`, `chunk_size`.

**Durum notu (2026-07-05, güncel):** R03 kod/doküman/test zemini tamamlandı.
Aynı gün CUDA-enabled ML/HPC ortamında PyTorch CUDA (2.5.1+cu121, GTX 1660 Ti)
kullanılabilir hale geldi ve açık kalan kanıt koşuldu. Not: aktif shell/venv
PyTorch kurulumunu görmüyorsa bu kod hatası değil; CUDA kanıtı aynı ML/HPC
ortamında yeniden koşularak doğrulanır.

**Kabul kriterleri (review'dan aynen):**
1. [x] 10k state batch, CPU fallback OLMADAN GPU'da propagate edilir.
   *(`test_hybrid_10k_batch_propagates_on_gpu_without_fallback` gerçek
   cuda:0'da PASSED — n=10 000, chunk=8192, float64, Sun+Earth TB açık.)*
2. [x] Third-body açıkken fallback yok.
   *(`test_policy_third_body_flags_select_hybrid_backend_no_fallback`,
   `test_policy_explicit_hybrid_request_honored`.)*
3. [x] Provenance alanları eksiksiz. *(`third_body_backend=
   "analytic_vectorized"`, `unsupported_forces` assert'li;
   registry kapsamı `test_registry_entry_matches_r03_scope`.)*
4. [x] Küçük batch'te CPU referans ile pozisyon farkı tolerans içinde.
   *(`test_hybrid_provider_matches_numpy_reference_rk4`.)*

### G4 kapısı — ✅ GEÇİLDİ (2026-07-05)
R03 kabul kriterleri 1–4 testli olarak yeşil
(tests/test_torch_third_body.py: 15 passed, CUDA-gated 10k testi dahil,
gerçek GPU'da 33 s, CUDA-enabled ML/HPC ortamı). Sprint 5'e
(frozen search workflow) geçilebilir.

---

## Sprint 5 — Frozen orbit search workflow

### R05 — Frozen / quasi-frozen sınıflandırması (önce tanım, sonra arama)

Sınıflar ve metrikler koda `frozen_score` + `classify_candidate` olarak girer:

- **A) strict frozen:** e-vektörü bounded; e ve h_peri'de anlamlı seküler
  drift yok; omega libration/near-zero drift; impact/escape yok; long-horizon
  validation geçmiş.
- **B) quasi-frozen:** e, h_peri görev süresince bounded; küçük seküler drift
  olabilir.
- **C) long-lived but not frozen:** impact yok ama e/h_peri/omega belirgin
  seküler drift.
- **D) unstable/invalid:** impact, escape, domain exit, e-growth limit aşımı,
  perilune güvenli eşiğin altı.

Metrikler: e_min/max/range, fitted de/dt, h_peri_min/max/range, fitted
dh_peri/dt, inclination drift, omega drift, omega circulation/libration
sınıflandırması, e-vektörü `h=e·sin(ω), k=e·cos(ω)` bounded-loop metriği,
impact time, domain exit time, escape flag.

- [x] Modül: `src/lunaris/analysis/frozen/metrics.py` + `classify.py`
      (yer önerisi; architecture-guardian ile teyit edilir). *(Test:
      `tests/test_frozen_classification.py`.)*
- [x] Eşikler config'te (görev süresine bağlı), hardcoded değil.
- [x] Dil kuralı: kod/rapor çıktılarında "candidate frozen orbit" /
      "quasi-frozen candidate"; "frozen orbit family discovered" ancak
      classical SH long-horizon validation etiketi varsa. *(Negatif test:
      `test_strict_like_candidate_is_not_validated_without_classical_sh`.)*

### R04 — surrogate_assisted_frozen_search pipeline

Adımlar: Sobol element sample → Cartesian dönüşüm → domain/impact precheck →
ST-LRPS kısa/orta propagation → frozen score → aday seçimi → classical SH
validation → local refinement → family/basin raporu.

Staged search (config-driven, sabit değil):

| Stage | Aday | Backend | Süre | Çıktı |
|-------|------|---------|------|-------|
| 1 | 100k–1M Sobol | ST-LRPS gravity-only veya +Earth TB | 7–30 gün | summary-only (R23) |
| 2 | Top 1000 | ST-LRPS+TB veya SH100/SH200 | 90 gün | summary + seçme trajeler |
| 3 | Top 100 | classical SH500/JGGRX_1800 | 180 gün | full validation |
| 4 | Top 20 | local refinement (R22) + classical SH500/SH1000 | final | family raporu |

- [x] CLI entry point (`lunaris-frozen-search` = `lunaris.cli.frozen_search:main`;
      pyproject + PUBLIC_API.md entry-point inventory güncel). *(2026-07-05)*
- [x] Her stage'in girdi/çıktısı dosya kontratı olarak tanımlı (resume
      edilebilir pipeline). *(`analysis/frozen/search.py`:
      `FrozenSearchPipeline` — stage0_samples.npz / stage1_screening.npz /
      stage2_candidates.json / stage3_validation.json / stage4_families.json +
      manifest.json; resume testi
      `test_pipeline_resume_reuses_stage_files` + gerçek koşuda stage 0-3
      diskten yüklendi.)*
- [x] Sobol seed + sample count provenance'ta. *(manifest
      `sampling_provenance` — seed/count/method/bounds/base-2 notu; test:
      `test_sobol_samples_are_deterministic_with_provenance`.)*
- Backend'ler protokolle enjekte edilir: `TorchSHScreeningPropagator`
  (R07 loop'u üzerinde torch classical SH, CPU/CUDA, dürüst adlandırma) ve
  `ClassicalSHValidationPropagator` (CPU referans DOP853) —
  `analysis/frozen/search_backends.py`. ST-LRPS screening artifact'i aynı
  protokolü uygular (gelecek bağlama işi, pipeline değişikliği gerektirmez).
- **Bilinen kısıt:** stage-4 refinement objective'i screening propagator'ı
  N=1 ile çağırır; GPU'da kernel-launch overhead'i domine eder (yavaş).
  Refinement için CPU adapter veya kısa-horizon objective önerilir.

### R27 — Domain guard frozen search'te zorunlu — DONE (2026-07-05)

- [x] Propagation sırasında: altitude envelope, radius, NaN/Inf, impact,
      escape, domain-exit-time kontrolleri. *(`analysis/frozen/domain_guard.py`
      — `evaluate_domain_guard` (T,N,6) blok üzerinde; post-impact satırlar
      hariç; ilk ihlal zamanı + neden kodu.)*
- [x] Domain exit ⇒ candidate invalid veya low-confidence;
      `score += domain_exit_penalty`; metadata `domain_exit=true`.
      *(`FrozenSearchDomainGuard.policy` = invalid | low_confidence;
      `apply_domain_guard_to_scores`; aday kaydında `domain_guard` bloğu.
      Ek: perilün emniyet tabanı stage-2'de de skor filtresi.)*
- [x] Kural: ST-LRPS domain dışına çıkan trajectory final candidate OLAMAZ.
      *(`assert_candidate_domain_clean` — stage-3 girişinde ve stage-4
      refinement öncesinde RuntimeError; testler:
      `test_domain_exited_candidate_cannot_be_final`,
      `test_pipeline_domain_guard_excludes_out_of_envelope_samples`.)*

### R21 — Final validation kuralı (enforcement) — DONE (2026-07-05)

- [x] Pipeline, `status ∈ {strict_frozen, quasi_frozen}` etiketini yalnız
      `validation_backend=classical_SH*` sonucu varsa yazabilir — kod
      seviyesinde zorlanır, konvansiyon değil. *(Üç katman:
      (1) `classify_candidate` classical olmayan backend'e asla validated
      status vermez; (2) `search.enforce_classical_validation_rule` her kayıt
      diske yazılmadan önce yeniden kontrol eder (RuntimeError);
      (3) `family_report.validate_family_report` şema seviyesinde reddeder.
      Negatif testler:
      `test_pipeline_negative_no_frozen_status_without_classical_backend`,
      `test_enforce_classical_validation_rule_blocks_non_classical`,
      `test_family_report_enforces_r21_language_rule`.)*
- [x] Model-degree sensitivity kontrolü validation raporuna girer.
      *(`sensitivity_validations` — ikinci classical SH degree ile aynı aday
      yeniden doğrulanır, `status_agrees` alanı stage3'e yazılır; G5
      koşusunda deg30-primary + deg50-sensitivity 10/10 uyum. Earth/Sun
      third-body duyarlılığı, ephemeris-bağlı validation config'i
      eklendiğinde aynı mekanizmadan geçer — adapter fail-closed
      NotImplementedError bırakır, sessiz gravity-only iddiası yapılmaz.)*

### R22 — Local refinement modülü — DONE (2026-07-05)

- [x] Girdi: top Sobol adayları. Karar değişkenleri: a, e, i, RAAN, ω,
      anomali. Amaç: `minimize frozen_score`. *(`analysis/frozen/refine.py`
      — `refine_candidate(elements0, score_fn, config)`; pipeline stage-4
      objective'i propagate→metrics→frozen_score + impact/domain-guard inf.)*
- [x] Kısıtlar: bounds + e_max penalty-inf; impact/escape/domain-envelope
      kontrolleri pipeline objective'inde (R27 guard aynen). Perilün emniyeti
      frozen_score'un invalid kuralından gelir.
- [x] Optimizerlar: Nelder-Mead + differential_evolution (scipy).
      *(CMA-ES/least_squares/Bayesian bilinçli eklenmedi — plan "artan
      sırayla eklenir" der; ihtiyaç doğduğunda aynı arayüze eklenir.)*
- [x] Çıktı: refined_elements, original_score, refined_score, improved,
      `validation_status="requires_classical_sh_validation"` (R21 dili),
      optimizer_metadata (n_evaluations/converged/message/seed).
      *(Testler: `test_refine_*` 4 test + pipeline entegrasyonu
      `test_pipeline_refinement_stage_runs_with_analytic_objective`;
      kötüleşme asla döndürülmez.)*

### R30 — Candidate family JSON raporu — DONE (2026-07-05)

- [x] Review'daki JSON şeması (family_id, status, screening_backend,
      validation_backend, gravity_model, third_body, validation_days,
      element_ranges, stability_metrics, provenance) versiyonlu şema olarak
      eklendi. *(`analysis/frozen/family_report.py` —
      `FAMILY_REPORT_SCHEMA_VERSION=1`, `validate_family_report` saf-Python
      şema doğrulayıcı (R21 yapısal kuralı dahil), `build_family_report`
      üretimden önce doğrular; aile durumu üyelerin EN ZAYIF statüsüdür.
      Testler: roundtrip + 3 parametrize red + R21 red + weakest-member.)*

### R31 — Frozen orbit plot/report generator — DONE (2026-07-05)

- [x] Plotlar: e(t), h_peri(t), ω(t), h–k faz portresi, a–e score map,
      i–ω stability map, perilune safety map, score histogramı,
      screening-vs-validation score. *(`analysis/frozen/plots.py` —
      `generate_frozen_report_figures(run_dir)` stage dosya kontratlarından
      okur; eksik stage'in figürü atlanır, yanıltıcı placeholder üretilmez.
      G5 koşusunda 9 figür üretildi.)*
- [x] scientific-figures disiplinine uygun: her figür başlıkta analitik
      soruyu, eksenlerde birimleri, footer'da veri kaynağı/frame/ölçeği
      beyan eder.

### G5 kapısı — ✅ GEÇİLDİ (2026-07-05)
1. [x] Küçük ölçekli uçtan uca demo tek komutla koştu:
   `lunaris-frozen-search --out outputs/frozen_search/g5_demo_10k
   --n-samples 10000 --seed 42 --screening-days 1 --screening-degree 8
   --top-k 10 --validation-days 2 --validation-degree 30
   --sensitivity-degree 50` — 10 000 Sobol (torch_cuda_sh screening, GTX
   1660 Ti, ~2 dk) → top 10 → classical_sh_deg30 validation (+deg50
   sensitivity, 10/10 status uyumu) → 9 aile + 9 figür. Sonuç dürüst:
   1 aday quasi_frozen (validated), 9 aday long_lived_not_frozen.
2. [x] Hiçbir aday classical-SH etiketi olmadan frozen/quasi-frozen statüsü
   alamıyor (negatif test:
   `test_pipeline_negative_no_frozen_status_without_classical_backend` +
   classifier/choke-point/şema katmanları).
3. [x] Family JSON şema testi yeşil (tests/test_frozen_search_pipeline.py —
   24 test; suite bölümü aşağıdaki final koşuda).

---

## Sprint 6 — Paper-safe benchmark ve parity

### S6a ön işleri (performans hazırlığı)

**R13 — SH-only Numba fast-path (`_rhs_sh_only_numba`)**
- [x] İçerik: state unpack, (gerekiyorsa) ephemeris quaternion frame
      rotation, lunar SH acceleration, inertial dönüş, dydt. İÇERMEZ:
      third-body, SRP, albedo, thermal, tides, relativity, kullanılmayan
      flag'ler. *(2026-07-05: `DynamicsEngine.build_rhs()` SH-only durumda
      `rhs_path=sh_only_numba` seçiyor; surrogate path ayrı kalıyor.)*
- [x] Amaç: GMAT/STK karşılaştırması için temiz SH-only baseline; SH500/1000
      benchmark'larında düşük overhead. *(Kod yolu hazır; gerçek benchmark
      matrisi R16'da.)*
- [x] Doğruluk testi: genel RHS ile bit-yakını eşleşme (aynı force set).
      *(Test: `tests/test_dynamics.py::test_sh_only_fast_path_matches_general_rhs_when_extra_force_is_zero`;
      dynamics suite: 40 passed, 1 skipped.)*

**R14 — RHS'de frame rotation tekilleştirme**
- [x] RHS başında bir kez: `r_fixed`, `sun_fixed`, `earth_fixed`; SH/tides/
      albedo/thermal bunları paylaşır. *(2026-07-05: classical Numba ve
      surrogate-Python RHS içinde fixed-frame vektörleri ephemeris fetch sonrası
      tek noktada hazırlanıyor.)*
- [x] Kabul: aynı RHS evaluation'da aynı vektör aynı frame'e birden fazla
      döndürülmez (kod incelemesi + mikro-benchmark). *(2026-07-05:
      `tools/rhs_frame_rotation_microbench.py` →
      ignored runtime output `outputs/optimization/rhs_frame_rotation_microbench.json`;
      generated outputs stay under `outputs/` per validation policy. Local SH50,
      sabit-attitude ephem rotasyonu, n=8000: sh_only fast path 17.57 µs/eval
      vs genel RHS (sıfır-J2 extra force) 17.99 µs/eval; rotasyon-katı bir
      fark yok, genel yol farkı ~%2 bookkeeping. Not: acceleration-breakdown
      diagnostik yolu (engine.py ~1811+) force başına yeniden döndürür —
      bilinçli; hot RHS değildir, sonuç etkilemez.)*

**R15 — Fixed-step out-buffer**
- [x] solve_ivp reference path olarak kalır (allocation overhead kabul);
      fixed-step batch path out-buffer kullanır. *(2026-07-05:
      `run_batched_fixed_step()` opsiyonel `output_buffer` kabul ediyor,
      shape/dtype/contiguous guard ve `output_buffer_reused` metriği var.
      Torch-enabled ortamda tests/test_batched_fixed_step.py 35/35 yeşil
      (skip yok); PyTorch'suz/eksik venv'lerde bu dosyanın skip etmesi
      beklenen opsiyonel-bağımlılık davranışıdır. solve_ivp-vs-fixed-step
      farkının rapor tablosunda gösterimi R16 matris koşusuyla gelir.)*

### R17 — JIT warm-up protokolü

- [~] Protokol: load → build RHS → dummy call/propagation → timer →
      benchmark. Rapor: `cold_time, warm_time, jit_compile_time,
      propagation_time`. *(2026-07-05: output contract `cold_time_s`,
      `warm_time_s`, `jit_compile_time_s`, `propagation_time_s` olarak
      standartlaştırıldı; gerçek RHS dummy warm-up ölçümü R16 koşusunda
      doğrulanacak.)*
- [x] Kural: makale tablolarında warm time esas; cold time ayrıca verilir.
      Benchmark runner bu ayrımı otomatik üretir. *(validator cold/warm/JIT
      kolonlarını zorunlu kılıyor ve cold >= warm + JIT tutarlılığını kontrol
      ediyor; test: `test_missing_runtime_timing_columns_fail`.)*

### R16 — Paper-safe benchmark matrisi

- [ ] Gravity degree: SH25 / SH50 / SH100 / SH200 / SH500.
- [ ] Backend: classical CPU SH, torch_cuda_sh (varsa), ST-LRPS GPU,
      ST-LRPS+third-body (R03 sonrası).
- [ ] Süre: 1g / 5g / 30g / (frozen adayları için) 90–180g.
- [ ] Batch: 1 / 100 / 1000 / 10000 / 100000 (screening mode).
- [~] Metrikler: wall time, cold/warm time, JIT compile time, RHS eval
      sayısı, accel eval/s, propagated-seconds-per-wall-second, final pos
      error, RMS pos error, RIC (radial/along/cross), phase lag,
      phase-corrected RMS, energy drift, impact/domain failure, memory,
      chunk size. *(2026-07-05: CSV/JSON output contract ve validator bu
      kolonları bekliyor; gerçek paper-safe matris henüz koşulmadı.)*
- [~] reproducible-benchmarks disipliniyle koşulur (provenance + manifest +
      validation_report). *(manifest/validation contract hazır; pahalı gerçek
      benchmark bu oturumda koşulmadı.)*

### R19 — Genişletilmiş validation metrikleri

- [~] Anlık acceleration RMSE tek başına YETMEZ; ekle: accel max error,
      potential error, trajectory RMS, final pos error, RIC, phase lag,
      along-track drift, time-shift-corrected RMS, energy drift, domain exit
      count. Frozen search metrikleri R05'ten gelir. *(2026-07-05:
      `scenario_results.csv` için genişletilmiş kolonlar, `metrics_summary.json`
      için acceleration/potential/energy_drift unit guard'ı ve synthetic/legacy
      standardizasyonu eklendi. Bilimsel evidence için boş extended kolonlar
      fail ediyor; gerçek non-synthetic metrik hesaplaması R16 koşusunda.)*

### R20 — Phase-drift analizinin ana validation'a entegrasyonu

**Not:** Teşhis altyapısı hazır ([PHASE_DRIFT_PLAN.md](PHASE_DRIFT_PLAN.md)
Faz 1–4 tamam; benchmark CSV phase kolonlarını zaten üretiyor). Buradaki iş:
- [ ] G0 kapısını (gerçek koşu) tamamla — PHASE_DRIFT_PLAN'daki kriterlerle.
- [~] Rapor standardı: raw RMS, phase-corrected RMS, estimated time shift,
      kalan radial/cross error her validation raporunda. *(phase kolonları
      benchmark validator'da varsayılan zorunlu; gerçek G0 koşusu bekliyor.)*
- [ ] UQ along-track şişmesi ↔ phase drift ilişkisi analizi (PHASE_DRIFT
      G0b'deki UQ hizalanması ile birleşik).

### R18 — GMAT/STK parity protokolü

- [x] Protokol: aynı initial state, central-body sabitleri, gravity
      degree/order, frame varsayımları, süre, output interval, (mümkün olan
      en yakın) tolerans; force model = lunar SH only; third-body iki
      tarafta da yoksa yok; Lunaris için JIT warm-up hariç. *(2026-07-05:
      `docs/GRAVITY_ENGINE_EXTERNAL_VALIDATION.md` içine GMAT/STK parity
      protocol eklendi.)*
- [ ] Rapor: wall time, function evaluations, final state farkı, trajectory
      RMS farkı, enerji-benzeri diagnostic, output interpolation error.
- [x] **Claim kuralı:** "Comparable to GMAT" ancak bu benchmark SONRASI;
      öncesinde yalnız "optimized compiled CPU SH reference path".

### G6 kapısı
1. Benchmark matrisi paper-safe koşuldu; manifest + validation_report tam.
2. Warm/cold ayrımı tablolarda; float32/float64 ayrı raporlandı (R10).
3. Gravity-only vs full-dynamics tabloları ayrı (R02).
4. Parity raporu üretildi VEYA parity yapılamadıysa claim dili
   "optimized compiled CPU SH reference path" olarak sabitlendi.

**Durum (2026-07-05, güncel):** G6 henüz geçilmedi. S6a artık kapalı
(R13/R14/R15 [x]; R14 mikro-benchmark script'i tracked, ölçüm çıktısı
`outputs/optimization/rhs_frame_rotation_microbench.json` altında ignored
runtime artefaktı olarak yeniden üretilebilir). PyTorch CUDA'nın CUDA-enabled
ML/HPC ortamında çalıştığı kaydedildi (G4 bununla kapandı); aktif shell/venv
PyTorch'u görmüyorsa R16 CUDA kolonları aynı ML/HPC ortamında koşulmalıdır.
Kalanlar: (a) R16 paper-safe benchmark matrisi gerçek koşusu (R17 warm-up
ölçümü ve R19 gerçek metrikler aynı koşuda doğrulanır), (b) R20 PHASE_DRIFT G0
gerçek koşusu, (c) R18 GMAT/STK artefaktı — GMAT erişimi yoksa "yapılamadı" +
claim dili sabit (G6 bunu kabul eder). Bunlar reproducible-benchmarks
disipliniyle, ayrı ve gözetimli bir benchmark oturumunda koşulmalı; bu plan
güncellemesiyle otomatik tetiklenmedi.

---

## Paralel iz — training/kod kalitesi (sprint'lere bağlı değil)

### R24 — run_epoch decomposition — DONE (2026-07-05)
- [x] `prepare_batch / compute_loss / backward_step / validate_numerics /
      update_metrics / log_progress / write_history` ayrışması. *(engine.py
      commit'lendikten sonra yapıldı: `_EpochState` dataclass (tüm epoch
      akümülatörleri açık state olarak) + `_prepare_batch`, `_compute_loss`,
      `_validate_numerics`, `_nan_sentinel`, `_write_failure_manifest`,
      `_collocation_laplacian_step`, `_backward_step` (AMP ve non-AMP tek
      clip/kayıt yardımcılarını paylaşır — aynı logging kontratı),
      `_update_metrics`, `_log_progress`, `_write_history_summary`;
      run_epoch ~410 satırdan ~140 satırlık orkestratöre indi. İki NaN
      sentinel'i tek parametrik yardımcıya dedupe edildi. Training smoke:
      run_epoch'u gerçek train/val fazlarıyla çalıştıran testler dahil
      703 passed / 4 skipped seçki yeşil; ruff temiz.)*
- [x] AMP branch'te literal-string bug kontrolü (f-string olmalı); AMP ve
      non-AMP aynı logging contract. *(2026-07-05: run_epoch AMP dalındaki
      grad_norm>50 uyarısı `"batch={n_batches}"` literal string idi — AMP
      yolunda `batch={n_batches}` düz metin basıyordu; `f`-prefix eklendi,
      non-AMP dalıyla eşitlendi. Ayrıca log_progress'te atılan iki ölü ifade
      (kullanılmayan w_a_eff float'ı + hiçbir yere gitmeyen dir/extra_terms
      f-string'leri) temizlendi. 114 training testi yeşil.)*
- [x] Mevcut train() refactor desenini izle (build_training_session +
      _run_training_loop ayrışması referans). *(Aynı desen: küçük, isimli,
      state-açık yardımcılar; davranış/return-şeması bire bir korunur.)*

### R25 — Laplacian scaling tutarlılığı — DONE (2026-07-05)
- [x] In-batch Laplacian ile collocation Laplacian aynı coordinate/potential
      scaling'de mi — test et; değilse düzelt. *(DEĞİLDİ: in-batch
      `_laplacian_penalty` chain-rule ile fiziksele çeviriyordu
      (`×u_scale/x_scale²`), collocation ise ham scaled Laplacian²'yi
      döndürüyordu — `(u_scale/x_scale²)²` faktörü kadar farklı birim.
      `collocation_laplacian_loss` fiziksele çevrilerek eşitlendi;
      `_extract_xu_scale` helper'ı hem ScalerPack hem SobolevLoss'u destekler.)*
- [x] Metrik adları açık / metadata'ya units yazılır. *(iki estimator da artık
      mean((∇²U_phys)²) [s⁻⁴]; docstring'ler + stats dict'te inline birim notu.
      Not: mevcut `loss_laplacian*` anahtarları yeniden adlandırılmadı —
      history/CSV tüketicilerini ve testleri kırmamak için birim yorumla
      belgelendi.)*
- [x] Kabul: Laplacian penalty'nin fiziksel anlamı ve birimi belirsiz kalmaz.
      *(regresyon testi test_collocation_laplacian_physical_units: U=Σx²
      için scaled ∇²=6, fiziksel loss=(6·u_scale/x_scale²)²; 13+114 test yeşil,
      ruff temiz.)*

### R28 — print → logger
- [~] Library kodunda `logging.getLogger(__name__)`; verbose flag/log level;
      benchmark stdout temiz; CLI handler ekler. *(2026-07-05: batch backend
      policy/engine ve DynamicsEngine library hazır mesajları logger'a taşındı;
      `rg "print\\(" src/lunaris/core/dynamics/engine.py src/lunaris/core/batched_fixed_step.py src/lunaris/batch/backend_policy.py src/lunaris/batch/engine.py`
      temiz. İkinci dalga (aynı gün): `core/propagation/propagator.py`
      [STEP]/[PROP]/[GRAV] verbose diagnostikleri, `core/torch_sh_propagator.py`
      ve `core/torch_batch_propagator.py` [BATCH] mesajları logger.info'ya
      taşındı (telemetry JSON satırı bilinçli stdout — makine-okur akış).
      Benchmark CLI rapor çıktıları kasıtlı stdout; kalan UI/CLI/rapor
      print'leri rapor-vs-diagnostik tasarımıyla ayrı ele alınır.)*

---

## 5. Makale claim sözlüğü (bağlayıcı)

**Kullanılabilir:**
- "ST-LRPS is a conservative residual-potential surrogate intended for
  high-throughput lunar gravity evaluation in large ensemble propagation."
- "Classical SH propagation remains the reference method for individual
  high-fidelity trajectories."
- "ST-LRPS is used as a screening accelerator; final candidate orbits are
  validated with classical high-degree spherical harmonics."
- "The method is demonstrated on surrogate-assisted global search for lunar
  quasi-frozen orbit candidates."

**Yasak (ön koşulu sağlanmadan):**
- "ST-LRPS replaces GMAT/STK/NASA propagators." — her zaman yasak.
- "ST-LRPS is faster for single-trajectory propagation." — her zaman yasak.
- "ST-LRPS is a full-fidelity GPU dynamics engine." — her zaman yasak.
- "We discovered new frozen orbit families." — yalnız uzun-süreli classical
  SH validation + literatür karşılaştırması SONRASI ve "candidate/
  quasi-frozen" dilinden terfi gerekçesi raporlanarak.
- "Comparable to GMAT." — yalnız R18 parity raporu SONRASI.

---

## 6. Riskler ve açık kararlar

1. **force_direct kaldırma dalgası geniş** (~20 dosya): tek büyük PR yerine
   (a) capability/config kapatma, (b) runtime reddi, (c) kod temizliği
   şeklinde 2–3 PR; her adımda suite yeşil.
2. **Refactor tuzakları (yaşanmış):** compat shim'ler dinamik fold olmalı
   (ruff static re-export'ları siliyor, CI'ı 3 kez kırdı); modül taşımada
   `@njit` dekoratörleri düşebilir; ARCHITECTURE.md import-linter kontrat
   adlarını birebir içermeli.
3. **R07 ortak loop birleşmesi** Torch SH davranışını değiştirebilir —
   birleşme öncesi golden-output testleri sabitle.
4. **clarity-remediation Faz 2 iptali** resmî karar ister: bu plan merge
   edildiğinde eski planın force_direct-first-class maddesine süpersede notu
   düşülür.
5. **GMAT/STK erişimi** yoksa R18 "yapılamadı" olarak kapanır ve claim dili
   buna göre sabitlenir (G6 kapısı bunu açıkça kabul eder).
6. **100k smoke testleri CI'da koşamaz** — işaretli (slow/gpu) test katmanı
   ve elle koşulan checklist gerekir; sonuç artefaktları repoya kanıt olarak
   bağlanır.
7. **Frozen tanım eşikleri** literatüre dayanmalı (Ely, Folta, Lara vb.
   lunar frozen orbit çalışmaları); eşik seçimi raporda gerekçelendirilir —
   aksi halde sınıflandırma keyfî görünür.

---

## 7. Sıralı özet (tek bakış)

```
S1  Scope cleanup      : R01 R02 R29a            → G1
S2  Backend correctness: R09 R10 R12 R26 R29b    → G2
S3  Batch infra        : R07 R08 R06 R23 R11     → G3
S4  Hybrid backend     : R03                     → G4
S5  Frozen search      : R05 R04 R27 R21 R22 R30 R31 → G5
S6  Paper-safe bench   : R13 R14 R15 R17 R16 R19 R20 R18 → G6
∥   Paralel            : R24 R25 R28
```

Nihai konumlandırma: Lunaris classical SH = truth/reference/single
trajectory/final validation; ST-LRPS = high-throughput screening (10k–100k
orbit, Sobol/MC/UQ) lunar gravity residual surrogate. Frozen orbit
uygulaması: ST-LRPS geniş uzayı tarar → quasi-frozen adaylar skorla seçilir
→ classical SH500/SH1000/JGGRX_1800 doğrular → sonuç "surrogate-assisted
global search for lunar quasi-frozen orbit candidates" olarak sunulur.
