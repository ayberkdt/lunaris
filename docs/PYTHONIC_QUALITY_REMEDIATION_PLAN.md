# Pythonic Kod Kalitesi İyileştirme Planı

**Durum:** Maddeler 1-2, 4-9 uygulandı 2026-07-08 (suite 2722 yeşil). Kalan:
madde 3 (SciPy `EventSpec` dataclass) — istenmedi; madde 6 numba-kernel bölme
kasıtlı yapılmadı (parite); madde 9 modül-bazlı coverage eşiği opsiyonel.
**Kaynak:** dış Pythonic/okunabilirlik kod incelemesi, alındığı tarih 2026-07-08
**Branch bağlamı:** `refactor/frame-handling-and-physics`
**Kapsam koruması:** genel davranışı ve sayısal sonuçları koru; yeni fizik ekleme;
numba hot-path'lerinde performansı bozacak "temizlik" yapma; her adım sonrası
`ruff` + `mypy` temiz ve tüm test paketi yeşil kalmalı. **Commit'lerde Claude
co-author trailer'ı YOK** (author = yalnızca kullanıcı).

Bu plan çalışan ve mimarisi sağlam bir kod tabanının **okunabilirlik ve
Pythonic-lik** seamlarını sıkılaştırır. Yeniden tasarım değildir; her madde
mevcut ağaca karşı doğrulanmıştır ve "Mevcut durum" satırları bugün gerçekte ne
olduğunu belirtir ki yapılmış işi tekrar etmeyelim.

---

## Depo doğrulama anlık görüntüsü (2026-07-08)

| # | İnceleme maddesi | Doğrulanan mevcut durum | Uygulandı? |
|---|------------------|--------------------------|:----------:|
| 1 | Ruff global ignore'ları fazla geniş | `E402/E701/E702` globalden çıkarıldı; numba kernel (`batch_propagator`, `math_utils`) + bootstrap dosyaları için per-file-ignore; 190 E701/E702 + 2 E402 düzeltildi | ✅ |
| 2 | Geniş `except Exception` + sessiz fallback | Core katmanı daraltıldı: `config.py` (7 optional-import → `ImportError`), `events.py` (11 → `(TypeError, ValueError)`), `checkpoint.py` (9 → typed), `propagator.py` (7 → typed) | ✅ (core) |
| 3 | SciPy event'lerine dinamik attribute yapıştırma | `_wrap_event_first6()`, `build_events()` içinde `terminal`/`direction`/`_event_role` fonksiyon nesnesine set ediliyor | ❌ (Faz 6) |
| 4 | Star import + private helper export'ları | 8 studio dosyasında `import *` kaldırıldı (explicit `__all__` + resolver); `F403/F405` + qt_common `F401` ignore silindi; `core.dynamics.__init__` 4 private sembol export'u kaldırıldı | ✅ |
| 5 | CLI parser/validation çok prosedürel | `parse_args` → 7 `_add_*_args` builder; `validate_args` → 6 odaklı `_validate_*`; davranış birebir aynı (aynı hata mesajları/exit-code) | ✅ |
| 6 | `DynamicsEngine.build_rhs()` God-function | `_prepare_force_packs()` + `_PreparedForces` NamedTuple çıkarıldı (prepare/orchestration ayrımı, section marker'lar); **numba kernel closure gövdeleri kasıtlı bölünmedi** (parite koruması, bkz. B2 deferral) | ◐ (kısmi) |
| 7 | Stringly-typed config (`str` yerine `Literal`/`Enum`) | `GravityBackend = Literal["classic_sh","st_lrps"]`; `GravityConfig.backend` tiplendi; CLI sınırı `cast` + `__post_init__` SSOT doğrulaması | ✅ |
| 8 | Library modülünde `__main__` smoke-test bloğu | Blok `tools/check_config.py`'ye taşındı; `core/config.py` artık sessiz kütüphane (print/`__main__` yok) | ✅ |
| 9 | Kalite sinyali tutarsızlıkları | Classifier `4 - Beta` → `3 - Alpha` (README "actively developed research software" ile tutarlı; kullanıcı kararı). Modül-bazlı coverage eşiği yapılmadı (opsiyonel takip) | ✅ (kısmi) |

---

## Öncelik sırası ve commit dilimlemesi

İncelemenin verdiği "net öncelik sırası" bağımlılık-güvenli ve her adım küçük,
gözden geçirilebilir bir commit. Önerilen sıra:

1. **C1 — Faz 1:** Ruff global ignore'larını daralt (Madde 1)
2. **C2 — Faz 2:** `except Exception` daralt + library'de logging/typed exception (Madde 2)
3. **C3 — Faz 3:** Star import'ları kaldır + private helper sızıntısını kes (Madde 4)
4. **C4 — Faz 4:** `build_rhs()` ve `cli/options.py` büyük prosedürel blokları böl (Madde 5 + 6)
5. **C5 — Faz 5:** Config'te `Literal`/`Enum` + `Protocol` (Madde 7)
6. **C6 — Faz 6:** Küçük tutarlılık maddeleri (Madde 3, 8, 9)

Her commit bir sonrakine geçmeden önce paketi yeşil, `ruff`/`mypy` temiz bırakmalı.

---

## Faz 1 — Ruff global ignore'larını daralt (Madde 1)

**Sorun.** `pyproject.toml:173-186`'da `E501` (satır uzunluğu), `E402`
(import-not-at-top), `E701`/`E702` (compound statement / noktalı virgül) tüm repo
için kapatılmış. Bu, en çok okunabilirlik bozan yazımlara genel af çıkarıyor;
kod tabanı zamanla "bilimsel prototip scriptleri" gibi büyür.

**Hedef.**
- `E701`/`E702`'yi globalden çıkar; yalnızca gerçek numba hot-path dosyalarına
  `per-file-ignores` ile ver (ör. `core/dynamics/**/kernels*.py`, propagator
  kernel modülleri). UI/CLI/config/analysis'te açık kalsın.
- `E402`'yi globalden çıkar; yalnızca kanıtlanmış pre-import bootstrap dosyalarına
  (Qt platform setup, run-as-script `sys.path` düzeltmeleri, guarded optional
  import) `per-file-ignores` ver.
- `E501`: kitlesel reflow churn'ünü önlemek için şimdilik korunabilir; ama
  `line-length = 100` zaten var, isteğe bağlı olarak `E501`'i açıp mevcut ihlalleri
  ölçüp ayrı bir dilimde temizlemeyi değerlendir (düşük öncelik).

**Yöntem.** Önce `ruff check --select E402,E701,E702 --statistics` ile ihlal
sayısını ve dosya dağılımını ölç; gerçekten kernel/bootstrap olan dosyaları
`per-file-ignores`'a taşı, geri kalanı düzelt. **Autofix tuzağı:** `ruff --fix`
dinamik-fold compat shim'lerini strip edebilir — shim dosyalarına dokunma
(bkz. memory `[[shim-dynamic-fold-vs-ruff]]`).

---

## Faz 2 — `except Exception` daralt, library'de sessiz fallback'i kaldır (Madde 2)

**Sorun.** `ensure_model_configs()` optional import'ları geniş `except Exception`
ile yutuyor; `events.py` event time/state parse hatalarında `None` dönüyor / `pass`
ediyor; CLI runtime aşamaları `except Exception as e: print("[FATAL] …"); return 1`.
Programlama hatası, eksik bağımlılık ve bozuk veri formatı aynı kefeye konuyor —
bilimsel kodda yanlış fiziksel sonuç sessizce geçebilir.

**Hedef.**
- Beklenen hata sınıflarını ayrı yakala: `ImportError`, `FileNotFoundError`,
  `ValueError`, `TypeError`, `IndexError`, `KeyError`.
- Library/core/physics katmanında `print` yerine `logging` (bkz. memory
  `[[roadmap-parallel-track-r24-r25-r28]]` R28: library print'leri logger'a taşındı —
  aynı deseni izle) veya özel exception tipleri.
- CLI en dış katman beklenen exception'ı kullanıcı dostu mesaja + exit-code'a
  çevirebilir; ama core/physics hatayı saklamamalı, yukarı bırakmalı.
- Gerçekten "optional feature yok" durumu ile "veri bozuk" durumunu ayırt et:
  ilki sessizce degrade olabilir, ikincisi fail-closed olmalı (bkz. memory
  `[[architecture-audit-2026-07]]` fail-closed legacy fallback deseni).

**Yöntem.** `grep -rn "except Exception" src/lunaris` ile envanter çıkar;
her call-site'ı "gerçekten geniş yakalama gerekli mi?" diye sınıflandır. En dıştaki
CLI sınırı hariç hiçbir core/physics call-site'ı çıplak `except Exception` +
`pass`/`None` bırakmamalı.

---

## Faz 3 — Star import'ları kaldır, private helper sızıntısını kes (Madde 4)

**Sorun.**
- ST-LRPS UI'da `from .qt_common import *` / `from .common_widgets import *`;
  `pyproject.toml:192-195` bunu `F403/F405` ignore ile bilinçli kabul etmiş.
- `core.dynamics.__init__` `_sample_albedo_dn_scaled`, `_select_adaptive_sh_degree`,
  `_AlbedoPack`, `_is_surrogate_gravity_provider` gibi underscore sembolleri
  package facade'dan export ediyor. "Public API nedir, internal nedir?" sınırı
  bulanıklaşıyor.

**Hedef.**
- `qt_common` hub'ı için explicit `__all__` tanımla; sonra `import *` yerine
  `from .qt_common import QApplication, QMainWindow, QStackedWidget, pyqtSignal, …`
  şeklinde açık import'a geç. Bu yapılınca `F403/F405` per-file-ignore'u kaldırılabilir.
- `core.dynamics.__init__`'te underscore-prefix helper'ları `__all__` dışına al;
  gerçekten public olması gereken varsa underscore'u kaldırıp dokümante et,
  gerisini internal modülden import ettir. (Bkz. memory `[[architecture-seam-cleanup-plan]]`
  Madde 1'deki propagation facade `__all__` daraltmasıyla aynı desen.)

**Yöntem.** Star import kaldırmadan önce hangi sembollerin gerçekten kullanıldığını
`ruff check --select F405` ile listele; `__all__`'ı ona göre kur. Facade daraltmasında
private-helper testlerini kanonik internal modüle repoint et (facade'a değil).

---

## Faz 4 — Büyük prosedürel blokları böl (Madde 5 + 6)

### 4a — `cli/options.py` parser/validation (Madde 5)

**Sorun.** `parse_args()` çok argüman tanımlıyor; `validate_args()` uzun if-chain +
thermal/albedo/tide doğrulaması aynı fonksiyonda. `TimeConfig`/`PropagatorConfig`/
`GravityConfig` dataclass'ları zaten varken CLI paralel bir doğrulama sistemi kuruyor —
iki sistem zamanla drift eder.

**Hedef.**
- Argument group builder'ları: `add_time_args`, `add_gravity_args`, `add_thermal_args`,
  `add_albedo_args`, `add_tide_args` gibi ayrı fonksiyonlar.
- Doğrulamayı mümkün olduğunca config dataclass'larının `__post_init__` kontrollerine
  devret; CLI yalnızca string/argparse → dataclass dönüşümü yapsın. (Bkz. memory
  `[[review-2-contracts-2026-07]]` / `[[review-remediation-2026-07]]` — CLI surface SSOT
  ve declarative CLI patch işi; onunla tutarlı kal, çift doğrulamayı çoğaltma.)
- **Not:** `cli/options.py` mypy kapsamında (`pyproject.toml:230`) — bölme sonrası
  mypy temiz kalmalı.

### 4b — `DynamicsEngine.build_rhs()` God-function (Madde 6)

**Sorun.** Tek RHS fabrikasında gravity/ephemeris/albedo/Earth-J2/tides/thermal
prepare ediliyor, sonra çok sayıda flag/scalar/array closure'a capture ediliyor,
sonra central/third-body/tides/SRP/albedo aynı büyük RHS içinde dallanıyor. Yeni
force model eklemek veya test etmek devasa fabrikayı değiştirmeyi gerektiriyor.

**Hedef (orchestration ayrı, hot-path ayrı).**
- Prepare mantığını ayır: `prepare_force_packs()` (mevcut `dynamics.preparation`
  extraction'ıyla uyumlu — bkz. memory `[[review-remediation-2026-07]]` Faz 3).
- Kernel kurulumunu force-model bazlı modüler fonksiyonlara böl: `build_sh_rhs`,
  `build_surrogate_rhs`, `build_full_rhs`.
- **Kritik koruma:** numba closure ve hot-path davranışını değiştirme; bu bir
  yeniden-yapılandırma değil, aynı closure'ları daha küçük fonksiyonlara taşımak.
  Sayısal sonuç birebir aynı kalmalı (CPU/GPU parite testleri koru). Surrogate-vs-SH
  RHS path asimetrisi zaten bilinen bir borç — bkz. memory
  `[[symplectic-guard-and-dynamics-asymmetry]]` (B2 unify dup blocks deferred). Bu
  fazı o asimetriyi çözmek için değil, sadece okunabilirliği artırmak için tut.
- Force-model seam'i için mevcut `ForceEvaluator` tasarım notuyla hizala
  (bkz. memory `[[architecture-seam-cleanup-plan]]` Madde 6, `docs/development/FORCE_EVALUATOR_DESIGN.md`).

**Yöntem.** Bu en riskli faz. Küçük dilimlerle git; her dilimden sonra tam
astrodynamics-validation + CPU/GPU parite koş. Şüpheye düşersen `astrodynamics-validation`
skill'ini çalıştır.

---

## Faz 5 — Config'te tip güvenliği (Madde 7)

**Sorun.** `GravityConfig.backend: str` ve runtime `"classic_sh"`/`"st_lrps"`
kontrolü stringly-typed; typo'lar pahalı. Surrogate mypy'da `follow_imports=skip`
(`pyproject.toml:267-269`).

**Hedef.**
```python
GravityBackend = Literal["classic_sh", "st_lrps"]

@dataclass(frozen=True, slots=True, kw_only=True)
class GravityConfig:
    backend: GravityBackend = "classic_sh"
```
- Dar, sabit seçim kümesi olan config alanlarını `Literal`/`Enum`'a geçir.
- `Any` kullanılan yerlerde davranış sözleşmesi netse `Protocol` kullan (ör.
  gravity_provider — bkz. memory `[[roadmap-sprint-progress]]` açık `gravity_provider`
  pre-contract kararı; onunla çakışmayacak şekilde ilerle).
- `common`/`core` zaten mypy-temiz olduğundan (`pyproject.toml:221-223`) bu değişiklik
  mypy'da regresyon üretmemeli; aksine daralttıkça exhaustiveness kazanılır.

---

## Faz 6 — Küçük tutarlılık maddeleri (Madde 3, 8, 9)

### 6a — SciPy event dinamik attribute'ları (Madde 3)
`_wrap_event_first6()` / `build_events()` fonksiyon nesnesine `terminal`/`direction`/
`_event_role` yapıştırıyor. Küçük bir `EventSpec` dataclass (callable + role +
terminal + direction) tanımla; SciPy'ye verirken bir adapter ile `terminal`/`direction`
attribute'larını set et. Bu SciPy API gereği kalıcı ama metadata'yı tip-güvenli
tutar ve refactor/IDE desteğini iyileştirir.

### 6b — Library `__main__` smoke-test bloğu (Madde 8)
`core/config.py` sonundaki `if __name__ == "__main__":` print'li config check'i
`tests/`'e (veya `tools/` / `lunaris-validate` entry point'ine) taşı. Core modül
import edilebilir, test edilebilir, sessiz bir kütüphane olmalı. (Not: coverage
zaten bu bloğu exclude ediyor — `pyproject.toml:283` — ama kaldırmak yine de daha temiz.)

### 6c — Kalite sinyali tutarsızlıkları (Madde 9)
- `pyproject.toml:35` "Development Status :: 4 - Beta" vs README "alpha-stage research
  prototype": ikisini uyumla. Proje sahibi karar vermeli — hangisi doğruysa diğerini ona
  çek. **[KARAR GEREKLİ: Beta mı Alpha mı?]**
- `fail_under = 55` vs yorumdaki %58 baseline: erken aşamada anlaşılır; ama
  orbit-propagation/gravity gibi sayısal-hassas modüller için **modül bazlı daha yüksek
  eşik** hedefle (ör. `core`/`physics` için ayrı bir coverage gate). Bu ayrı bir dilim
  olarak ele alınabilir; ratchet planı zaten yorumda mevcut.

---

## Riskler ve koruma bantları

- **Faz 4b (build_rhs)** açık ara en riskli; sayısal parite kırılırsa geri al. Yeni
  fizik veya davranış YOK.
- **Ruff autofix** dinamik-fold compat shim'lerini strip eder — Faz 1/3'te `--fix`'i
  körlemesine koşma (bkz. `[[shim-dynamic-fold-vs-ruff]]`, CI'yı 3× kırdı).
- **dynamics taşımaları** `@njit` decorator'larını düşürebilir — taşınan kernel'lerde
  decorator'ları doğrula.
- Her faz sonrası: tam CPU test paketi + `ruff check` + `mypy` temiz olmalı.

---

## İlişkili memory / plan dosyaları

- `[[architecture-seam-cleanup-plan]]` — facade `__all__` daraltma deseni (Faz 3'e model)
- `[[review-remediation-2026-07]]` — CLI surface SSOT + dynamics.preparation extraction (Faz 4)
- `[[symplectic-guard-and-dynamics-asymmetry]]` — RHS path asimetrisi borcu (Faz 4b sınırı)
- `[[shim-dynamic-fold-vs-ruff]]` — ruff autofix tuzağı (Faz 1/3)
- `[[architecture-audit-2026-07]]` — fail-closed fallback deseni (Faz 2)
- `[[roadmap-parallel-track-r24-r25-r28]]` — library print→logger deseni (Faz 2)
