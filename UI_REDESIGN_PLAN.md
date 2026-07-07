# Lunaris Mission Studio — UI Yenileme Planı (UI Redesign Plan)

> Durum: ONAYLANDI, uygulama bekliyor. Tarih: 2026-07-07.
> Bu doküman kendi kendine yeterlidir: herhangi bir oturum/model bu dosyayı okuyarak
> fazları bağımsız uygulayabilir. Her maddede sorun → ilke → somut değişiklik →
> doğrulama vardır. Fazlar bağımsız PR'lar halinde uygulanabilir.

## Bağlam ve zorunlu kurallar

- Tasarım sistemi SSOT: `src/lunaris/ui_foundation/` (`tokens.py` → `DESIGN_TOKENS`,
  `palette.py` → `THEME`/`ORBIT_THEME`, `stylesheet.py` → `build_app_stylesheet`).
  Renk/boyut/spacing değişiklikleri **yalnızca** burada yapılır; sayfa-lokal hex veya
  sabit piksel eklemek YASAK. Bkz. `docs/UI_THEME.md`.
- Ana pencere: `src/lunaris/ui/app.py` (`MainWindow._build_ui`, satır ~346).
- Sayfalar: `src/lunaris/ui/pages/*.py`. Paylaşılan widget'lar: `src/lunaris/ui/core/ui_commons.py`.
- Skill'ler: görsel/UX kararları için `lunaris-ux-design`, widget implementasyonu için
  `lunaris-pyside6-ui`, kontrast denetimi için `accessibility-audit`.
- Tasarım dili: Lunar Graphite. Glassmorphism, glow, gradient, yeni accent rengi YASAK.
  "Premium" his; elevation merdiveni + disiplinli accent + tipografik hiyerarşi +
  boyut tutarlılığından gelir. Hiyerarşi grayscale'de de okunmalıdır.
- Doğrulama araçları:
  - Ekran görüntüsü: `python tools/ui/capture_main_window.py` (çıktı `outputs/ui/`;
    `--page`, `--state`, `--dialog` parametreleri var).
  - Kontrast: `.claude/skills/accessibility-audit/scripts/contrast_check.py`
  - Ham hex taraması: `.claude/skills/lunaris-pyside6-ui/scripts/scan_hardcoded_colors.py`
- Commit mesajlarına Claude co-author satırı EKLENMEZ.

## Uygulama sırası

| Faz | Kapsam | Maddeler |
|-----|--------|----------|
| P0 | Mekanik, düşük risk, anında etki | M4 (wheel), M5 (spin okları), M3 (boyut tokenları), M7 (metin) |
| P1 | Chrome (header) | M1 |
| P2 | Sayfa düzenleri + tema inceltme | M2 (Orbit), M6 (Telemetry), M0 (tema) |
| P3 | CLI→UI parite | M8 tablosu, satır satır bağımsız |

Her faz kapanışında: önce/sonra capture + contrast_check + scan_hardcoded_colors +
`pytest tests/ -k ui` (varsa ilgili UI testleri) yeşil.

---

## M0 — Renk teması ve "premium" his (P2) — ✅ DONE (2026-07-07)

> Uygulandı: `size_subsection_pt: 14.0` token eklendi; `Section` kart başlığı
> `#panelTitle` (14pt) oldu, iç grup etiketleri `#sectionTitle` (12pt) kaldı;
> 4 telemetry plot başlığı accent renklerden `fg_main`'e çevrildi. KPI/metrik
> değerlerinin `family_mono` kullanımı zaten mevcuttu. Offscreen capture + 723
> UI testi yeşil ile doğrulandı.


**Sorun:** Yüzeyler aynı düzlükte; accent mavi her yerde (badge, link, plot başlığı)
kullanılıp anlamını yitirmiş; başlık hiyerarşisi tek kademede sıkışık.

**İlke:** Value-tabanlı elevation; accent yalnızca birincil eylem/seçim/odak;
tipografik kademeler ~1.2 oranlı modüler ölçek.

**Değişiklikler:**
1. `tokens.py` `TypographyTokens`'a `size_subsection_pt: float = 14.0` ekle;
   `stylesheet.py`'de panel/alt-panel başlıkları (ör. Orbit Preview başlığı) bu kademeyi
   kullansın. Sayfa başlığı 20 / alt-panel 14 / section 12 / body 10 / caption 9.
2. `stylesheet.py`: `bg_card` üstüne oturan kartlara 1px `border_soft` çerçeve;
   `bg_card_alt` yalnızca gerçekten yükseltilmiş öğelerde (aktif segment, KPI chip,
   popover). Kart-içi-kart aynı fill kullanmasın.
3. Plot/panel başlıklarındaki accent mavi metinleri `fg_main`'e çevir
   (ör. Live Telemetry'deki "Orbital Altitude" pyqtgraph başlığı — 
   `live_telemetry_page.py` içinde plot title rengi).
4. KPI/metrik değer etiketlerinde (`telemetryKpiStrip`, `orbitMetric*`) `family_mono`
   kullan — sayısal değerler zıplamaz (tabular hizalama etkisi).
5. Opsiyonel (ayrı commit): `family_ui`'yi Inter'e taşı. Font loader platform
   fallback'i zaten yönetiyor; Inter repoya eklenecekse lisans dosyasıyla birlikte.

**Doğrulama:** Desatüre ekran görüntüsünde kademeler ayırt edilebilir olmalı;
contrast_check yeni metin/zemin çiftlerinde ≥4.5:1 (metin) / ≥3:1 (non-text).

---

## M1 — Header birleştirme, subheader'ın kaldırılması (P1) — ✅ DONE (2026-07-07)

> Uygulandı: `missionStatusBar` ribbon bloğu ve Preflight/Run rozetleri
> (`lbl_preflight_status`, `lbl_run_status`) tamamen kaldırıldı. Gravity ve Output
> bilgileri header'a iki tıklanabilir `#headerContextChip` olarak taşındı (gravity
> → `_on_gravity_settings`, output → `_browse_out_dir`). `_update_status_bar`
> chip'leri güncelleyecek şekilde sadeleştirildi. Ölü kod temizlendi (`_make_lbl`,
> `_status_divider`, `#statusDivider`/`#missionStatusBar` QSS). `#statusLabel`/
> `#statusValue` QSS korundu (başka sayfalar kullanıyor). NOT: `header_height`
> token'ı hiçbir yerde tüketilmiyor, dokunulmadı — dikey kazanç ribbon
> kaldırmasından geldi. Offscreen capture (Orbit+Telemetry) + 723 test yeşil.


**Sorun:** İki katlı chrome ~100px yiyor. `missionStatusBar` (Mission Readiness
Ribbon, `app.py` `_build_ui` içinde ~satır 457-524) boştayken "Preflight IDLE /
Run IDLE" gibi sıfır bilgi taşıyan rozetler gösteriyor.

**Karar (kullanıcı onaylı):** Ribbon TAMAMEN kaldırılacak; Preflight/Run rozetleri
silinecek.

**Değişiklikler (`app.py`):**
1. `status_bar_frame` (`missionStatusBar`) bloğunu ve `root.addWidget(status_bar_frame)`
   çağrısını kaldır. Bağlı alanlar: `lbl_gravity_status`, `lbl_output_status`,
   `lbl_preflight_status`, `lbl_run_status` — referansları güncelle
   (`app.py` ~satır 2315-2346 civarındaki güncelleme kodu dahil).
2. Ribbon'daki değerli iki bilgiyi header'a taşı: başlık + sayfa badge'inin sağına
   iki sessiz "context chip":
   - **Gravity chip** (ör. "SH · 200"): tıklanınca mevcut Gravity Model dialog'unu açar
     (menubar'daki `a_gravity` eylemiyle aynı slot).
   - **Output chip** (kısaltılmış yol): tıklanınca output klasör seçiciyi açar
     (`_build_menubar`/mevcut "Select Output Directory" slotu, ~satır 2113).
   Chip stili: `bg_card_alt` zemin, `fg_soft` metin, `radius.pill`; QSS'e
   `#headerContextChip` olarak ekle.
3. Preflight geri bildirimi Run akışında kalır: `_start_preflight_validation`
   zaten Run'a basınca çalışıyor; hata/uyarılar console'a ve dialog'a düşüyor.
   Rozet silmek bilgi kaybı yaratmaz.
4. `LayoutTokens.header_height: 64` → `56`.
5. Koşu göstergeleri değişmez: progress bar + run dot + Stop yalnızca koşarken
   görünür (mevcut davranış, satır ~448-453).

**Doğrulama:** Tüm sayfalarda capture; Telemetry sayfasında plot alanının kazandığı
dikey piksel raporlanır. `lbl_preflight_status` vb. referans kalmadığı
`grep -rn "lbl_preflight_status\|lbl_run_status\|missionStatusBar" src/` ile doğrulanır.

---

## M2 — Orbit Setup sayfası yeniden düzeni (P2)

Dosya: `src/lunaris/ui/pages/orbit_config_page.py` (`OrbitPage._build_ui` ~1094,
`_create_params_group` ~1154, `OrbitViz3D._build_controls` ~392, `_metric_chip` ~1077).

**Sorunlar:** (a) Form tek dev Section içinde 11 satırlık düz grid; ghost alanlar
(türetilmiş a/e) kesikli borderlı input gibi görünüp düzenlenebilir sanılıyor.
(b) Önizleme panelinde metrik chip'leri serbest satırda taşıyor/üst üste biniyor
(1280 genişlikte "s 90.0" çakışması gözlendi); layer toggle satırı kesiliyor.
(c) İki panelin üst hizası kaymış.

**Değişiklikler:**
1. **Form gruplama:** Tek Section yerine üç alt grup (mevcut `sectionTitle`
   etiketleri yerine paylaşılan `Subsection` bileşeni): *Orbit size & shape*,
   *Plane & orientation*, *Epoch* (start date buradaysa). Mod segmenti en üstte kalır.
2. **Ghost alanlar:** Aktif olmayan moddaki türetilmiş değerler (`ent_a`/`ent_e`
   Altitude modunda vb.) input görünümünden çıkarılır: `bg_inset` zemin, çerçevesiz,
   `fg_soft` metin, yanında küçük "derived" etiketi (`fg_muted`, caption boyu).
   Odak alamaz (`setFocusPolicy(NoFocus)`), tab sırasından çıkar.
3. **Alan genişliği:** `add_param` içindeki `widget.setMaximumWidth(360)` yerine
   `ControlMetrics`'e `input_width_standard: int = 240` ekle ve kullan; birim etiketi
   bitişik sabit sütunda kalır.
4. **Metrik chip grid'i:** `_create_viz_group` altındaki chip satırı 2×3 sabit
   `QGridLayout`'a alınır (Period/Periselene/Aposelene üstte; Eccentricity/
   Inclination/Energy altta). Chip yüksekliği tek token'dan türetilir.
5. **Layer toggle'ları** (Labels/Nodes/Velocity/Plane) chip grid'inin altında ayrı
   satır; dar genişlikte kırılabilir.
6. Sağ panel açıklaması tek satıra iner (M7 üslup kuralı):
   "Two-body preview. The mission run adds the selected perturbations."
7. Splitter: `setHandleWidth(12)` → `8`.

**Doğrulama:** 1280×800 ve 1920×1080 capture; <1000px compact modda (dikey yığın)
chip/toggle taşması olmamalı. Ghost alanın tab-order dışı kaldığı klavye ile doğrulanır.

---

## M3 — Kontrol boyutları tutarlılığı (P0) — ✅ DONE (2026-07-07)

> Uygulandı: base `QPushButton`'a `min-width:72px` (tiny "Fit"/"Clear" butonları
> düzeldi); ikon-kare butonlar için `#iconButton` QSS kuralı (icon_button_size) +
> force_models'daki iki gear butonu bu objectName'e alındı (setFixedSize kaldırıldı,
> artık 30×30 kare). batch_propagation ve ensemble_analysis_panel'deki keyfi buton
> yükseklikleri (40/42/36/32/28/24) control token'larına bağlandı (primary_height /
> minimum_height / compact_height / status_badge_height). Capture ile doğrulandı.


**Sorun:** `ControlMetrics` (30/34/38) tanımlı ama sayfalar onlarca `setFixedHeight/
Width/Size` ile kendi boyutunu basıyor (ör. `app.py`: progress bar 165×16, StatusBadge
`setFixedWidth(70)`; `orbit_config_page.py`: segmented control `setFixedHeight(40)`).

**Kural:** Birincil eylem `primary_height` (38); form kontrolleri `minimum_height`
(34); toolbar içi `compact_height` (30); badge `status_badge_height` (24). Boyut
kararı token + QSS'te; sayfada `setFixedHeight` yalnız gerçekten sabit öğelerde
(ikon buton, run dot) kalır.

**Değişiklikler:**
1. `stylesheet.py`'de `QPushButton` sınıflarına `min-height` ve `min-width: 96px`
   ekle (objectName bazlı: `primaryBtn`/`dangerBtn` 38, genel 34, toolbar contexti 30).
2. Envanter çıkar: `grep -n "setFixedHeight\|setFixedWidth\|setFixedSize" src/lunaris/ui --include=*.py`
   — her kullanımı ya token'a bağla ya beyaz-listeye al.
3. Beyaz-liste dışı sabit boyut girişini yakalayan küçük bir test ekle
   (`tests/` altında, AST veya grep tabanlı; mevcut entry-point inventory testi
   desenine benzer).
4. StatusBadge'lerdeki `setFixedWidth(70)` kaldırılır; içerik + padding ile doğal
   genişlik, `min-width` QSS'te.

**Doğrulama:** Capture'da yan yana kontrollerin yükseklikleri eşit; test yeşil.

---

## M4 — Wheel ile değer değişiminin TAMAMEN kapatılması (P0) — ✅ DONE (2026-07-07)

> Uygulandı: `ui_commons`'a `NoWheelSpinBox`/`NoWheelDoubleSpinBox`/`NoWheelComboBox`
> (wheelEvent→ignore, StrongFocus). 9 dosyadaki tüm QSpinBox/QDoubleSpinBox/QComboBox
> (30 örnek) bu sınıflara taşındı. Regresyon testi: `tests/test_ui_wheel_safe_controls.py`
> (5 test, yeşil). `grep` ile doğrudan constructor kalmadığı doğrulandı.


**Sorun:** `QSpinBox`/`QDoubleSpinBox`/`QComboBox` Qt varsayılanıyla hover/focus'ta
tekerlek olayını yiyor → sayfa kaydırırken sessiz değer bozulması. Kullanım yerleri
(15 spinbox): `data_files_page.py`(1), `force_models_page.py`(6),
`frozen_search_page.py`(6), `live_telemetry_page.py`(1), `result_exports_page.py`(1)
+ tüm QComboBox'lar. `NumericDragLineEdit` QLineEdit tabanlı, etkilenmez.

**Karar (kullanıcı onaylı):** Focus'ta bile wheel KAPALI — tamamen kaldırılıyor.

**Değişiklikler:**
1. `ui_commons.py`'ye üç sınıf: `NoWheelSpinBox(QSpinBox)`,
   `NoWheelDoubleSpinBox(QDoubleSpinBox)`, `NoWheelComboBox(QComboBox)` —
   hepsinde `def wheelEvent(self, e): e.ignore()` (olay üst scroll area'ya düşer,
   sayfa kayar).
2. Yukarıdaki 5 dosyadaki tüm `QtWidgets.QSpinBox/QDoubleSpinBox/QComboBox`
   örneklerini bu sınıflarla değiştir (mekanik değişim). QComboBox'lar DAHİL
   (Telemetry "Plot Type" combo'sunun tekerlekle plot değiştirmesi aynı hata).
3. Klavye davranışı (ok tuşları, PageUp/Down) DEĞİŞMEZ — erişilebilirlik kaybı yok.

**Doğrulama:** UI testi: spinbox üzerine sentetik `QWheelEvent` gönder → değer
değişmemeli; aynı olayla üst scroll area kaymalı.
`grep -rn "QtWidgets.QSpinBox(\|QtWidgets.QDoubleSpinBox(\|QtWidgets.QComboBox(" src/lunaris/ui/pages/`
sıfır dönmeli.

---

## M5 — Arttır/azalt oklarının modernizasyonu (P0) — ✅ DONE (2026-07-07)

> Uygulandı (QSS yolu — tüm spinbox/combobox'ları kapsar): `ui_commons`'a
> `stepper_arrow_icons()` (qtawesome chevron PNG cache); `build_app_stylesheet`'e
> `spin_up_icon`/`spin_down_icon` string parametreleri (binding-neutral) + tam
> yükseklikte themed stepper şeridi QSS'i (22px, hover/pressed) ve combobox down-arrow;
> `_apply_theme` ikon yollarını üretip geçiriyor; `theme/__init__.py` wrapper **kwargs
> forward ediyor. NumericDragLineEdit migrasyonu YAPILMADI (davranış/sinyal değişimi
> riski) — QSS yolu tüm spinbox'ları zaten kapsıyor; migrasyon opsiyonel takip işi.


**Sorun:** `stylesheet.py`'de `::up-button/::down-button` kuralı YOK → native
minik oklar (~8px, tıklanamaz, temayla alakasız).

**Değişiklikler:**
1. **Tercih edilen yol:** Fiziksel/orbital sayısal alanlarda `NumericDragLineEdit`'e
   geçiş (drag + çift-tık; Orbit sayfası zaten bunu kullanıyor). Force Models ve
   Frozen Search'teki uygun spinbox'lar taşınır. (M4'teki NoWheel sınıflarıyla
   çakışmaz; taşınan alan spinbox olmaktan çıkar.)
2. **Kalan spinbox'lar için QSS** (`stylesheet.py`):
   `QSpinBox::up-button/::down-button` (ve QDoubleSpinBox/QDateTimeEdit) sağda tam
   yükseklikte tek dikey stepper şeridi: genişlik 22px, `bg_card_alt` zemin,
   `border_soft` sol ayraç, hover'da `bg_hover`. Ok görselleri: qtawesome chevron
   ikonlarından üretilmiş temalı pixmap'ler (`get_icon('fa6s.chevron-up', THEME['fg_soft'])`)
   veya QSS `image:` için data-URI SVG — renk MUTLAKA `THEME`'den.
3. Tıklama hedefi minimum 22×15px.

**Doğrulama:** Capture; scan_hardcoded_colors temiz.

---

## M6 — Live Telemetry alan kazanımı (P2)

Dosya: `src/lunaris/ui/pages/live_telemetry_page.py` (`MultiTelemetryPlot.__init__`
~123: iki satırlı toolbar; `TelemetryPage._build_ui` ~1093: KPI şeridi).

**Sorun:** Plot; 2 katlı header + KPI şeridi + 2 satırlık toolbar + console arasında
artık alana sıkışıyor (sayfa yüksekliğinin ~%35'i). Sayfanın amacı plot.

**Değişiklikler:**
1. M1'in kazandırdığı ~50-60px otomatik buraya yansır.
2. **Toolbar tek satıra iner:**
   - "Plot Type" combo'su yerine plot üstünde 4'lü segment kontrol
     (Altitude · Velocity · Eccentricity · Ground Track) — Orbit sayfasındaki
     `segmentedControl` deseni yeniden kullanılır.
   - Y-range grubu (min/max/margin %/Apply) bir "Scale" butonunun açtığı popover'a
     katlanır (nadir kullanılan uzman kontrolleri — progressive disclosure).
   - Kalan tek satır: segment + Time unit combo + T+ checkbox + Fit + Clear All.
3. Plot widget: `stretch=1` + `setMinimumHeight(320)`.
4. KPI şeridi kalır; değerler `family_mono` (M0.4); boş durumda `--` yerine
   soluk "no signal".
5. Boş-durum overlay metni: "No active run." (M7).
6. Opsiyonel (ayrı PR): 2×2 small-multiples ızgara görünümü toggle'ı
   (4 pyqtgraph PlotWidget zaten kurulu, ~373-472).

**Doğrulama:** 1280×800'de plot alanı / sayfa yüksekliği oranı ≥%55 ölçülür
(capture üzerinden piksel sayımı).

---

## M7 — Pazarlama üslubu temizliği (P0)

**Sorun:** UI metinleri araç satıyor ("…you prefer…", "…automatically so you can
review the full orbit shape at a glance", "Monitor the active run and compare live
engineering signals").

**Kural seti:** ≤1 cümle; fiil+nesne; gerçek veya eylem bildirir; değer vaadi yok.
Yasak kelimeler: `prefer, automatically, seamless(ly), at a glance, effortless,
powerful, premium, beautiful`.

**Kapsam:** `PAGE_DESCRIPTIONS` (app.py ~65-75), tüm `Section(title, description)`
ikinci argümanları, `EmptyState` metinleri, tooltip'ler, Orbit Preview paragrafı.

**Örnek dönüşümler:**
- "Choose the orbit entry style you prefer. Related values stay synchronized
  automatically so you can review the full orbit shape at a glance."
  → "Entry mode determines which elements are editable; the rest are derived."
- "Monitor the active run and compare live engineering signals."
  → "Signals from the active run."
- "Start a mission run to stream live engineering signals here." → "No active run."
- "Two-body Keplerian preview for geometry, viewing angle, and period — the mission
  run adds the selected perturbations, so the propagated trajectory will differ
  from this idealized orbit."
  → "Two-body preview. The mission run adds the selected perturbations."

**Doğrulama:**
`grep -rniE "prefer|automatically|at a glance|seamless|effortless|powerful" src/lunaris/ui --include=*.py`
UI string'lerinde sıfır (kod yorumları hariç).

---

## M8 — CLI→UI parite (P3)

Kaynaklar: `src/lunaris/cli/options.py`, `src/lunaris/cli/batch_runner.py`,
`src/lunaris/cli/frozen_search.py`, `src/lunaris/cli/data.py`.
UI üretici: `src/lunaris/ui/core/command_builder.py`.

**Doğrulanmış boşluklar (grep ile teyitli, 2026-07-07):**

| # | CLI yeteneği | Bayraklar | UI durumu | Önerilen yer |
|---|--------------|-----------|-----------|--------------|
| 1 | Thermal IR + termal parametreler | `--enable-thermal-ir`, `--thermal-temperature-k`, `--thermal-night-temperature-k`, `--thermal-emissivity`, `--thermal-surface-albedo`, `--thermal-ir-coefficient`, `--thermal-floor-flux-w-m2`, `--thermal-facet-lat-count`, `--thermal-facet-lon-count` | Sadece `--enable-thermal` toggle | Force Models → Thermal satırına ⚙ dialog (Albedo dialog deseniyle aynı) |
| 2 | Tide parametreleri | `--tide-bodies`, `--tide-k2`, `--tide-k3`, `--tide-r-ref-m` | Sadece `--enable-tides` + `--tides-kind`; k2/k3 toggle'ları değer GÖNDERMİYOR | Solid Body Tides bölümünde genişletilebilir parametre satırları |
| 3 | Albedo fail-closed | `--albedo-require-provider` | Gönderilmiyor | Albedo dialog'una "Require real provider (fail-closed)" checkbox |
| 4 | Batch GPU ince ayar | `--torch-dtype`, `--torch-sh-chunk-size` | Gönderilmiyor | Batch Propagation → GPU bölümü "Advanced" katlanır alan |
| 5 | UQ kovaryans raporu | `--uq-report-dir` | Gönderilmiyor | Batch sayfasına "Generate UQ covariance report" toggle + çıktı dizini. (UQ roadmap'indeki bekleyen "UI panel" işinin ilk yarısı) |

**Doğrulama gerektiren adaylar** (uygulamadan önce sayfa kodu okunmalı):
- Frozen Search sayfası: `--resume/--no-resume`, `--refine-top-n`,
  `--refine-max-iterations`, `--no-figures` sayfada var mı?
- Data & Files sayfası: `lunaris.cli.data`'nın `--dry-run`, `--overwrite`,
  `--no-verify`, `--include-optional` seçenekleri sayfada karşılanıyor mu?
- UI'sız CLI araçları (ayrı ürün kararı, bu planın kapsamı dışında not olarak):
  perturbation-budget CLI, `validation/gravity_reference` çapraz doğrulama,
  `analysis/ensemble/result_audit`.

**Her parite satırı için uygulama deseni:** UI alanı ekle → `command_builder.py`'de
bayrağı emit et → `build_preflight_snapshot`'a alan ekle (gerekiyorsa) →
`tests/`'te command_builder unit testine satır ekle (builder saf fonksiyon,
Qt gerektirmez).

---

## Kabul kriterleri (genel)

1. `scan_hardcoded_colors.py` yeni ihlal göstermiyor.
2. `contrast_check.py` yeni metin/zemin çiftlerinde WCAG eşiklerini geçiyor.
3. Tüm sayfalarda önce/sonra capture alınmış (`outputs/ui/`).
4. UI test paketi yeşil; M4 için wheel regresyon testi eklenmiş.
5. Grayscale'e çevrilen ekran görüntüsünde hiyerarşi hâlâ okunuyor.
6. Hiçbir sayfada `QtWidgets.QSpinBox(`/`QDoubleSpinBox(`/`QComboBox(` doğrudan
   kullanımı kalmamış (M4 sonrası).
