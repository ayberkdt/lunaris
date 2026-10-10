# ST-LRPS tasarım kararlarının Lean 4 formalizasyonu

Bu klasör, `outputs/training/st_lrps_train_20260917_233510` koşusunda kullanılan
ST-LRPS mimarisinin **neden** o şekilde kurulduğunu, Lean 4 + Mathlib'de
derlenen teoremlere bağlar. Her teorem, koşunun `config.json`'ındaki somut bir
değere ve `src/lunaris/surrogate/st_lrps/networks/models.py`'deki somut bir
satıra karşılık gelir.

Amaç "makine-doğrulanmış bir makale" değil; amaç, tasarımın dayandığı analitik
olguları **derlenen ve aksiyom-denetiminden geçen** ifadeler hâline getirip
belirsizliği kaldırmaktır. Kanıtlanan ile *varsayılan* aşağıda açıkça ayrılmıştır.

## Derleme ve denetim

```bash
cd formal
lake build                 # tüm teoremler
lake env lean Audit.lean   # her teorem için #print axioms
```

Araç zinciri: `leanprover/lean4:v4.33.1`, Mathlib `0df444a3` (etiket `v4.33.1`).
Derlenmiş Mathlib paketleri `.lake/packages/` altında (git dışı, ~7 GB); ağ
erişimi gerekmez. Derleme ve denetim çıktıları `build_logs/` içinde tutulur.

Beklenen denetim sonucu: her teorem yalnızca `propext`, `Classical.choice`,
`Quot.sound` aksiyomlarına dayanır; `sorryAx` hiçbir yerde görünmez.

## Modüller

| Dosya | Konu | Koşudaki değer |
|---|---|---|
| `StLrpsFormal/Activation.lean` | sinüs aktivasyonu | `activation = "sine"` |
| `StLrpsFormal/Init.lean` | başlangıç ağırlıkları | `w0_first = w0_hidden = 30`, `hidden = 512` |
| `StLrpsFormal/Bands.lean` | çok-ölçekli bantlar | `n_bands = 2`, `w0_bands = [15.3, 42.4]` |
| `StLrpsFormal/RadialDecay.lean` | radyal sönüm gömmesi | `embedding_type = "physical_radial_decay"`, `max_power = 4` |
| `StLrpsFormal/Residual.lean` | artık formülasyon, işaret, artık bloklar | `target_mode = "residual"`, `degree_min = 25`, `a_sign = +1` |
| `StLrpsFormal/CondonShortley.lean` | CS fazı | `4pi_geodesy_no_condon_shortley_v1` |

## Neden bunlar kullanılmalı — ayrıntılı liste

### 1. Sinüs aktivasyonu (`sine`)

ST-LRPS potansiyeli değil, potansiyelin **gradyanını** eğitir
(`potential_autograd`: `a = ∇ΔU`) ve Hutchinson Laplasyeni (`n_hutchinson_samples = 2`)
loglar. Dolayısıyla aktivasyonun birinci ve ikinci türevleri her yerde var ve
iyi huylu olmalıdır.

| Neden | Teorem | Ne diyor |
|---|---|---|
| Aktivasyon patlamaz | `sirenUnit_abs_le_one` | `|sin(w0 x)| ≤ 1` — ön-aktivasyon büyüklüğünü yalnızca ağırlıklar belirler |
| Gradyan analitik ve kapalı formda | `hasDerivAt_sirenUnit` | `d/dx sin(w0 x) = w0 cos(w0 x)` — autograd'ın döndürdüğü tam ifade |
| Türev yine bir SIREN | `deriv_sirenUnit_is_shifted_sine` | `w0 cos(w0 x) = w0 sin(w0 x + π/2)` — türev alınca fonksiyon sınıfı bozulmaz (Sitzmann 2020 §3.1) |
| Gradyan Lipschitz bütçesi `w0` ile kontrol edilir | `abs_deriv_sirenUnit_le` | `|w0 cos(w0 x)| ≤ |w0|` |
| Laplasyen için `C^∞` gerekir | `contDiff_sirenUnit` | her `n` için `ContDiff ℝ n` — ReLU'nun ikinci türevi h.h.y. sıfırdır, bu yüzden uygun değildir |
| `w0` uzamsal dalga boyunu belirler | `sirenUnit_periodic` | periyot `2π / w0` |

### 2. Başlangıç ağırlıkları (SIREN init)

`models.py`'deki gerçek kod:

```python
siren_init_first_:  bound = 1.0 / n_in                       # U(-1/n, 1/n)
siren_init_hidden_: bound = sqrt(6.0 / n_in) / w0            # U(-b, b)
head:               head_bound = 0.1 * (sqrt(6/hidden) / max(w0, 1))
SirenResBlock:      lin1 gizli-katman init, lin2 = 0
```

| Neden | Teorem | Ne diyor |
|---|---|---|
| `uniform_` çağrısının enjekte ettiği varyans | `uniform_second_moment` | `U(-b,b)` için ikinci moment `b²/3` (integral ile kanıtlanmış) |
| **Gizli sınırın `/w0` içermesinin sebebi** | `siren_hidden_bound_unit_variance` | `w0² · n · (b²/3) · ½ = 1` — ön-aktivasyon varyansı `n` ve `w0`'dan bağımsız olarak tam 1; her katman aynı rejimde başlar |
| İlk katman `[-1,1]`'de kalır | `first_layer_preactivation_bound` | `|wᵢ| ≤ 1/n`, `|xᵢ| ≤ 1` ⇒ `|Σ wᵢxᵢ| ≤ 1` |
| **`w0_first = 30`'un sebebi** | `first_layer_argument_bound` | ilk sinüsün argümanı `[-30, 30]` — ≈9,5 yarım-periyot; sıfır civarındaki doğrusal rejime sıkışmaz |
| Baş katman sönümlü | `head_init_is_damped` | `0.1·b < b` — ilk tahmin `ΔU ≈ 0`, analitik taban ilk adımda baskın |

**Varsayılan (kanıtlanmayan) adım:** geniş argümanlı `sin`'in ikinci momentinin
`½` olması (arksinüs yasası) ve katmanlar arası bağımsızlık. Bu, Sitzmann ve
ark. Teorem 1.8'in olasılıksal içeriğidir; burada girdi varsayımı olarak alınır.
Teorem, bu varsayım altında cebirin **tam olarak** 1 verdiğini gösterir.

### 3. Çok-ölçekli bantlar (`w0_bands = [15.3, 42.4]`)

`compute_harmonic_w0_bands(2, 25, 200)`: merkez dereceleri `lo = 26`, `hi = 200`
arasında geometrik yerleştirir ve `w0 = 3·√derece` (1 ondalık yuvarlama,
`[10, 100]` kırpma) uygular.

| Neden | Teorem | Ne diyor |
|---|---|---|
| Kural tersinir | `bandFreq_sq_div_nine` | `(3√d)²/9 = d` — bant frekansından hedef derece geri okunur |
| Bant sırası = spektrum sırası | `bandFreq_mono` | derece artınca frekans artar |
| İki bant tam uç noktalar | `band_centres_two_bands` | `t ∈ {0,1}` ⇒ merkezler tam `26` ve `200` |
| **15.3 nereden geliyor** | `band_low_bounds` | `3√26 ∈ (15.25, 15.35)` ⇒ yuvarlama 15.3 |
| **42.4 nereden geliyor** | `band_high_bounds` | `3√200 ∈ (42.35, 42.45)` ⇒ yuvarlama 42.4 |
| Kırpma etkisiz | `bands_inside_clamp` | iki değer de `(10, 100)` içinde — kaydedilen değerler saf `3√d` yasası |

Derin bloklar `w0_deep = w0_bands[-1] = 42.4`, birleştirme katmanı `w0_bands[0] = 15.3` kullanır.

**Varsayılan:** derece-`n` harmoniğinin karakteristik uzamsal frekansının `n`
ile orantılı olduğu (Jeans ilişkisi) ve `√`-yasasının uygun bir uzlaşma
olduğu — bunlar tasarım gerekçesidir, teorem değildir.

### 4. Radyal sönüm gömmesi (`physical_radial_decay`, `ρ¹…ρ⁴`)

`PhysicalRadialDecayEncoding` ağa `r_scaled`, birim vektör, `ρ, ρ², ρ³, ρ⁴`
(`ρ = r_ref / r`, `r_ref = 1738 km`) ve ham `x` verir: `1 + 3 + 4 + 3 = 11 = input_feature_dim`.
Küresel harmonik açılımda derece-`n` terimi `(R/r)^(n+1)` ile ölçeklenir; ağın
öğrendiği artık alan (26…200) geometrik sönümlü terimlerin toplamıdır.

| Neden | Teorem | Ne diyor |
|---|---|---|
| Eğitim kabuğunda `ρ < 1` | `rho_lt_one`, `rho_nonneg` | `0 ≤ R/r < 1` |
| Yüksek derece daha hızlı söner | `rho_pow_antitone`, `rho_pow_strictAnti` | `m < n ⇒ ρⁿ < ρᵐ` |
| Özellikler sınırlı, ölçekleme gerekmez | `rho_pow_mem_Ioo` | `0 < ρᵏ < 1` |
| **Artık alanın zarfı geometrik** | `tail_geometric` | `Σ_{n≥N} ρⁿ = ρᴺ/(1-ρ)` |
| Kesme sonlu kalıntı bırakır | `tail_summable` | `degree_max = 200`'de kesmek toplanabilir bir kuyruk bırakır |

**Ampirik bağ.** Bu sönüm, değerlendirmede ölçülen davranışı doğrudan öngörür
(`evals/publication_eval/angular_error_by_altitude.csv`, test, 2M örnek):

| İrtifa (km) | ort. açı hatası |
|---|---|
| 100–150 | 0.076° |
| 250–300 | 0.180° |
| 400–450 | 2.96° |
| 600–650 | 29.2° |
| 950–1000 | 88.1° |

Artık ivme `ρ^(n+1)` ile sıfıra giderken yönü gürültüye dönüşür; hata modelin
bozulması değil, sinyalin geometrik olarak kaybolmasıdır. Aynı dosyanın
`angular_error_by_accel_norm.csv` eşi bunu diğer taraftan gösterir:
`‖a‖ > 1e-7` olan 1,003,026 örnekte ortalama açı **1.96°**, `< 1e-9` olanlarda ~90°.

### 5. Artık formülasyon, işaret ve artık bloklar

| Neden | Teorem | Ne diyor |
|---|---|---|
| Taban tam, ağ yalnız artığı öğrenir | `residual_deriv_split`, `residual_fderiv_split` | `∇(U₀ + ΔU) = ∇U₀ + ∇ΔU` — tek boyutta ve herhangi bir normlu uzayda (ℝ³ dâhil) |
| `a_sign = +1` (jeodezi) | `hasDerivAt_pointMass`, `pointMass_deriv_neg` | `U = μ/r` için `dU/dr = -μ/r² < 0` ⇒ `+∇U` içe, yani çekici; fizik konvansiyonu `a = -∇Φ` için işaret `-1` olurdu ve sözleşme bunu kaydettiği için sessizce karışamaz |
| Derinlik başlangıçta zararsız | `resblock_identity_at_init` | `lin2 = 0` ⇒ `x + 0·h = x`; tüm paylaşımlı yığın başlangıçta birim dönüşüm |

### 6. Condon–Shortley fazı

Legacy eğitim verisi `(-1)^m` fazını yanlışlıkla uygulamıştı; çekirdek
(`src/lunaris/physics/spherical_harmonics.py`, satır 166–177) bunu kaldırdı.

| Neden | Teorem | Ne diyor |
|---|---|---|
| Tek `m`'de faz `-1` | `cs_phase_odd` | `(-1)^m = -1` |
| Çift `m`'de faz `1` | `cs_phase_even` | `(-1)^m = 1` |
| **Zonal testler kör** | `cs_phase_zonal_invariant` | `m = 0` etkilenmez — `J₂` testleri hatayı yakalayamaz |
| **Bozulma, yeniden parametrelendirme değil** | `cs_phase_changes_odd_term` | sıfır olmayan tek-`m` terimi değişir ⇒ farklı bir alan |
| Model kendi içinde tutarlı kalabilir | `cs_phase_involutive` | faz iki kez uygulanınca özdeşlik — legacy model *geçerli ama yanlış* bir alanı öğrenmişti |

Bu son iki teorem, legacy koşunun "kendi yanlış alanında başarılı" olmasının
ve bu koşunun onu doğru alanda yeniden üretmesinin matematiksel açıklamasıdır.

## Kanıtlanmayanlar (dürüstlük listesi)

1. SIREN varyans-yayılımının olasılıksal adımı (arksinüs yasası, bağımsızlık).
2. Küresel harmonik derece ↔ uzamsal dalga boyu ilişkisi ve `√`-yasasının optimalliği.
3. `round(…, 1)` ve `[10,100]` kırpmanın kendisi — yalnızca aralık sınırları kanıtlandı.
4. Ağın gerçekten yakınsadığı, genelleme yaptığı ya da hata sınırları — bunlar
   ampirik sonuçlardır ve yalnızca değerlendirme artefaktlarına atıfla verilir.
5. `SirenResBlock` içindeki LayerNorm'un etkisi.

## Kaynaklar

- Sitzmann, Martel, Bergman, Lindell, Wetzstein (2020). *Implicit Neural
  Representations with Periodic Activation Functions.* NeurIPS. (Teorem 1.8, §3.1–3.2)
- Koşu konfigürasyonu: `outputs/training/st_lrps_train_20260917_233510/config.json`
- Değerlendirme: `outputs/training/st_lrps_train_20260917_233510/evals/publication_eval/`
