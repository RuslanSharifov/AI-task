# Ev tapşırığı — Maşın öyrənməsi alqoritmləri və reqressiya metrikləri

**Mövzu:** Decision Tree, Random Forest, Gradient Boosting, XGBoost; MAE, MSE, RMSE, R²  
**Tapşırıqda göstərilən son tarix:** 03.11.2026, dərs başlayana qədər  
**Format:** Araşdırma və konspekt — kod yazmaq tələb olunmur.

> **Qeyd:** Bu sənəd ev tapşırığının bütün bəndlərini əhatə edən öyrənmə konspektidir. Misallar təlim məqsədi ilə qurulub; real dataset üzərində ölçülmüş nəticələr deyil. Məqsəd anlayışları öz sözlərinlə izah edə bilməkdir.

## Mündəricat

1. [Əsas anlayışlar](#0-əsas-anlayışlar)
2. [Decision Tree](#11-decision-tree--qərar-ağacı)
3. [Random Forest](#12-random-forest--təsadüfi-meşə)
4. [Gradient Boosting](#13-gradient-boosting--qradiyent-bustinqi)
5. [XGBoost](#14-xgboost--extreme-gradient-boosting)
6. [Dörd alqoritmin müqayisəsi](#15-dörd-alqoritmin-müqayisəsi)
7. [Reqressiya metrikləri](#2-reqressiya-metrikləri)
8. [Hesablama nümunəsi](#25-dörd-metrikin-eyni-nümunədə-hesablanması)
9. [Özünü yoxlama sualları](#3-özünü-yoxlama-sualları)
10. [Mənbələr](#4-mənbələr)

---

## 0. Əsas anlayışlar

**Maşın öyrənməsi (Machine Learning)** — kompüterin konkret qaydaların hamısı ayrıca yazılmadan məlumatdakı əlaqələri öyrənib yeni nümunələr üçün proqnoz verməsidir.

**Reqressiya (Regression)** — nəticənin ədədi qiymət olduğu problemdir. Məsələn, evin qiymətinin və ya çatdırılma vaxtının proqnozlaşdırılması.

**Xüsusiyyətlər (features, `X`)** — proqnoz vermək üçün istifadə olunan giriş məlumatları: evin sahəsi, yataq otaqlarının sayı və s. **Hədəf (target, `y`)** — tapmaq istədiyimiz nəticə: evin qiyməti.

**Training data** — modelin öyrəndiyi məlumatlar. **Test data** — əvvəldən görmədiyi nümunələrdə nə qədər düzgün proqnoz verdiyini qiymətləndirmək üçün ayrılmış məlumatlar.

**Overfitting (həddindən artıq uyğunlaşma)** — modelin təlim nümunələrinə, o cümlədən təsadüfi səs-küyə, həddindən artıq uyğunlaşması; nəticədə yeni məlumatlarda zəif işləməsi. **Underfitting (kifayət qədər öyrənməmə)** — modelin əsas əlaqələri də öyrənə bilməyəcək qədər sadə olması.

**Ensemble learning** — bir neçə modelin nəticəsini birləşdirərək proqnoz vermək yanaşmasıdır.

---

## 1. Maşın öyrənməsi alqoritmləri

### 1.1. Decision Tree — qərar ağacı

#### Nədir və necə işləyir?

Decision Tree məlumatları ardıcıl **əgər–onda (if–else)** tipli şərtlərlə bölən modeldir. Ağacın hər daxili düyünündə bir xüsusiyyətə aid sual, budaqlarda sualın nəticəsi, yarpaqlarda isə proqnoz yerləşir.

Ev qiymətinin proqnozlaşdırılmasına aid sadələşdirilmiş sxem:

```text
                  Ev böyükdür?
                  /          \
                Bəli         Xeyr
                /               \
        Yaxşı ərazidədir?     50 000
            /       \
          Bəli      Xeyr
         /            \
      120 000        80 000
```

Bu rəqəmlər yalnız izah üçündür. Real reqressiya ağacı təlim məlumatlarından öyrəndiyi bölünmə sərhədlərini istifadə edir. Ağac hər mərhələdə elə bir bölünmə seçməyə çalışır ki, yaranan qrupların daxilində hədəf qiymətləri daha oxşar olsun. Kvadratik xəta meyarı istifadə edildikdə yarpaqdakı proqnoz adətən həmin yarpağa düşən təlim qiymətlərinin **ortalamasıdır**.

**İş ardıcıllığı:** xüsusiyyət seçimi → məlumatın iki hissəyə bölünməsi → alt qruplarda yenidən bölünmə → dayandırma şərti → yarpaqda proqnoz.

#### Üstünlükləri

- Qərar yolu şərtlərlə təsvir edilə bildiyindən **izahı və vizuallaşdırılması asandır** (xüsusilə kiçik ağacda).
- Rəqəmsal məlumatlardakı qeyri-xətti əlaqələri və xüsusiyyətlərin qarşılıqlı təsirini öyrənə bilir.
- Adətən xüsusiyyətlərin miqyaslanmasını (`StandardScaler` və s.) tələb etmir.

#### Çatışmazlıqları

- Dərin ağac asanlıqla **overfitting** edə bilər.
- Təlim məlumatındakı kiçik dəyişiklik ağacın quruluşunu ciddi dəyişə bilər: **yüksək dəyişkənlik (variance)**.
- Tək ağacın proqnozları bəzi mürəkkəb problemlərdə qeyri-sabit ola bilər.

**Riskin azaldılması:** `max_depth`, `min_samples_leaf`, `min_samples_split` kimi məhdudiyyətlər qoymaq; nəticəni test və ya cross-validation ilə yoxlamaq.

**Tətbiq nümunəsi:** Evin sahəsi, otaq sayı və yerləşdiyi ərazi əsasında qiymətin təxmin edilməsi. Həmçinin təsnifat üçün də istifadə olunur: məsələn, müraciətin riskli olub-olmaması.

**Nə zaman zəif seçim ola bilər?** Bir neçə kiçik dəyişikliklə nəticənin kəskin dəyişməsinin qəbuledilməz olduğu və yüksək stabillik tələb olunan problemlərdə tək ağac əvəzinə ensemble model daha uyğun ola bilər.

### 1.2. Random Forest — təsadüfi meşə

#### Nədir və necə işləyir?

Random Forest **çoxlu qərar ağacını birləşdirən** ensemble üsuludur. Ağaclar, adətən, təlim məlumatından **bootstrap** adlanan təkrar seçimlə götürülmüş fərqli nümunələr üzərində öyrədilir. Bölünmələr zamanı xüsusiyyətlərin təsadüfi alt çoxluğu nəzərdən keçirilir. Bu təsadüfilik ağacların eyni səhvləri etməsini azalda bilər.

Reqressiyada yekun cavab ağacların proqnozlarının **orta qiymətidir**.

**Sadə nümunə:** Üç ağac bir ev üçün 60 000, 65 000 və 70 000 proqnozlaşdırırsa, meşənin proqnozu `(60 000 + 65 000 + 70 000) / 3 = 65 000` olar.

```text
Ağac 1 ── 60 000 ──┐
Ağac 2 ── 65 000 ──┼──► Orta ──► 65 000
Ağac 3 ── 70 000 ──┘
```

#### Üstünlükləri

- Tək Decision Tree ilə müqayisədə, çox vaxt **daha stabil** olur və overfitting riskini azaldır.
- Qeyri-xətti əlaqələri yaxşı öyrənir.
- Müxtəlif xüsusiyyətlərin əhəmiyyətinə dair göstəricilər əldə etmək mümkündür (lakin bu göstəricilər səbəb-nəticə əlaqəsi sübut etmir).
- Ağacların bir hissəsini paralel öyrətmək mümkündür.

#### Çatışmazlıqları

- Tək ağacdan daha çox **vaxt və yaddaş** sərf edə bilər.
- Yüzlərlə ağacın birlikdə qərarını bir ağac qədər asan izah etmək olmur.
- Daha çox ağac **həmişə** daha yaxşı test nəticəsi demək deyil; müəyyən nöqtədən sonra qazanc cüzi ola bilər.
- Təlimdə görülən hədəf aralığından kənara ekstrapolyasiya etməkdə ağac əsaslı reqressiya modelləri adətən zəifdir.

**Əsas parametrlər:** `n_estimators` (ağac sayı), `max_depth` (dərinlik), `max_features` (bölünmədə nəzərə alınan xüsusiyyətlər), `min_samples_leaf` (yarpaqdakı minimum nümunə sayı).

**Tətbiq nümunəsi:** Evin ölçüsü, otaqları, qarajı və digər xüsusiyyətlərinə əsasən qiymətin proqnozlaşdırılması. Həmçinin kredit riski və müştəri davranışı modellərində istifadə oluna bilər.

**Nə zaman zəif seçim ola bilər?** Çox məhdud hesablama resursu olduqda və ya nəticənin konkret qayda şəklində izahı əsas tələb olduqda kiçik Decision Tree üstün ola bilər.

### 1.3. Gradient Boosting — qradiyent bustinqi

#### Nədir və necə işləyir?

Gradient Boosting də çoxlu modelin birləşməsidir, amma Random Forest-dən fərqli olaraq **ağaclar ardıcıl qurulur**. Hər yeni ağac əvvəlki mərhələlərdə qalan proqnoz səhvini azaltmağa çalışır.

Sadələşdirilmiş təsvir (kvadratik xəta ilə reqressiya üçün):

1. Başlanğıc proqnoz verilir; bu, çox vaxt təlim hədəflərinin ortasıdır.
2. Həqiqi qiymətlə proqnoz arasındakı **qalıq** (səhv) hesablanır.
3. Yeni kiçik ağac həmin qalıqlardakı qanunauyğunluğu öyrənir.
4. Yeni ağacın düzəlişi `learning_rate` ilə kiçildilərək ümumi proqnoza əlavə edilir.
5. Proses ağaclar qurulduqca davam edir.

```text
İlk proqnoz
     │
     ▼
Səhvləri hesabla
     │
     ▼
Yeni ağac səhvləri öyrənsin
     │
     ▼
Kiçik düzəliş əlavə et
     │
     └────► Növbəti mərhələdə yenidən yoxla
```

**Niyə «qradiyent» adlanır?** Ümumi Gradient Boosting yanaşması itki funksiyasını azaltmaq üçün onun **mənfi qradiyenti** istiqamətində düzəlişlər qurur. Kvadratik xəta halında bu düzəlişlər qalıqlarla əlaqəlidir; başqa itki funksiyalarında sadəcə `həqiqi − proqnoz` formasında olmaya bilər.

#### Üstünlükləri

- Mürəkkəb, qeyri-xətti əlaqələri yaxşı öyrənə bilər.
- Düzgün parametrlərlə tabulyar məlumatlarda yüksək proqnoz keyfiyyəti verə bilər.
- İtki funksiyasına uyğun şəkildə səhvləri mərhələli azaltmağa çalışır.

#### Çatışmazlıqları

- `learning_rate`, ağac sayı və ağac dərinliyi yanlış seçilərsə overfitting və ya underfitting mümkündür.
- Ağacların ardıcıl asılılığı səbəbindən klassik Gradient Boosting-in öyrədilməsi tam paralelləşdirilmir.
- Parametr seçiminə Random Forest-dən daha həssas ola bilər.

**Əsas parametrlər:** `n_estimators`, `learning_rate`, `max_depth`, `subsample`.

**Tətbiq nümunəsi:** Satış həcminin, ev qiymətinin və ya müştərinin gözlənilən xərclərinin təxmin edilməsi.

**Nə zaman zəif seçim ola bilər?** Vaxt məhduddursa və yüksək dəqiqlik üçün uzun parametr axtarışı aparmaq mümkün deyilsə, Random Forest daha sadə başlanğıc variantı ola bilər.

### 1.4. XGBoost — Extreme Gradient Boosting

#### Nədir və necə işləyir?

**XGBoost (Extreme Gradient Boosting)** gradient boosting prinsipini yüksək səmərəlilik, miqyaslana bilmə və mürəkkəbliyə nəzarət imkanları ilə həyata keçirən kitabxana və alqoritmlər toplusudur. Ən çox istifadə olunan forması ağac əsaslı gradient boosting-dir.

Əsas məntiq Gradient Boosting ilə ortaqdır: **hər yeni ağac əvvəlki modelin nəticəsini yaxşılaşdırır**. XGBoost isə bu prosesi əlavə optimallaşdırmalarla həyata keçirir.

XGBoost-un mühüm xüsusiyyətləri:

- **Regularization (L1/L2):** modelin həddindən artıq mürəkkəbləşməsini cəzalandırmağa kömək edir.
- **İkinci dərəcəli məlumat:** optimallaşdırmada itki funksiyasının həm birinci, həm də ikinci törəmələrindən (qradiyent və Hessian) istifadə edə bilir.
- **Səmərəli bölünmə axtarışı və paralel hesablama:** xüsusən ağac daxilində mümkün bölünmələri hesablamaq üçün optimallaşdırmalar mövcuddur. **Bu, ağacların hamısının ardıcıl boosting asılılığını aradan qaldırmır.**
- **Çatışmayan dəyərlərin idarə edilməsi:** ağac əsaslı XGBoost bir çox hallarda `NaN` dəyərləri üçün uyğun istiqaməti öyrənə bilir; yenə də məlumatın hazırlanması və kateqorik xüsusiyyətlərin verilməsi istifadə olunan interfeysdən asılıdır.
- **Sətir və sütun altseçimi:** `subsample`, `colsample_bytree` kimi parametrlər mövcuddur.

#### Üstünlükləri

- Tabulyar məlumatlarda çox vaxt yüksək proqnoz keyfiyyəti əldə edir.
- Böyük məlumatlarda sürətli və resurs baxımından səmərəli işləmək üçün optimallaşdırılıb.
- Regularization və müxtəlif seçmə parametrləri overfitting riskini idarə etməyə imkan verir.
- Çatışmayan qiymətlərlə işləmək üçün daxili imkanları var.

#### Çatışmazlıqları

- Parametrlərin sayı çox olduğundan yeni başlayanlar üçün sazlama mürəkkəbdir.
- Yaxşı nəticə üçün diqqətli validation və parametr seçimi lazım ola bilər.
- Kiçik və sadə məsələlərdə əlavə mürəkkəblik əhəmiyyətli üstünlük verməyə bilər.
- Ensemble modeli tək qərar ağacı qədər şəffaf deyil.

**Əsas parametrlər:** `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_alpha` (L1), `reg_lambda` (L2).

**Tətbiq nümunəsi:** Ev qiymətlərinin proqnozlaşdırılması, kredit riskinin qiymətləndirilməsi, satışın proqnozlaşdırılması və tabulyar verilənlər əsasında digər reqressiya/təsnifat tapşırıqları.

**Nə zaman zəif seçim ola bilər?** Çox kiçik və sadə dataset üçün izahı asan model yetərli olduqda, XGBoost-un sazlama və hesablama mürəkkəbliyi lazımsız ola bilər.

### 1.5. Dörd alqoritmin müqayisəsi

| Meyar | Decision Tree | Random Forest | Gradient Boosting | XGBoost |
|---|---|---|---|---|
| Əsas ideya | Bir qərar ağacı | Çoxlu müstəqil/təsadüfiləşdirilmiş ağac | Bir-birinin səhvini azaldan ardıcıl ağaclar | Optimallaşdırılmış, regularizasiyalı gradient boosting |
| Ağacların əlaqəsi | Tək ağac | Ağaclar əsasən müstəqil qurulur | Ağaclar ardıcıl asılıdır | Boosting mərhələləri ardıcıl asılıdır |
| Reqressiyada yekun nəticə | Yarpağın proqnozu | Ağac proqnozlarının ortası | Addım-addım toplanan düzəlişlər | İtki və regularization nəzərə alınaraq toplanan düzəlişlər |
| Overfitting riski | Dərin ağacda yüksək | Adətən tək ağacdan aşağı | Parametrlərə həssas | Parametrlərə həssas; regularization var |
| İzaholunma | Kiçik ağacda yüksək | Daha çətin | Daha çətin | Daha çətin |
| Hesablama | Çox vaxt yüngül | Çox ağac resurs tələb edir | Ardıcıl öyrənmə baha ola bilər | Performans üçün optimallaşdırılmış, amma sazlanması mürəkkəbdir |
| Tipik üstünlük | Sadəlik, izah | Sabitlik | Dəqiq mərhələli düzəliş | Səmərəlilik, regularization, əlavə imkanlar |

#### Xüsusilə: Gradient Boosting və XGBoost arasında fərq

**Əsas fikir:** XGBoost, Gradient Boosting-dən tamamilə ayrı öyrənmə prinsipi deyil; **gradient boosting ailəsinin optimallaşdırılmış reallaşdırmalarından biridir**.

| Mövzu | Klassik Gradient Boosting (məs., scikit-learn `GradientBoostingRegressor`) | XGBoost-un ağac əsaslı modeli |
|---|---|---|
| Öyrənmə prinsipi | Ardıcıllıqla yeni ağac əlavə edir | Ardıcıllıqla yeni ağac əlavə edir |
| Optimallaşdırma | İtki funksiyasının mənfi qradiyenti ilə işləyir | Tipik ağac optimallaşdırmasında birinci və ikinci törəmələrdən istifadə edir |
| Regularization | Ağac dərinliyi, yarpaq ölçüsü, learning rate və s. ilə mürəkkəblik idarə olunur | Bunlardan əlavə açıq L1/L2 və bölünmə cəriməsi kimi mexanizmlər verir |
| Çatışmayan məlumat | Klassik `GradientBoostingRegressor` adətən `NaN` üçün əvvəlcədən hazırlıq tələb edir | Ağac əsaslı XGBoost `NaN` istiqamətlərini öyrənə bilir |
| Hesablama | Klassik reallaşdırma böyük datada yavaş ola bilər | Təlimin müxtəlif mərhələləri üçün optimallaşdırmalar və paralel hesablamalar var |
| Sazlama | Nisbətən az parametr | Daha geniş parametr seçimi |

**Vacib dəqiqləşdirmə:** «XGBoost həmişə Gradient Boosting-dən daha yaxşı və sürətlidir» demək **yanlışdır**. Datasetin ölçüsü, parametrlər, istifadə olunan reallaşdırma və aparat resursları nəticəyə təsir edir. Bundan başqa, scikit-learn-də ayrıca yüksək performanslı `HistGradientBoostingRegressor` də var; buna görə bütün Gradient Boosting sistemlərini «yavaş» adlandırmaq düzgün deyil.

**Yadda saxlamaq üçün bir cümlə:** *Decision Tree tək ağacdır; Random Forest ağacların ortasıdır; Gradient Boosting səhvləri ardıcıl düzəldir; XGBoost isə bu ardıcıl yanaşmanı əlavə regularization və hesablama optimallaşdırmaları ilə həyata keçirir.*

---

## 2. Reqressiya metrikləri

Reqressiya metrikləri modelin **ədədi proqnozlarının həqiqi qiymətlərə nə qədər yaxın olduğunu** ölçür.

İşarələr:

- `n` — müşahidələrin sayı;
- `yᵢ` — i-ci müşahidənin həqiqi qiyməti;
- `ŷᵢ` — i-ci müşahidənin proqnozlaşdırılan qiyməti;
- `ȳ` — həqiqi qiymətlərin orta qiyməti;
- `eᵢ = yᵢ − ŷᵢ` — proqnoz xətası (qalıq).

### 2.1. MAE — Mean Absolute Error (orta mütləq xəta)

**Düstur:**

$$
\mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

**Mənası:** Hər proqnozun mütləq xətasını hesablayıb ortasını tapır. Mənfi və müsbət səhvlər bir-birini ləğv etmir.

**Şərh:** `MAE ≥ 0`; **0 ən yaxşıdır**, kiçik nəticə daha yaxşıdır. Ölçü vahidi hədəflə eynidir (məsələn, qiymət hansı valyutadadırsa, MAE də həmin vahiddədir).

**Böyük xətalara reaksiya:** Xəta iki dəfə artarsa, həmin müşahidənin MAE-yə töhfəsi iki dəfə artır — **xətti cəza**. Buna görə kvadratlaşdıran MSE və RMSE-dən adətən daha az həssasdır, amma kənar xətalara tam laqeyd deyil.

**Sadə nümunə:** Həqiqi qiymət 100, proqnoz 110 olduqda mütləq xəta `|100 − 110| = 10`.

**Nə vaxt faydalıdır?** «Orta hesabla neçə vahid yanılıram?» sualına sadə və anlaşılan cavab istəyəndə.

### 2.2. MSE — Mean Squared Error (orta kvadratik xəta)

**Düstur:**

$$
\mathrm{MSE}=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

**Mənası:** Hər xətanı **kvadrata yüksəldir**, sonra ortasını tapır.

**Şərh:** `MSE ≥ 0`; **0 ən yaxşıdır**, kiçik nəticə daha yaxşıdır. Ölçü vahidi **kvadrat vahiddir**, buna görə gündəlik izahı MAE/RMSE qədər rahat deyil.

**Böyük xətalara reaksiya:** Xəta iki dəfə artırsa, həmin xətanın kvadratı **dörd dəfə** artır. Yüksək xətaları güclü cəzalandırır.

**Sadə nümunə:** Xəta 10-dursa, kvadratı `100`; xəta 30-dursa, kvadratı `900`-dür. 30 vahidlik xəta 10 vahidlik xətadan üç dəfə böyük olsa da, MSE-də doqquz dəfə çox təsir göstərir.

**Nə vaxt faydalıdır?** Böyük proqnoz səhvlərini xüsusilə azaltmaq lazım olduqda; həmçinin bir çox reqressiya modelinin optimallaşdırma məqsədi kimi.

### 2.3. RMSE — Root Mean Squared Error (orta kvadratik xətanın kvadrat kökü)

**Düstur:**

$$
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}=\sqrt{\mathrm{MSE}}
$$

**Mənası:** MSE-nin kvadrat kökünü alaraq xəta ölçüsünü **hədəfin ilkin vahidinə** qaytarır.

**Şərh:** `RMSE ≥ 0`; **0 ən yaxşıdır**, kiçik nəticə daha yaxşıdır.

**Böyük xətalara reaksiya:** Əvvəlcə xətalar kvadratlaşdırıldığı üçün böyük səhvlərə həssasdır. Kvadrat kök nəticənin vahidini bərpa edir; böyük xətalara həssaslığı aradan qaldırmır.

**Sadə nümunə:** `MSE = 300` olduqda `RMSE = √300 ≈ 17,32`.

**Nə vaxt faydalıdır?** Böyük səhvlərə diqqət yetirməklə yanaşı nəticəni qiymətlə eyni vahiddə görmək istəyəndə.

### 2.4. R² — determinasiya əmsalı (`r2_score`)

**Düstur:**

$$
R^2=1-\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}{\sum_{i=1}^{n}(y_i-\bar y)^2}
$$

Burada surətdə **modelin kvadratik səhvlərinin cəmi** (`SSE`), məxrəcdə **həqiqi qiymətlərin orta qiymətdən kvadratik yayınmalarının cəmi** (`SST`) yerləşir.

**Mənası:** Modelin proqnozlarını bütün müşahidələr üçün sadəcə həqiqi qiymətlərin ortasını proqnozlaşdıran sadə başlanğıc üsulla müqayisə edir.

**Şərh (standart, sabit terminli və ən az iki müşahidəli hal):**

- `R² = 1`: mükəmməl proqnoz.
- `R² = 0`: modelin kvadratik xətası orta qiyməti proqnozlaşdırmaq qədərdir.
- `0 < R² < 1`: model orta qiymət proqnozundan yaxşıdır.
- `R² < 0`: model orta qiymət proqnozundan **daha pisdir**.

**Vacib:** R²-nin mənfi olması mümkündür. Buna görə «R² həmişə 0 ilə 1 arasındadır» fikri yanlışdır. Həqiqi qiymətlərin hamısı eyni olduqda məxrəc sıfır olur və adi düstur xüsusi hal tələb edir; `scikit-learn`-in standart davranışı bu halları ayrıca idarə edə bilər.

**Böyük xətalara reaksiya:** Surətdə xətaların kvadratları olduğundan böyük səhvlər R²-ni ciddi aşağı sala bilər. Həmçinin R² **məlumatdakı hədəf dəyişkənliyindən** asılıdır; fərqli datasetlərdə eyni R² eyni miqdarda qiymət xətası demək deyil.

**Sadə nümunə:** `SSE = 1200`, `SST = 12500` olduqda `R² = 1 − 1200/12500 = 0,904`.

**Nə vaxt faydalıdır?** Modelin sadə orta qiymət proqnozu ilə müqayisədə nə qədər əlavə fayda verdiyini qiymətləndirəndə. Bununla yanaşı, MAE/RMSE-yə də baxmaq lazımdır.

### 2.5. Dörd metrikin eyni nümunədə hesablanması

**Təlim nümunəsi:** Dörd evin həqiqi qiymətləri və modelin proqnozları:

| Ev | Həqiqi qiymət (`y`) | Proqnoz (`ŷ`) | Xəta (`y−ŷ`) | Mütləq xəta | Xətanın kvadratı |
|---|---:|---:|---:|---:|---:|
| 1 | 100 | 110 | −10 | 10 | 100 |
| 2 | 150 | 140 | 10 | 10 | 100 |
| 3 | 200 | 190 | 10 | 10 | 100 |
| 4 | 250 | 280 | −30 | 30 | 900 |
| **Cəm** | **700** | **720** | **−20** | **60** | **1200** |

Bu qiymətlər sadə hesablama üçün götürülüb; real ev qiymətləri və ya `Housing.csv` nəticələri deyil.

**MAE:**

$$
\mathrm{MAE}=\frac{10+10+10+30}{4}=\boxed{15}
$$

**MSE:**

$$
\mathrm{MSE}=\frac{100+100+100+900}{4}=\boxed{300}
$$

**RMSE:**

$$
\mathrm{RMSE}=\sqrt{300}\approx\boxed{17{,}32}
$$

**R² üçün əvvəlcə həqiqi qiymətlərin ortasını tapırıq:**

$$
\bar y=\frac{100+150+200+250}{4}=175
$$

Sonra həqiqi qiymətlərin bu ortadan kvadratik yayınmalarını toplayırıq:

$$
SST=(100-175)^2+(150-175)^2+(200-175)^2+(250-175)^2
$$

$$
SST=5625+625+625+5625=12500
$$

Kvadratik səhvlərin cəmi `SSE = 1200` olduğu üçün:

$$
R^2=1-\frac{1200}{12500}=\boxed{0{,}904}
$$

**Nəticələrin mənası:** Bu nümunədə proqnozlar orta hesabla **15 vahid** mütləq səhvə malikdir; **RMSE ≈ 17,32**-dir. R² isə bu konkret nümunədə **0,904**-dür, yəni modelin kvadratik səhvi orta qiyməti proqnozlaşdıran başlanğıc üsulun səhvinin təxminən **9,6%-i** qədərdir. R²-ni «hər evin qiymətini 90,4% düzgün tapır» kimi şərh etmək **yanlışdır**.

#### Böyük səhvə həssaslığı müqayisə edək

Təsəvvür edək ki, yalnız dördüncü proqnoz **280 əvəzinə 330** olur. O zaman dördüncü xətanın mütləq qiyməti 30-dan 80-ə artır.

| Metrik | İlkin proqnozlar | Bir böyük xəta əlavə edildikdə | Hansı istiqamət yaxşıdır? |
|---|---:|---:|---|
| MAE | 15 | 27,5 | Aşağı |
| MSE | 300 | 1675 | Aşağı |
| RMSE | 17,32 | 40,93 | Aşağı |
| R² | 0,904 | 0,464 | Yuxarı |

**Niyə?** Son sətirdə xətanın kvadratı `30² = 900`-dən `80² = 6400`-ə qalxır. Buna görə MSE və RMSE böyük proqnoz səhvinə daha çox həssasdır; R² də kvadratik xətaların artması səbəbindən azalır.

### 2.6. Metriklərin qısa müqayisəsi

| Metrik | Nəyi ölçür? | Ən yaxşı nəticə | Vahidi | Böyük xətaya münasibət |
|---|---|---|---|---|
| **MAE** | Mütləq xətaların ortası | 0; azalan yaxşıdır | Hədəfin vahidi | Xətti təsir |
| **MSE** | Kvadratik xətaların ortası | 0; azalan yaxşıdır | Hədəf vahidinin kvadratı | Çox güclü cəza |
| **RMSE** | MSE-nin kvadrat kökü | 0; azalan yaxşıdır | Hədəfin vahidi | Böyük xətalara həssas |
| **R²** | Orta qiymət proqnozuna nəzərən uyğunluq | 1; yüksələn yaxşıdır | Vahidsiz | Kvadratik xətalar artdıqca azalır |

**Hansı metriki seçmək lazımdır?** Məqsəddən asılıdır: orta tipik yanılmanı anlamaq üçün MAE, böyük səhvlərin xüsusi baha başa gəldiyi hallarda RMSE/MSE, orta qiymət proqnozuna nisbətən faydanı qiymətləndirmək üçün R². Adətən bir neçə metriki birlikdə təhlil etmək daha düzgündür.

---

## 3. Özünü yoxlama sualları

Konspekti oxuduqdan sonra bu suallara **koda baxmadan, öz sözlərinlə** cavab verməyə çalış:

1. Qərar ağacı evi hansı məntiqlə qiymətləndirir?
2. Dərin Decision Tree niyə overfitting edə bilər?
3. Random Forest-də niyə bir neçə ağacın ortası alınır?
4. Random Forest və Gradient Boosting arasındakı ən əsas fərq nədir?
5. `learning_rate` hansı işi görür?
6. XGBoost nə üçün adi Gradient Boosting-dən ayrı adla tanınır?
7. XGBoost-da `reg_alpha` və `reg_lambda` nəyi idarə edir?
8. MAE ilə MSE böyük xətaya niyə fərqli reaksiya verir?
9. RMSE niyə MSE-dən daha asan şərh oluna bilər?
10. R² mənfi olarsa, bu nə deməkdir?
11. R² = 0,90 nəticəsini «proqnozlar 90% düzgündür» kimi demək niyə səhvdir?
12. Yeni modelin yaxşı olduğunu niyə yalnız təlim nəticəsinə baxaraq demək olmaz?

### Təqdimat üçün qısa yekun

**Alqoritmlər:** Decision Tree bir ağacla qərar verir; Random Forest çoxlu müxtəlif ağacın proqnozunu ortalayaraq stabilliyi artırmağa çalışır; Gradient Boosting ağacları ardıcıl qurub səhvləri düzəldir; XGBoost isə gradient boosting yanaşmasını daha inkişaf etmiş optimallaşdırma və regularization imkanları ilə həyata keçirir.

**Metriklər:** MAE orta mütləq səhvi, MSE orta kvadratik səhvi, RMSE isə MSE-nin ilkin vahiddə ifadəsini göstərir. R² modelin səhvini orta qiymətlə proqnoz verən üsulla müqayisə edir. MAE/MSE/RMSE üçün kiçik qiymət, R² üçün böyük qiymət daha yaxşıdır (R²-nin maksimumu 1-dir).

---

## 4. Mənbələr

Tapşırığın mətni əsas götürülüb; anlayışların texniki dəqiqliyi üçün aşağıdakı rəsmi sənədlərdən istifadə edilib:

1. scikit-learn — Decision Trees: https://scikit-learn.org/stable/modules/tree.html
2. scikit-learn — Ensemble methods (Random Forest, Gradient Boosting): https://scikit-learn.org/stable/modules/ensemble.html
3. scikit-learn — Regression metrics and scoring: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
4. scikit-learn — Metrics API: https://scikit-learn.org/stable/api/sklearn.metrics.html
5. XGBoost — Introduction to Boosted Trees: https://xgboost.readthedocs.io/en/stable/tutorials/model.html
6. XGBoost — Parameters: https://xgboost.readthedocs.io/en/stable/parameter.html
7. XGBoost — Notes on Parameter Tuning: https://xgboost.readthedocs.io/en/stable/tutorials/param_tuning.html

**Qeyd:** Bu mənbələrdən götürülən ümumi texniki məlumatlar, sadələşdirilmiş izahlar və müəllif tərəfindən qurulmuş hesablama nümunələri birləşdirilərək konspekt hazırlanıb.
