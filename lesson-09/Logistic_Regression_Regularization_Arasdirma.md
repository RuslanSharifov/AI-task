# Logistic Regression və Regularization — nəzəri araşdırma və konspekt

> **Mənbə tapşırıq:** `AI_LogisticRegression_Regularization_Nezeri_Tapsiriq.pdf` (2 səhifə). Bu sənəddə PDF-də göstərilən **8 + 11 = 19 mövzu** eyni ardıcıllıqla cavablandırılıb. Düsturlar və nümunələr mövzuların mənimsənilməsi məqsədilə genişləndirilib. Əlavə praktika: `Logistic_Regression_Regularization_Praktika.ipynb`.

## Öyrənmə xəritəsi

| Bölmə | Mövzular | Əsas sual |
|---|---|---|
| 1. Logistic Regression | 1.1–1.8 | Sinfi ehtimalla necə proqnozlaşdırırıq və necə qiymətləndiririk? |
| 2. Regularization | 2.1–2.11 | Modelin həddən artıq əzbərləməsinin qarşısını necə alırıq? |

**Əsas ardıcıllıq:** Məlumat → Train/Test bölgüsü → yalnız train üzərində preprocessing → Modelin öyrədilməsi → Test proqnozu → Uyğun metrikalar → Nəticənin şərhi.

---

# 1. LOGISTIC REGRESSION

## 1.1. Logistic Regression nədir və hansı məqsədlə istifadə olunur?

**Logistic Regression** giriş xüsusiyyətlərindən (`features`, **X**) istifadə edib müəyyən sinfə aid olma ehtimalını hesablamağa çalışan **supervised classification** alqoritmidir. Adında “Regression” olsa da, əsas istifadəsi **klassifikasiya**dır.

Məsələn, bank əməliyyatının fırıldaq (`1`) və ya normal (`0`) olduğunu, məktubun spam olub-olmadığını proqnozlaşdırır. Model `P(y=1 | X)` ehtimalını, məsələn `0.82` hesablayır; `0.5` həddi seçiləndə sinif `1` olur. **0.82 ehtimalı 82% dəqiqlik demək deyil**: konkret nümunə üçün modelin qiymətləndirməsidir və kalibrlənməmiş ola bilər.

**Niyə istifadə olunur?** Sürətli başlanğıc (baseline), başa düşülən əmsallar, ehtimal qiymətləndirməsi və nisbətən sadə qərar sərhədi verir. **Niyə hər zaman seçilmir?** Mürəkkəb qeyri-xətti əlaqələr üçün feature engineering və ya başqa modellər tələb oluna bilər.

## 1.2. Regression və Classification arasındakı fərqlər

| Meyar | Regression | Classification |
|---|---|---|
| Məqsəd | Davamlı ədədi qiymət | Diskret sinif |
| Nümunə | Mənzilin qiyməti: 165000 | Kredit qaytarılacaq: 0/1 |
| Tipik model | Linear Regression, Ridge | Logistic Regression |
| Tipik metrikalar | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |

**Vacib fərq:** Klassifikasiya modelinin ehtimal çıxışı rəqəmdir, lakin öyrənmə hədəfi sinifdir. `LinearRegression` nəticəsini `0.5` ilə kəsmək Logistic Regression ilə eyni deyil; xətti reqressiyanın çıxışı `[0,1]` intervalından kənara çıxa bilər.

## 1.3. Logistic Regression modelinin işləmə prinsipi

1. Model xətti kombinasiya hesablayır: **`z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`**. Burada `w` əmsallar, `b` sərbəst həddir.
2. `z` qiyməti **sigmoid** funksiyasına göndərilir: **`p = σ(z)`**.
3. `p` birinci sinif ehtimalı kimi şərh edilir.
4. Seçilmiş `threshold` əsasında `0` və ya `1` çıxarılır.
5. Təlim zamanı `w, b`, adətən **binary cross-entropy / log loss** minimum edilməsi ilə öyrənilir:

   `L = -(1/m) Σ [yᵢ ln(pᵢ) + (1-yᵢ) ln(1-pᵢ)]`.

**Niyə log loss?** Yanlış, amma çox əmin proqnozları sərt cəzalandırır; ehtimal öyrənməsi üçün uyğundur. **Niyə sadəcə accuracy-ni optimallaşdırmırıq?** `0/1` qərarı üzərində accuracy fasiləli olduğundan birbaşa gradient-əsaslı optimallaşdırma çətindir.

**Mikro-nümunə:** `z = 1.386` olduqda `σ(z) ≈ 0.80`. Hədd `0.5`-dirsə nəticə `1`dir; hədd `0.9`-dursa nəticə `0` olur. Model eyni qalıb, yalnız **qərar qaydası** dəyişib.

## 1.4. Sigmoid funksiyası, düsturu və qrafiki

**Düstur:** `σ(z) = 1 / (1 + e^(-z))`.

| z | σ(z) (təxmini) |
|---:|---:|
| -4 | 0.018 |
| -2 | 0.119 |
| 0 | 0.500 |
| 2 | 0.881 |
| 4 | 0.982 |

```text
p
1.0 |                        ______
    |                    ___/
0.5 |---------------●---/------------   z=0 → p=0.5
    |          ____/
0.0 |_________/_____________________ z
             -      0       +
```

**Niyə sigmoid?** İstənilən real `z` qiymətini `(0,1)` aralığına gətirir; diferensiallana bilir (`σ'(z)=σ(z)(1-σ(z))`). **Məhdudiyyət:** Çox böyük müsbət/mənfi `z`-də əyri doyur, gradient kiçilə bilər. Notebook-da dəqiq qrafik qurulur.

## 1.5. Ehtimal əsasında klassifikasiya və Decision Boundary

**Threshold:** Standart ikili qayda `ŷ = 1`, əgər `p ≥ 0.5`; əks halda `0`.

**Decision boundary:** Modelin qərarı dəyişdiyi nöqtələr toplusudur. `0.5` üçün `σ(z)=0.5` ⇒ `z=0` ⇒ `w·x+b=0`. İki ölçülü girişdə bu düz xətt, daha çox ölçüdə hiper-müstəvi ola bilər. `t ≠ 0.5` üçün sərhəd `w·x+b = ln(t/(1-t))` olur.

**Niyə həddi dəyişmək olar?** Xəstəliyi qaçırmağın baha olduğu mühitdə həddi aşağı salmaq recall-u yüksəldə bilər, amma false positive-ləri də artıra bilər. Həddi **test dəstinə baxaraq seçmək** qərəzli qiymətləndirmə yaradır; validation dəsti istifadə etmək daha düzgündür.

## 1.6. Confusion Matrix: TP, TN, FP, FN

Müsbət sinfi `1` hesab edək:

| | Proqnoz 1 | Proqnoz 0 |
|---|---:|---:|
| **Həqiqət 1** | TP (düz müsbət) | FN (qaçırılan müsbət) |
| **Həqiqət 0** | FP (yanlış həyəcan) | TN (düz mənfi) |

**Misal:** 100 nümunədə `TP=30, FN=10, FP=5, TN=55`. `TP + FN + FP + TN = 100`. `sklearn.metrics.confusion_matrix(y_true, y_pred)` standart ikili etiket sırası `[0,1]` üçün `[[TN, FP], [FN, TP]]` qaytarır; vizual cədvəli kod nəticəsi ilə qarışdırma.

**Nə üçün lazımdır?** İki modelin accuracy-si eyni olsa belə, birinin çox `FN`, digərinin çox `FP` verməsi əhəmiyyətli fərqdir.

## 1.7. Accuracy, Precision, Recall və F1-score

| Metrika | Düstur | Hansı suala cavab verir? |
|---|---|---|
| Accuracy | `(TP+TN)/(TP+TN+FP+FN)` | Ümumi neçə proqnoz doğrudur? |
| Precision | `TP/(TP+FP)` | “Müsbət” dediklərimizin neçəsi doğrudur? |
| Recall (Sensitivity) | `TP/(TP+FN)` | Bütün real müsbətlərin neçəsini tapdıq? |
| F1-score | `2PR/(P+R)` = `2TP/(2TP+FP+FN)` | Precision və Recall balansı necədir? |

Yuxarıdakı nümunədə: **Accuracy=0.85**, **Precision≈0.857**, **Recall=0.75**, **F1≈0.80**.

**İmbalanced data problemi:** 1000 əməliyyatın 990-ı normaldırsa, “həmişə normaldır” modeli **99% accuracy** ala bilər, amma fırıldaqları tapmaq üçün `Recall=0` olacaq. Buna görə positive class-ı müəyyən et, confusion matrix, precision, recall və F1-i birlikdə yoxla. Bəzən PR-AUC/ROC-AUC da lazımdır. Payda 0 olduqda metrika ayrıca qaydaya əsasən hesablanır (`zero_division` seçimi).

## 1.8. Train/Test bölgüsü və StandardScaler-in əhəmiyyəti

`train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)` — təxmini 80% train, 20% test ayırır; `stratify` sinif nisbətini qorumağa kömək edir.

**StandardScaler:** `x_scaled = (x - μ_train) / σ_train`. Hər feature train orta qiymətinə görə mərkəzləşir, train standart sapmasına görə miqyaslanır. Bu, xüsusilə regularization tətbiq olunan modellərdə əmsal cəzalarının müqayisəsini daha mənalı edir.

**Yaxşı ardıcıllıq:** Əvvəl train/test ayır → scaler-i yalnız train-ə `fit` et → hər ikisinə `transform` et. Daha yaxşısı `Pipeline([('scale', StandardScaler()), ('model', LogisticRegression())])` istifadə et.

**Pis seçim — data leakage:** Bütün dataset üzərində `fit_transform` edib **sonra** train/test ayırmaq, test məlumatının orta/sapmasının təlim prosesinə sızmasına səbəb olur. `StandardScaler` ağac modelləri üçün çox vaxt vacib deyil, amma Logistic Regression/Ridge/Lasso üçün faydalıdır. Zaman sıralarında təsadüfi bölünmə əvəzinə vaxt sırasını qoru.

---

# 2. REGULARIZATION

## 2.1. Regularization nədir və nə üçün lazımdır?

**Regularization** təlim itki funksiyasına əlavə **cərimə (penalty)** daxil edərək modelin əmsallarını nəzarətdə saxlayan yanaşmadır:

`Optimallaşdırılan məqsəd = verilənlərə uyğunlaşma xətası + λ × mürəkkəblik cəriməsi`.

**Niyə?** Model train datasındakı təsadüfi səs-küyü öyrənib yeni məlumatda zəif nəticə verə bilər. Əmsalları məhdudlaşdırmaq bəzən generalization-u yaxşılaşdırır. **Niyə həddindən artıq regularization pisdir?** Əhəmiyyətli siqnallar da zəifləyir, underfitting yaranır.

## 2.2. Overfitting və Underfitting anlayışları

| Hal | Train xətası | Test xətası | Şərh |
|---|---|---|---|
| Underfitting | Yüksək | Yüksək | Model kifayət qədər öyrənmir |
| Uyğun fit | Aşağı | Aşağı | Ümumiləşdirmə nisbətən yaxşıdır |
| Overfitting | Çox aşağı | Yüksək | Train detallarını əzbərləyir |

**Underfitting-ə qarşı:** Daha uyğun xüsusiyyətlər, daha çevik model, bəzən regularization-u azaltmaq. **Overfitting-ə qarşı:** Cross-validation, daha çox münasib data, sadələşdirmə, regularization. **Diqqət:** Train/test fərqi təkcə overfitting sübutu deyil; dataset shift və sızma da yoxlanmalıdır.

## 2.3. L1 və L2 Regularization arasındakı fərqlər

- **L1 norm:** `||w||₁ = Σ|wⱼ|` — bəzən əmsalları **dəqiq sıfır** edir, implicit feature selection yaradır.
- **L2 norm kvadratı:** `||w||₂² = Σwⱼ²` — böyük əmsalları daha çox cəzalandırır, adətən onları sıfıra yaxınlaşdırır.

| Xüsusiyyət | L1 | L2 |
|---|---|---|
| Əmsallara təsir | Bəziləri 0 ola bilər | Adətən kiçilir, tam 0 nadirdir |
| Feature selection | Bəli, potensial | Adətən yox |
| Korrelyasiyalı feature-lər | Seçim qeyri-sabit ola bilər | Çəkiləri bölüşdürməyə meyllidir |

**Niyə feature scaling lazımdır?** Əks halda böyük ölçü vahidli feature-in əmsalına düşən cərimə digərindən fərqli effekt verə bilər.

## 2.4. Ridge Regression — iş prinsipi və düsturu

Ridge **xətti reqressiyanın L2-regularization** versiyasıdır. `scikit-learn` məqsəd funksiyası:

`min_w [ ||y-Xw||₂² + α||w||₂² ]` (intercept standart quruluşda cəzalandırılmır).

`α` böyüdükcə əmsallar daha çox sıxılır; bu, variance-i azalda bilər, bias-ı artıra bilər. **Nə zaman?** Çox və bir-biri ilə korrelyasiyalı feature-lər olduqda yaxşı başlanğıcdır. **Nə zaman kifayət deyil?** Həqiqətən sıfır əmsallarla xüsusiyyət seçmək istəyirsənsə, Lasso-nu da sına.

## 2.5. Lasso Regression — iş prinsipi və düsturu

Lasso, **L1** cəriməsi ilə xətti reqressiyadır. `scikit-learn` məqsədi:

`min_w [ (1/(2m))||y-Xw||₂² + α||w||₁ ]`.

`α` artdıqda bəzi `w`-lər 0 ola bilər. **Üstünlük:** Daha sadə, interpretasiya oluna bilən model; eyni zamanda feature selection. **Risk:** Güclü korrelyasiyalı dəyişənlər içərisindən seçilən feature qeyri-sabit ola bilər; çox böyük `α` underfitting doğurar.

## 2.6. ElasticNet Regression — L1 və L2-nin birlikdə istifadəsi

`scikit-learn` məqsədi:

`min_w [ (1/(2m))||y-Xw||₂² + α·r||w||₁ + (α/2)(1-r)||w||₂² ]`, burada `r = l1_ratio`.

`r=1` → Lasso (L1); `0<r<1` → qarışıq; `r=0` → L2 tipli cərimə (praktikada sırf Ridge üçün `Ridge` istifadə et). **Niyə?** Feature seçmə təsiri ilə korrelyasiyalı dəyişənlərə qarşı L2 stabilliyini qismən birləşdirir.

## 2.7. Ridge, Lasso və ElasticNet müqayisəsi

| Metod | Cərimə | Tipik üstünlük | Nəzərə alınmalı məsələ |
|---|---|---|---|
| Ridge | L2 | Korrelyasiyalı çoxlu feature | Feature silmir |
| Lasso | L1 | Sparse model, feature selection | Korrelyasiyalı feature seçimi dəyişə bilər |
| ElasticNet | L1+L2 | Seçmə və stabillik arasında kompromis | Həm `alpha`, həm `l1_ratio` sazlanmalıdır |

**Vacib:** Bu üç sinif `Ridge/Lasso/ElasticNet` **regression** hədəfləri üçündür. Klassifikasiya üçün `LogisticRegression` modelinin L1/L2/Elastic-Net regularization variantları mövcuddur; **Ridge ilə Logistic Regression eyni model deyil**.

## 2.8. `alpha` və `l1_ratio` parametrlərinin rolu

- `alpha ≥ 0`: Ridge/Lasso/ElasticNet cəriməsinin gücünü idarə edir (böyük → daha güclü regularization).
- `l1_ratio ∈ [0,1]`: yalnız ElasticNet daxilində **L1 payını** idarə edir (`0` → L2; `1` → L1).
- **Mühüm fərq:** `LogisticRegression`-da `C` **regularization gücünün tərsini** idarə edir: **kiçik `C` → daha güclü regularization**. Bu, `alpha` ilə eyni istiqamətdə deyil.

**Yaxşı seçim:** Bir neçə `alpha` və `l1_ratio` dəyərini train üzərində cross-validation ilə müqayisə et. Test dəstini hiperparametr axtarışına qatma.

## 2.9. Feature Selection anlayışı

**Feature selection** proqnoz üçün lazım olan giriş sütunlarını seçməkdir. Məqsəd səs-küyü azaltmaq, sadəlik və bəzən performans qazanmaqdır.

- **Filter:** təkdəyişənli statistika (məs., `SelectKBest`).
- **Wrapper:** model performansına əsasən alt dəstləri yoxlama (`RFE`).
- **Embedded:** modelin özü seçir (məs., L1/Lasso, `SelectFromModel`).

**Pis seçim:** Bütün dataset-in hədəfinə baxıb ən yaxşı feature-ləri seçmək, sonra test performansını hesablamaq — **target leakage**. Feature selection-u da `Pipeline` daxilində train üzərində fit et. Seçilmək **səbəb-nəticə əlaqəsi** demək deyil.

## 2.10. `PolynomialFeatures` və `Pipeline` nə üçün istifadə olunur?

**PolynomialFeatures** giriş sütunlarından polinom və qarşılıqlı təsir feature-ləri yaradır. Məsələn `x₁, x₂` və `degree=2` üçün: `x₁, x₂, x₁², x₁x₂, x₂²` (seçimdən asılı olaraq əlavə sabit 1 sütunu da olur).

**Niyə?** Xətti model yeni feature fəzasında qeyri-xətti əlaqələri tuta bilir. **Risk:** Dərəcə və feature sayı böyüdükcə çoxlu əlavə sütun, overfitting və hesablama xərci yarana bilər.

**Pipeline** preprocessing + model mərhələlərini ardıcıllıqla birləşdirir:

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0)
)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
```

**Niyə scaler polynomial-dan sonra?** Yeni yaranan kvadrat/interaction feature-lərin də miqyası uyğunlaşdırılır. Böyük dərəcələrdə ölçü artımını və yaddaş sərfini ayrıca qiymətləndir.

## 2.11. RMSE və R² metrikalarının izahı

**RMSE (Root Mean Squared Error):**

`RMSE = sqrt((1/m) Σ(yᵢ-ŷᵢ)²)`.

Proqnoz xətasının **hədəflə eyni vahiddə** ölçüsüdür; **kiçik** olması yaxşıdır. Böyük xətaları kvadrata yüksəltdiyi üçün onlara həssasdır.

**R² (coefficient of determination):**

`R² = 1 - [Σ(yᵢ-ŷᵢ)² / Σ(yᵢ-ȳ)²]`.

`R²=1` ideal proqnoz; `0` həmin nümunələrdə orta qiyməti sabit proqnoz verməklə eyni kvadrat xəta; **mənfi** qiymət ortalama baseline-dan da zəif ola bilər. **R² “düz proqnoz faizi” deyil**. Sabit `y` üçün xüsusi hallar ayrıca qiymətləndirilir.

**Niyə bu metrikalar Logistic Regression-a tətbiq edilmir?** RMSE/R² əsasən davamlı hədəfin reqressiyası üçündür; Logistic Regression klassifikasiyadır. Əsasən confusion matrix, precision, recall, F1, log loss və b. istifadə edilir.

---

# 3. Yadda saxlanmalı vacib fərqlər

| Yanlış fikir | Düzgün yanaşma |
|---|---|
| “Logistic Regression adi regresiyadır.” | Adına baxmayaraq klassifikasiya modelidir. |
| “99% accuracy mütləq yaxşıdır.” | Sinif balansı, FN/FP qiyməti və recall/precision da lazımdır. |
| “Scaler-i bütün data üzərində fit etmək olar.” | Yalnız train üzərində fit et; Pipeline daha təhlükəsizdir. |
| “Ridge feature-ləri sıfırlayır.” | Əsasən Lasso əmsalları dəqiq sıfıra endirə bilir. |
| “Böyük `alpha` həmişə daha yaxşıdır.” | Həddindən artıq cərimə underfitting yaradır. |
| “Böyük `C` daha güclü cərimədir.” | LogisticRegression-da əksinədir: kiçik `C` daha güclüdür. |
| “R²=0.9 o deməkdir ki, 90% proqnoz düzdür.” | R² kvadrat xətaya əsaslanan uyğunluq göstəricisidir. |

## 4. Şifahi yoxlama üçün suallar

1. Niyə Logistic Regression-ın nəticəsi `0` və `1` arasında olur?
2. `threshold=0.5`-dən `0.3`-ə enəndə adətən precision/recall necə dəyişir?
3. Hansı hallarda accuracy təkbaşına aldadıcıdır?
4. Confusion Matrix-də `FN` ilə `FP` fərqi nədir?
5. `StandardScaler.fit()` niyə test dəstində çağırılmamalıdır?
6. Overfitting ilə underfitting necə ayırd edilir?
7. Ridge, Lasso və ElasticNet arasındakı başlıca kompromis nədir?
8. Lasso niyə feature selection üçün yararlıdır?
9. `alpha` ilə `C` hansı istiqamətdə işləyir?
10. `PolynomialFeatures(degree=2)` yeni hansı sütunları yaradır?
11. Niyə regression üçün RMSE/R², classification üçün Precision/Recall/F1 istifadə edirik?

## 5. Praktik məşq (müstəqil işləmək üçün)

1. Notebook-dakı təsnifat nümunəsini işə sal, sigmoid qrafikini izah et.
2. `threshold` dəyərlərini `0.3`, `0.5`, `0.7` et. TP, FP, FN, TN və F1 dəyişməsini qeydə al; səbəbini izah et.
3. `LogisticRegression(C=0.1)` və `C=10` müqayisəsi apar. `C`-nin məntiqini öz sözlərinlə yaz.
4. Regression nümunəsində `alpha`-nı dəyişərək Ridge/Lasso/ElasticNet RMSE, R² və sıfır əmsalların sayını yoxla.
5. `PolynomialFeatures` dərəcəsini `1` və `3` et, nəticəni müqayisə et; test datasına əsasən model seçməyin nə üçün səhv olduğunu izah et.

## 6. İstinadlar və yoxlama mənbələri

- Tapşırıq PDF-i (istifadəçinin təqdim etdiyi): `AI_LogisticRegression_Regularization_Nezeri_Tapsiriq.pdf`, səhifə 1–2.
- scikit-learn: [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- scikit-learn: [Linear Models — Logistic Regression](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
- scikit-learn: [Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), [Lasso](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html), [ElasticNet](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html)
- scikit-learn: [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html), [PolynomialFeatures](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html), [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- scikit-learn: [Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) və [Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics)

> **Qeyd:** Nəzəri izahda Ridge, Lasso və ElasticNet məqsəd funksiyaları `scikit-learn`-in öz normalizasiya konvensiyasına uyğun yazılıb. Müxtəlif kitablarda cəmlərin `m`, `2m` ilə normallaşdırılması fərqli ola bilər; `alpha` parametrləri buna görə ədədi baxımdan birbaşa eyni sayılmamalıdır. Buradakı kodlar və sintaksis kitabxana versiyasına görə kiçik fərqlər göstərə bilər.
