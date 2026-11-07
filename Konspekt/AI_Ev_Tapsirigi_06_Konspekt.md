# Machine Learning — Ev tapşırığı №6
## Data Preprocessing və Regression Metrics

**Mövzular:** One-Hot Encoding vs `pd.factorize()`, `SimpleImputer`, MAE, MSE, RMSE, R²  
**Tapşırığın PDF-də göstərilən son tarixi:** 29.10.2026  
**İstinad dərsi:** [AI 06 — House Price Regression](https://github.com/Zamanof/Python_AI_Apllications_FSDE_Oct_24_6_az/blob/main/AI%2006/AI%2006.%20House%20Price%20Regression.ipynb)

> **Məqsəd:** Kateqorik verilənlərin rəqəmlərə çevrilməsini, boş dəyərlərin doldurulmasını və reqressiya modelinin səhvinin ölçülməsini anlamaq. Aşağıdakı hər əsas mövzuya işlək Python nümunəsi əlavə edilmişdir.

---

## 1. One-Hot Encoding vs `pd.factorize()`

### 1.1. One-Hot Encoding nədir və necə işləyir?

**One-Hot Encoding** kateqorik dəyərləri (məsələn, şəhər və ya rəng adlarını) 0 və 1-lərdən ibarət yeni sütunlara çevirir. Hər fərqli kateqoriya üçün ayrıca sütun yaradılır.

Məsələn:

| Şəhər | Bakı | Gəncə | Sumqayıt |
|---|---:|---:|---:|
| Bakı | 1 | 0 | 0 |
| Gəncə | 0 | 1 | 0 |
| Bakı | 1 | 0 | 0 |
| Sumqayıt | 0 | 0 | 1 |

**Nə üçün lazımdır?** Bir çox Machine Learning alqoritmi mətn dəyərlərini birbaşa qəbul etmir. One-Hot Encoding onları ədədi xüsusiyyətlərə çevirir və şəhərlərə saxta böyüklük/sıra münasibəti vermir: məsələn, `Gəncə = 2` olması onun `Bakı = 1`-dən “iki dəfə böyük” olduğu anlamına gəlmir.

### 1.2. `pd.factorize()` nədir?

`pd.factorize()` fərqli dəyərlərə bütöv ədədi kod verir və **iki nəticə** qaytarır:

1. `codes` — hər sətrin ədədi kodu;
2. `uniques` — kodların hansı ilkin dəyərlərə uyğun gəldiyi.

Məsələn, `["Bakı", "Gəncə", "Bakı", "Sumqayıt"]` üçün standart halda kodlar `[0, 1, 0, 2]` olur. Eyni kateqoriya eyni kod alır. Çatışmayan dəyər (`NaN`) standart halda `-1` ilə işarələnir.

**Vacib məqam:** Bu kodlar kateqoriyanın real ölçüsünü və ya sırasını göstərmir. Onları məsafə və ədədi böyüklük kimi qəbul edən modeldə yanlış əlaqə yarana bilər.

### 1.3. İki üsulun fərqi

| Meyar | One-Hot Encoding | `pd.factorize()` |
|---|---|---|
| Nəticə | Bir neçə 0/1 sütunu | Bir ədədi kod sütunu |
| Kateqoriyalar arasında sıra | Süni sıra yaratmır | Kodlar süni sıra təsiri yarada bilər |
| Sütun sayı | Kateqoriya sayı artdıqca artır | Bir sütun saxlanılır |
| Tipik istifadə | Nominal kateqoriyalar və çoxsaylı ML modelləri | Unikal dəyərləri kodlaşdırmaq, analiz, identifikatorlar |
| Risk | Çox kateqoriyada sütun sayı böyüyür | Model 0, 1, 2 kodlarını böyüklük kimi başa düşə bilər |

**Hansını seçmək lazımdır?** `rəng`, `şəhər`, `marka` kimi təbii sırası olmayan kateqoriyalar modelə veriləcəksə, xüsusilə xətti reqressiyada, çox vaxt One-Hot Encoding daha uyğundur. `pd.factorize()` sürətli kodlaşdırma və unikal dəyərlərin identifikasiyası üçün əlverişlidir, lakin onun kodları nominal kateqoriyalara həqiqi ədədi məna vermir. Çox kateqoriyalı sütunlarda əlavə alternativləri qiymətləndirmək lazımdır.

### 1.4. Python nümunəsi

```python
import pandas as pd

df = pd.DataFrame({
    "seher": ["Bakı", "Gəncə", "Bakı", "Sumqayıt"]
})

# 1) Factorize: kateqoriyalara 0, 1, 2... kodları veririk.
codes, uniques = pd.factorize(df["seher"])
print("Factorize kodları:", codes.tolist())
print("Kod xəritəsi:", dict(enumerate(uniques)))

# 2) One-Hot Encoding: hər kateqoriyaya ayrıca 0/1 sütunu.
one_hot = pd.get_dummies(df, columns=["seher"], dtype=int)
print("\nOne-Hot nəticəsi:")
print(one_hot)
```

**Nəticənin izahı:** `factorize()` eyni şəhərə eyni rəqəmi verir; `get_dummies()` isə `seher_Bakı`, `seher_Gəncə` və `seher_Sumqayıt` adlı indikator sütunları yaradır. `dtype=int` istifadə etdiyimiz üçün nəticə `True/False` əvəzinə `1/0` kimi görünür.

**06-cı dərslə əlaqə:** Notebookda həm `pd.factorize()`, həm də `pd.get_dummies(..., drop_first=True)` göstərilir. `drop_first=True` bir kateqoriyanın sütununu çıxarıb qalan sütunlarla onu ifadə edir; bu yanaşma bəzi xətti modellərdə tam xətti asılılığın qarşısını almağa kömək edir, amma **hər model üçün məcburi deyil**. Eyni notebookun ML `Pipeline` hissəsində `OneHotEncoder(handle_unknown="ignore")` işlədilir. Bu parametr test zamanı yeni kateqoriya çıxarsa kodun xəta ilə dayanmasının qarşısını alır.

---

## 2. `SimpleImputer`

### 2.1. `SimpleImputer` nədir?

`SimpleImputer` — `scikit-learn` kitabxanasında **çatışmayan dəyərləri** (`NaN`) seçilmiş qayda ilə dolduran vasitədir. Real verilənlərdə yaş, qiymət, sahə və ya şəhər məlumatları bəzən qeyd olunmur. Bir çox ML modeli belə boş dəyərlərlə birbaşa işləyə bilmir.

Misal: `[20, 30, NaN, 40]` → `mean` tətbiq etdikdə `[20, 30, 30, 40]`.

### 2.2. Dörd əsas strategiya

| Strategiya | Necə işləyir? | Nə vaxt uyğundur? |
|---|---|---|
| `mean` | Ədədi sütunun orta qiyməti ilə doldurur | Güclü kənar dəyərlər azdırsa |
| `median` | Sıralanmış ədədi dəyərlərin medianı ilə doldurur | Kənar (`outlier`) dəyərlər varsa |
| `most_frequent` | Ən çox təkrarlanan dəyərlə doldurur | Kateqorik və bəzi ədədi sütunlarda |
| `constant` | Özümüzün verdiyimiz sabit qiymətlə doldurur | Məsələn, `0` və ya `"Naməlum"` |

**Niyə həmişə `mean` seçmirik?** `[10, 20, 100]` nümunəsində orta qiymət `43.33`, median isə `20`-dir. `100` kimi kənar dəyər ortanı yuxarı çəkir; buna görə `median` bəzən daha düzgün təmsilçi olur.

### 2.3. Python nümunəsi — bütün strategiyalar

```python
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

numeric = pd.DataFrame({"yas": [10, 20, np.nan, 100]})
categorical = pd.DataFrame({
    "seher": ["Bakı", np.nan, "Bakı", "Gəncə"]
})

mean_imp = SimpleImputer(strategy="mean")
median_imp = SimpleImputer(strategy="median")
mode_imp = SimpleImputer(strategy="most_frequent")
constant_imp = SimpleImputer(strategy="constant", fill_value="Naməlum")

print("Mean:", mean_imp.fit_transform(numeric).ravel())
print("Median:", median_imp.fit_transform(numeric).ravel())
print("Most frequent:", mode_imp.fit_transform(categorical).ravel())
print("Constant:", constant_imp.fit_transform(categorical).ravel())
```

**Gözlənilən məntiq:**

- `mean`: boş yaş dəyəri təxminən `43.33` olur.
- `median`: boş yaş dəyəri `20` olur.
- `most_frequent`: boş şəhər dəyəri `"Bakı"` olur.
- `constant`: boş şəhər dəyəri `"Naməlum"` olur.

**`fit_transform()` nə edir?** `fit` sütun üzrə uyğun doldurma dəyərini öyrənir, `transform` isə həmin dəyərlə boş xanaları doldurur. İkisi birlikdə `fit_transform()` şəklində yazıla bilər.

**Vacib düzgün təcrübə:** Məlumat `train` və `test` hissələrinə ayrıldıqdan sonra `SimpleImputer` **yalnız `train` üzərində `fit`** edilməli, `test` üzərində isə sadəcə `transform` işlədilməlidir. Əks halda test məlumatı dolayısı ilə öyrənmə mərhələsinə daxil olar (*data leakage*). Ən rahat yanaşma `Pipeline` istifadə etməkdir.

**06-cı dərslə əlaqə:** Notebookda ədədi xüsusiyyətlər üçün `SimpleImputer(strategy="median")`, kateqorik xüsusiyyətlər üçün `SimpleImputer(strategy="most_frequent")` seçilir; sonrakı mərhələdə bu addımlar `Pipeline` və `ColumnTransformer` ilə birləşdirilir.

---

## 3. Regression Metrics — reqressiya metrikaları

Reqressiya modeli ev qiyməti kimi **ədədi nəticəni** proqnozlaşdırır. Proqnozun keyfiyyətini qiymətləndirmək üçün gerçək dəyərlər (`y`) və modelin proqnozları (`ŷ`) müqayisə olunur.

**İşarələr:** `n` — nümunə sayı; `yᵢ` — i-ci həqiqi dəyər; `ŷᵢ` — i-ci proqnoz; `ȳ` — həqiqi dəyərlərin ortası.

### 3.1. MAE — Mean Absolute Error

**Formula:**

$$
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|
$$

**Nəyi ölçür?** Səhvlərin modulunu götürüb orta qiymətini tapır. Səhvin istiqamətini (`+` və ya `−`) nəzərə almır.

**Şərh:** `MAE = 17.5` olarsa, proqnozların orta mütləq səhvi `17.5` vahiddir. **Kiçik olması yaxşıdır**, idealı `0`-dır. İlkin dəyərlə eyni ölçü vahidindədir.

### 3.2. MSE — Mean Squared Error

**Formula:**

$$
MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
$$

**Nəyi ölçür?** Hər səhvi kvadrata yüksəldib ortasını tapır. Böyük səhvləri daha güclü cəzalandırır.

**Şərh:** Kiçik MSE daha yaxşıdır; idealı `0`-dır. Kvadratlaşdırmaya görə ölçü vahidi də kvadrat olur (məsələn, `manat²`), buna görə birbaşa şərhi MAE-dən çətindir.

### 3.3. RMSE — Root Mean Squared Error

**Formula:**

$$
RMSE = \sqrt{MSE} =
\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
$$

**Nəyi ölçür?** MSE-nin kvadrat kökünü götürür. Böyük səhvlərə həssaslığını qoruyur, lakin nəticə yenidən ilkin vahidə qayıdır.

**Şərh:** Kiçik RMSE daha yaxşıdır; idealı `0`-dır. Böyük səhvlərin əhəmiyyətli olduğu hallarda faydalıdır.

### 3.4. R² — R-squared (determinasiya əmsalı)

**Formula:**

$$
R^2 = 1 -
\frac{\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}
{\sum_{i=1}^{n}(y_i-\bar{y})^2}
$$

**Nəyi ölçür?** Modelin, hər dəfə sadəcə həqiqi dəyərlərin ortasını proqnozlaşdıran sadə yanaşma ilə müqayisədə, hədəf dəyişkənliyini nə qədər yaxşı izah etdiyini göstərir.

**Şərh:**

- `R² = 1`: tam uyğun proqnoz.
- `R² = 0`: ortanı proqnozlaşdıran sadə model səviyyəsində nəticə.
- `R² < 0`: həmin sadə modeldən də zəif nəticə.
- `R² = 0.97`: bu nümunədə dəyişkənliyin təxminən **97%-i izah olunur**; bu, “97% düzgün cavab” mənasına gəlmir.

**Diqqət:** Həqiqi dəyərlər hamısı eynidirsə, R² üçün xüsusi hallar yaranır; çox kiçik test dəstlərində də nəticəni ehtiyatla qiymətləndirmək lazımdır.

### 3.5. Metrikaların müqayisəsi

| Metrika | Əsas xüsusiyyət | Daha yaxşı nəticə | Ölçü vahidi |
|---|---|---|---|
| **MAE** | Orta mütləq xəta | `0`-a yaxın | İlkin vahid |
| **MSE** | Böyük xətaları kvadratla cəzalandırır | `0`-a yaxın | Vahidin kvadratı |
| **RMSE** | Böyük xətalara həssasdır, şərhi asandır | `0`-a yaxın | İlkin vahid |
| **R²** | Orta proqnoza nisbətən izah gücü | `1`-ə yaxın | Vahidsiz |

### 3.6. Python nümunəsi — dörd metrikanın hesablanması

```python
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Həqiqi qiymətlər və proqnozlar
y_true = [100, 200, 300, 400]
y_pred = [110, 180, 330, 390]

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, y_pred)

print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")
```

**Gözlənilən nəticə:**

```text
MAE:  17.50
MSE:  375.00
RMSE: 19.36
R²:   0.97
```

**Əl ilə yoxlama:** Mütləq səhvlər `10, 20, 30, 10` → `MAE = 70 / 4 = 17.5`. Səhvlərin kvadratları `100, 400, 900, 100` → `MSE = 1500 / 4 = 375`; `RMSE = √375 ≈ 19.36`. Həqiqi qiymətlərin ortası `250` olduğuna görə `R² = 1 − 1500 / 50000 = 0.97`.

**06-cı dərslə əlaqə:** Notebookda qiymətlər `LinearRegression` ilə proqnozlaşdırılır və modelin nəticəsi məhz `mean_absolute_error`, `mean_squared_error`, `np.sqrt(mse)` və `r2_score` ilə ölçülür.

---

## 4. Yekun nəticə

1. **One-Hot Encoding** nominal kateqoriyaları model üçün 0/1 indikatorlarına çevirir; **`pd.factorize()`** isə kateqoriyaları bir sütunda bütöv rəqəmlərlə işarələyir.
2. **`SimpleImputer`** çatışmayan dəyərləri orta, median, ən çox rast gəlinən və ya sabit dəyərlə əvəz edir. Doldurma qaydasını yalnız öyrənmə (`train`) verilənlərindən hesablamaq lazımdır.
3. **MAE, MSE və RMSE** proqnoz səhvlərini müxtəlif üsullarla ölçür; hamısında kiçik nəticə yaxşıdır. **R²** isə modelin orta proqnozla müqayisədə nə qədər yaxşı olduğunu göstərir; böyük nəticə adətən daha yaxşıdır.

Beləliklə, *Data Preprocessing* modelə uyğun və etibarlı məlumat hazırlamağa, *Regression Metrics* isə modelin nəticəsini yoxlamağa xidmət edir.

---

## 5. İstifadə olunmuş mənbələr

**Tapşırıq və dərs materialları**
1. `AI_Ev_Tapsirigi_06.pdf` — müəllimin ev tapşırığı (29.10.2026).
2. [Kursun 06-cı dərsi: AI 06. House Price Regression.ipynb](https://github.com/Zamanof/Python_AI_Apllications_FSDE_Oct_24_6_az/blob/main/AI%2006/AI%2006.%20House%20Price%20Regression.ipynb) — `factorize`, One-Hot, `SimpleImputer`, reqressiya metrikaları və `Pipeline`.

**Rəsmi sənədlər (əlavə araşdırma)**
3. [pandas — `get_dummies()`](https://pandas.pydata.org/docs/reference/api/pandas.get_dummies.html).
4. [pandas — `factorize()`](https://pandas.pydata.org/docs/reference/api/pandas.factorize.html).
5. [scikit-learn — `OneHotEncoder`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).
6. [scikit-learn — `SimpleImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html).
7. [scikit-learn — Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html).
8. [scikit-learn — `root_mean_squared_error`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html).

> **Qeyd:** Bu konspekt PDF-dəki üç əsas mövzunu əhatə edir. Dərsin notebookunda əlavə olaraq EDA, `StandardScaler`, `MinMaxScaler`, feature engineering, `train_test_split` və cross-validation da var; onlar ev tapşırığının məcburi üç əsas mövzusuna daxil olmadığı üçün burada genişləndirilməyib.
