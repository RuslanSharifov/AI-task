# 🧠 Neyron şəbəkələrinin əsasları — PyTorch, Loss, Softmax və Optimizer

> **Python / AI / Ev tapşırığı**  
> **Araşdırmanın son tarixi:** 22 noyabr 2026  
> **Mövzular:** 01 — PyTorch & Neural Networks · 02 — Loss Functions · 03 — Softmax & Cross-Entropy · 04 — Optimizers  
> **Format:** GitHub Markdown (`.md`) + icra oluna bilən Jupyter Notebook (`.ipynb`)  
> **Səviyyə:** Başlanğıcdan orta səviyyəyə · **Dil:** Azərbaycan dili

Bu sənəd şəkildəki tapşırıq planına əsaslanır və onu genişləndirərək **“nədir?”, “necə işləyir?”, “niyə lazımdır?”, “harada istifadə olunur?”, “olmasa nə baş verər?”, “üstünlük və çatışmazlıqları nələrdir?”** suallarına cavab verir. Bütün əsas anlayışlar üçün kod nümunələri, tipik səhvlər, təcrübələr və yoxlama sualları verilir.

**Əsas fikir:** Neyron şəbəkəsi proqnoz verir → **loss** səhvi ölçür → `backward()` qradiyentləri hesablayır → **optimizer** parametrləri dəyişir → model tədricən öyrənir. **Softmax** isə çoxsinifli təsnifatda xam çıxışları ehtimallara çevirir.

---

## 📚 Mündəricat

1. [Başlamazdan əvvəl: əsas anlayışlar](#1-başlamazdan-əvvəl-əsas-anlayışlar)
2. [PyTorch və neyron şəbəkələri](#2-pytorch-və-neyron-şəbəkələri)
3. [Loss Functions — itki funksiyaları](#3-loss-functions--itki-funksiyaları)
4. [Softmax və Cross-Entropy](#4-softmax-və-cross-entropy)
5. [Optimizers — parametrlərin yenilənməsi](#5-optimizers--parametrlərin-yenilənməsi)
6. [Başdan sona praktiki model](#6-başdan-sona-praktiki-model)
7. [Real tətbiq ssenariləri və seçim qaydaları](#7-real-tətbiq-ssenariləri-və-seçim-qaydaları)
8. [Ən çox rast gəlinən səhvlər](#8-ən-çox-rast-gəlinən-səhvlər)
9. [Müqayisə cədvəlləri və yaddaş kartı](#9-müqayisə-cədvəlləri-və-yaddaş-kartı)
10. [Yoxlama sualları və praktiki tapşırıqlar](#10-yoxlama-sualları-və-praktiki-tapşırıqlar)
11. [Terminlər lüğəti](#11-terminlər-lüğəti)
12. [İstinadlar və daha çox oxu](#12-istinadlar-və-daha-çox-oxu)

---

## 1. Başlamazdan əvvəl: əsas anlayışlar

### 1.1. AI, Machine Learning və Deep Learning fərqi

- **Süni intellekt (AI):** İnsan intellektinə bənzəyən funksiyalar yerinə yetirən sistemlərin geniş sahəsi.
- **Maşın öyrənməsi (ML):** Verilənlərdən qanunauyğunluqlar öyrənərək proqnoz verən alqoritmlər.
- **Dərin öyrənmə (Deep Learning):** Çoxqatlı neyron şəbəkələrindən istifadə edən ML istiqaməti.
- **Neyron şəbəkəsi (Neural Network, NN):** Çəki (`weight`) və sürüşmə (`bias`) kimi öyrənilən parametrləri olan riyazi model.

**Vacib fərq:** Hər AI sistemi neyron şəbəkəsi deyil; hər ML problemi üçün də neyron şəbəkəsi ən yaxşı həll sayılmır. Kiçik cədvəl verilənlərində sadə modellər daha ucuz və izaholunan ola bilər.

### 1.2. Model nəyi öyrənir?

Bir giriş vektoru götürək:

$$x = [x_1, x_2, \ldots, x_d]$$

Sadə neyronun çıxışı:

$$z = w_1x_1 + w_2x_2 + \cdots + w_dx_d + b = w^Tx+b$$

Aktivasiya əlavə edildikdə:

$$a = f(z)$$

Burada:

| İşarə | Nədir? | Nəyə lazımdır? |
|---|---|---|
| `x` | Giriş xüsusiyyətləri | Modelə müşahidəni ötürür |
| `w` | Çəkilər | Hər girişin təsirini öyrənir |
| `b` | Bias | Qərar sərhədinin yerini dəyişməyə kömək edir |
| `f` | Aktivasiya funksiyası | Qeyri-xətti əlaqələri modelləşdirməyə imkan verir |
| `z` | Logit / xətti çıxış | Xam hesablamanın nəticəsidir; hələ ehtimal deyil |

**Çəkilər olmasa:** model giriş xüsusiyyətlərinin hansı dərəcədə vacib olduğunu öyrənə bilməz.  
**Bias olmasa:** model daha məhdud funksiyalar ailəsi ilə işləyər; bəzi qərar sərhədlərini qura bilməz.  
**Qeyri-xətti aktivasiya olmasa:** ardıcıl bir neçə xətti qat yenə də bir xətti çevrilməyə bərabər olar. Mürəkkəb qeyri-xətti qaydaları öyrənmək çətinləşər.

### 1.3. Şəbəkənin quruluşu

```text
Giriş qatı                Gizli qat                  Çıxış qatı
x1 (məsələn, məsafə) ──┐
x2 (məsələn, sürət)   ───┼──► [Linear + ReLU] ──► [Linear] ──► logits
x3 (məsələn, vaxt)   ───┘

                  İrəli keçid (forward)
      x ───────► model(x) ───────► proqnoz
                                      │
                         loss(proqnoz, doğru cavab)
                                      │
                           loss.backward()
                                      │
                           optimizer.step()
                                      │
                            yenilənmiş çəkilər
```

- **Input layer:** Verilənlərin daxil olduğu hissə.
- **Hidden layer:** Aralıq xüsusiyyətləri öyrənən qatlar.
- **Output layer:** Tapşırığa uyğun çıxış; məsələn, üç sinif üçün 3 logit.
- **Parameter:** `weight` və `bias`; təlim zamanı dəyişir.
- **Hyperparameter:** `learning_rate`, batch ölçüsü, qat sayı; biz seçirik.
- **Epoch:** Təlim verilənlərinin bütün nümunələri üzərindən bir keçid.
- **Batch:** Bir addımda istifadə olunan nümunələr qrupu.
- **Iteration/step:** Adətən bir batch üzrə parametr yenilənməsi.

### 1.4. Train, validation və test bölgüsü

| Bölmə | Funksiya | Qayda |
|---|---|---|
| **Train** | Parametrləri öyrətmək | `backward()` və `optimizer.step()` burada işləyir |
| **Validation** | Hiperparametr/model seçimi | Qradiyent və parametr yenilənməsi aparılmır |
| **Test** | Son, qərəzsiz qiymətləndirmə | Seçim prosesində təkrar-təkrar istifadə edilməməlidir |

**Data leakage (məlumat sızması):** Nəticəni birbaşa və ya dolayı yolla açıqlayan məlumatın təlimə daxil olması. Məsələn, oyun başlamamış nəticəni proqnozlaşdıran modelə oyunun yekun hesabını daxil etmək sızmadır; yüksək accuracy aldatsa da, real istifadədə işləməz.

---

## 2. PyTorch və neyron şəbəkələri

### 2.1. PyTorch nədir?

**PyTorch** — Python ekosistemində istifadə olunan, tensor əməliyyatlarını, avtomatik diferensiallamanı və neyron şəbəkələrinin təlimini dəstəkləyən açıq mənbəli maşın öyrənməsi kitabxanasıdır.

**Harada istifadə olunur?** Kompüter görməsi (şəkil tanıma), təbii dil emalı, səs, proqnozlaşdırma, tədqiqat prototipləri və sənaye ML sistemləri. CPU üzərində işləyə bilir; uyğun mühitdə GPU hesablamalarından da faydalanır.

**Niyə istifadə olunur?** `Tensor`, `autograd`, `nn.Module` və `optim` vasitəsilə sıfırdan törəmə və parametr yeniləmə mexanizmi yazmadan təlim qururuq.

**PyTorch olmasa:** Eyni konsepsiyaları TensorFlow/JAX və ya digər çərçivələrdə reallaşdırmaq mümkündür. Amma hər şeyi sıfırdan NumPy ilə yazmaq adətən daha çox kod, daha çox səhv ehtimalı və GPU/autograd üçün əlavə mühəndislik deməkdir. **PyTorch anlayış olaraq məcburi deyil; onun verdiyi imkanlar vacibdir.**

| Üstünlüklər | Məhdudiyyətlər / mənfi tərəflər |
|---|---|
| Python ilə rahat inteqrasiya | Böyük modellər yüksək RAM/VRAM tələb edə bilər |
| Avtomatik qradiyent hesablanması | Tensor ölçüsü/`device` səhvləri yeni başlayanları çaşdırır |
| `nn.Module`, `DataLoader`, `optim` ekosistemi | GPU/CUDA uyğunluğu mühitə görə fərqlənir |
| Araşdırma və prototip hazırlamaq rahatdır | Düzgün model seçimi və verilən keyfiyyəti problemlərini özü həll etmir |
| Hazır optimizasiya mexanizmləri | Real məhsul üçün deploy, monitorinq və təhlükəsizlik ayrıca işdir |

### 2.2. Quraşdırma

Python-un uyğun versiyası və virtual mühit tövsiyə edilir. **CPU üçün sadə başlanğıc**:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

python -m pip install torch notebook matplotlib
jupyter notebook
```

> **Qeyd:** CUDA/GPU və əməliyyat sisteminə uyğun son quraşdırma komandasını [PyTorch-un rəsmi quraşdırma səhifəsindən](https://pytorch.org/get-started/locally/) seçin. Notebook nümunələri **CPU-da və internetsiz** işləyir; əlavə dataset endirilməsinə ehtiyac yoxdur.

### 2.3. `torch.Tensor` nədir?

Tensor — ədədlərin çoxölçülü strukturudur: `scalar` (0D), `vector` (1D), `matrix` (2D), daha yüksək ölçülü massivlər.

```python
import torch

x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
print(x.shape)       # torch.Size([2, 2])
print(x.dtype)       # torch.float32 (bu nümunədə)
print(x.mean())      # tensor(2.5000)
```

Tensorların vacib atributları: `shape`, `dtype`, `device`, `requires_grad`.

- **`shape`:** Ölçülər. `[32, 4]` — batch-də 32 nümunə, hərəsində 4 xüsusiyyət.
- **`dtype`:** Tip; məsələn, `float32`, `int64` (`torch.long`).
- **`device`:** CPU / GPU yerləşməsi. Eyni əməliyyatda tensorlar və model uyğun cihazda olmalıdır.
- **`requires_grad`:** Əməliyyatlar üçün qradiyent izlənməsi aktiv olsunmu?

### 2.4. Autograd: törəmələr avtomatik necə hesablanır?

`autograd` hesablama qrafı qurur, zəncir qaydası (**chain rule**) ilə törəmələri hesablayır.

$$y = x^2 + 3x, \qquad \frac{dy}{dx}=2x+3$$

```python
x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3*x
y.backward()
print(x.grad.item())  # 7.0
```

**Nəyə lazımdır?** Milyonlarla parametr üçün törəmələri əllə yazmaq əvəzinə avtomatik hesablamağa.  
**Autograd olmasa:** Sonlu fərqlər kimi təxmini üsullar və ya əl ilə törəmə lazım olar; bu isə mürəkkəb şəbəkələrdə bahalı və səhvə açıqdır.  
**Mənfi tərəf:** Qradiyent izlənməsi əlavə yaddaş tələb edir. İnferens zamanı `torch.no_grad()` istifadə etmək bu yükü azaldır.

> `backward()` qradiyentləri hesablayır, **çəkiləri dəyişmir**. Parametrləri optimizer yeniləyir. `x.grad` toplanan qiymətdir; təkrar `backward()` çağırışlarında əvvəlki qradiyentin üzərinə əlavə oluna bilər.

### 2.5. `torch.nn.Module` nədir?

`nn.Module` model və qatları təşkil etmək üçün əsas sinifdir. Onun sayəsində parametrlər qeydiyyata alınır, alt qatlar izlənir, `model.parameters()`, `model.train()`, `model.eval()`, `state_dict()` kimi interfeyslər işləyir.

```python
from torch import nn

class SadəŞəbəkə(nn.Module):
    def __init__(self):
        super().__init__()
        self.gizli = nn.Linear(4, 8)
        self.aktivasiya = nn.ReLU()
        self.cixis = nn.Linear(8, 3)

    def forward(self, x):
        x = self.gizli(x)
        x = self.aktivasiya(x)
        return self.cixis(x)  # 3 logit; Softmax YOX

model = SadəŞəbəkə()
x = torch.randn(5, 4)      # 5 nümunə, hərəsi 4 xüsusiyyət
logits = model(x)          # model.__call__ -> forward + hook mexanizmləri
print(logits.shape)        # torch.Size([5, 3])
```

**`forward()` nə iş görür?** Girişin qatlar və hesablamalardan keçərək çıxışa çevrilməsini təyin edir. Praktikada `model(x)` çağırılır; `model.forward(x)` çağırışı adətən tövsiyə edilmir, çünki `__call__` mexanizminin əlavə davranışlarını yan keçə bilər.

**`nn.Module` olmasa:** Sadə tensor əməliyyatları ilə də model yazmaq olar, lakin parametrlərin idarə edilməsi, saxlama, alt modullar və təlim/eval rejimləri çətinləşər.

### 2.6. `nn.Linear` nədir?

Xətti (tam əlaqəli) qat:

$$Y = XW^T+b$$

`nn.Linear(in_features, out_features)` qatında:

- Çəki matrisinin forması: `(out_features, in_features)`;
- Bias vektorunun forması: `(out_features,)`;
- Parametr sayı: `in_features × out_features + out_features` (**bias aktivdirsə**).

**Nümunə:** `nn.Linear(4, 8)` → `4×8 + 8 = 40` parametr; `nn.Linear(8, 3)` → `8×3 + 3 = 27`; ümumilikdə **67 öyrənilən parametr**.

```python
qat = nn.Linear(4, 8)
print(qat.weight.shape)  # [8, 4]
print(qat.bias.shape)    # [8]
```

**Üstünlük:** Sadə, sürətli, başa düşülən, vektorlaşdırılmış hesablamadır.  
**Məhdudiyyət:** Təkbaşına qeyri-xətti asılılıqları model edə bilmir; çox böyük girişlərdə parametr sayı sürətlə böyüyür. Şəkillər üçün CNN, ardıcıllıqlar üçün başqa arxitekturalar daha münasib ola bilər.

### 2.7. Aktivasiya funksiyaları: ReLU niyə lazımdır?

$$\mathrm{ReLU}(z)=\max(0,z)$$

| Funksiya | Çıxış aralığı | Tipik istifadə | Məhdudiyyət |
|---|---|---|---|
| `ReLU` | $[0,\infty)$ | Gizli qatlar | Mənfi tərəfdə qradiyent sıfır ola bilər ("dead ReLU") |
| `Sigmoid` | $(0,1)$ | İkili/multi-label inferens ehtimalları | Doyma sahəsində kiçik qradiyent |
| `Tanh` | $(-1,1)$ | Bəzi gizli qatlar/RNN dizaynları | Doyma və vanishing gradient |
| `Softmax` | Hər sinif üçün $(0,1)$, cəm 1 | Qarşılıqlı istisna olunan sinif ehtimalları | Multi-label üçün uyğun deyil |

> **Çox vacib:** Çoxsinifli təsnifatda `nn.CrossEntropyLoss()` istifadə ediləcəksə, son xətti qatın ardınca təlim məqsədilə `Softmax` qoymayın. Loss funksiyası xam **logitləri** gözləyir.

### 2.8. `model.train()` və `model.eval()` nəyi dəyişir?

- `model.train()`: Dropout və BatchNorm kimi qatları **təlim rejiminə** keçirir.
- `model.eval()`: Onları **qiymətləndirmə rejiminə** keçirir.
- **`model.eval()` qradiyentləri söndürmür.** Bunun üçün `with torch.no_grad():` (və ya uyğun hallarda `torch.inference_mode()`) istifadə edilir.
- Sadə `Linear`+`ReLU` şəbəkəsində bu rejimlər eyni hesablamanı verə bilər, amma düzgün təcrübə kimi istifadə olunmalıdır.

---

## 3. Loss Functions — itki funksiyaları

### 3.1. Loss function nədir və nə üçün lazımdır?

**Loss (itki/səhv funksiyası)** modelin proqnozu ilə hədəf dəyərin arasındakı uyğunsuzluğu bir skalyar ədədlə ifadə edir. Təlim zamanı optimizer bu itkini azaltmaq üçün qradiyentlərdən istifadə edir.

$$L = \mathrm{Loss}(\hat{y},y)$$

Burada $\hat y$ — proqnoz, $y$ — düzgün hədəfdir.

**Loss olmasa:** Modelin hansı nəticəsinin daha yaxşı olduğunu riyazi şəkildə təyin etmək və standart qradiyent əsaslı təlim aparmaq mümkün olmaz. Alternativ öyrənmə məqsədləri qurmaq mümkündür, lakin hər halda optimallaşdırılan bir obyektiv lazımdır.

**Harada istifadə olunur?** Reqressiyada qiymət proqnozu, təsnifatda obyekt tanıma, seqmentasiya, mətn modeli təlimi və s.

**Üstünlük:** Öyrənməyə ölçülə bilən istiqamət verir.  
**Mənfi tərəf:** Səhv loss seçimi modeli yanlış məqsədə optimallaşdıra bilər. Aşağı loss hər zaman real dünyada yaxşı nəticə, düzgün kalibrlənmiş ehtimal və ya ədalətli qərar demək deyil.

### 3.2. `MSELoss` — Mean Squared Error

Reqressiya üçün yayılmış itki funksiyası:

$$\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2$$

Əl ilə misal: real dəyərlər `[2,4,6]`, proqnozlar `[3,5,4]`:

$$\mathrm{MSE}=\frac{(3-2)^2+(5-4)^2+(4-6)^2}{3}=\frac{6}{3}=2$$

```python
prediction = torch.tensor([3.0, 5.0, 4.0])
target = torch.tensor([2.0, 4.0, 6.0])
loss = nn.MSELoss()(prediction, target)
print(loss.item())  # 2.0
```

**Harada?** Temperatur, qiymət, məsafə və digər ədədi proqnozlar.  
**Niyə?** Böyük xətaları kvadrat səbəbilə daha sərt cəzalandırır; diferensiallanması rahatdır.  
**Üstünlük:** Sadə və hamar riyazi forma.  
**Mənfi tərəf:** Kənar/outlier dəyərlərə çox həssasdır; təsnifat üçün standart seçim deyil.  
**Olmasa:** MAE və Huber kimi alternativlər var — reqressiya üçün məhz MSE məcburi deyil.

### 3.3. MAE (`L1Loss`) və Huber (`SmoothL1Loss`)

$$\mathrm{MAE}=\frac{1}{N}\sum_i |\hat y_i-y_i|$$

- **MAE:** Kənar dəyərlərin təsirini MSE-dən az böyüdür; sıfırda hamarlığı yoxdur.
- **Huber / SmoothL1:** Kiçik səhvlərdə kvadratik, böyük səhvlərdə xətti xarakterli kompromis; PyTorch-da bu iki funksiyanın parametr və miqyas davranışı tam eyni deyil.

```python
pred = torch.tensor([2.0, 10.0, 3.0])
true = torch.tensor([2.0,  2.0, 4.0])
print('MSE:', nn.MSELoss()(pred, true).item())
print('MAE:', nn.L1Loss()(pred, true).item())
print('SmoothL1:', nn.SmoothL1Loss()(pred, true).item())
```

### 3.4. Binary Cross-Entropy — iki sinif üçün

İki sinifli (məsələn, spam/deyil) tapşırıq üçün:

$$\mathrm{BCE} = -\big[y\ln(p)+(1-y)\ln(1-p)\big]$$

`p` — müsbət sinif ehtimalı, `y ∈ {0,1}`.

| Modelin çıxışı | Tövsiyə olunan loss | Niyə? |
|---|---|---|
| **Xam logit** | `nn.BCEWithLogitsLoss()` | `Sigmoid` + BCE-ni ədədi sabitliklə birləşdirir |
| Artıq `Sigmoid`-dən keçmiş ehtimal | `nn.BCELoss()` | `BCELoss` ehtimal gözləyir, amma logit əsaslı variant çox vaxt daha əlverişlidir |

```python
logits = torch.tensor([[1.2], [-0.5], [0.7]])
labels = torch.tensor([[1.0], [0.0], [1.0]])
loss = nn.BCEWithLogitsLoss()(logits, labels)
probabilities = torch.sigmoid(logits)
print('Loss:', round(loss.item(), 4))
print('Ehtimallar:', probabilities.squeeze().tolist())
```

**Mühüm:** BCE-də hədəflər adətən `float` tipində və çıxışla eyni ölçüdə olur. Bir logitlik ikili model üçün `(N, 1)` forması uyğundur.

**Üstünlük:** İkili ehtimal proqnozları üçün düzgün məqsəd.  
**Mənfi tərəf:** Sinif balanssızlığı zamanı azlıq sinfini nəzərə almaq üçün `pos_weight`, nümunə çəkiləri, uyğun metriklər və ya məlumat strategiyası tələb oluna bilər.  
**Olmasa:** Birbaşa `accuracy` kimi diskret metriki diferensiallaşdırıb standart təlim aparmaq mümkün deyil; diferensiallana bilən məqsədə ehtiyac var.

### 3.5. Cross-Entropy Loss — çoxsinifli təsnifat

Üç qarşılıqlı istisna olunan sinif: məsələn, `0=ev sahibi qalib`, `1=heç-heçə`, `2=qonaq qalib`.

$$\mathrm{CE}(y,p)=-\ln(p_{y})$$

Doğru sinfin ehtimalı $p_y$ nə qədər kiçik olsa, itki o qədər böyükdür.

```python
logits = torch.tensor([[2.0, 1.0, 0.1], [0.2, 0.3, 1.9]])  # [2, 3]
target = torch.tensor([0, 2], dtype=torch.long)           # [2]
loss = nn.CrossEntropyLoss()(logits, target)
print('Cross-entropy:', round(loss.item(), 4))
```

**Burada `target` bir ölçülü sinif indeksləridir.** PyTorch `CrossEntropyLoss` uyğun formada **sinif ehtimal paylanmaları** ilə də işləyə bilir, lakin bu materialdakı tipik və daha səmərəli nümunələrdə `long` indekslərdən istifadə edilir.

**Üstünlük:** Çoxsinifli təsnifata uyğundur; düzgün sinfə yüksək pay ayırmağı öyrədir.  
**Mənfi tərəf:** Səhv etiketlərə və həddindən artıq əmin, yanlış proqnozlara həssasdır; sinif balanssızlığı problem yarada bilər.  
**Olmasa:** Başqa loss seçmək mümkündür, amma təsnifat üçün diferensiallana bilən, uyğun öyrənmə obyektivi lazımdır.

### 3.6. Loss və metric eyni şey deyil

| Anlayış | Məqsəd | Nümunə |
|---|---|---|
| **Loss** | Təlimə qradiyent vermək | MSE, BCEWithLogits, CrossEntropy |
| **Metric** | Nəticənin keyfiyyətini qiymətləndirmək | Accuracy, Precision, Recall, F1, MAE |

Məsələn, iki model eyni `accuracy` ala bilər, lakin biri yanlış cavablarda hədsiz əmin olduğu üçün `CrossEntropyLoss` dəyəri daha yüksək ola bilər. Buna görə bir neçə metrik izləmək faydalıdır.

### 3.7. `reduction`, sinif çəkiləri və label smoothing

- **`reduction='mean'`** — itkiləri orta qiymətə gətirir (standart seçimdir).
- **`reduction='sum'`** — itkiləri toplayır; batch ölçüsü nəticəyə daha çox təsir edir.
- **`reduction='none'`** — hər element/nümunə üçün ayrı itki qaytarır.
- **`weight`** (`CrossEntropyLoss`) — siniflərin vacibliyini fərqləndirə bilər.
- **`pos_weight`** (`BCEWithLogitsLoss`) — pozitiv hədəflərin loss-dakı çəkisini idarə edir.
- **`label_smoothing`** — hədəflərin hədsiz sərt paylanmasını yumşalda bilər; hər tapşırıqda avtomatik faydalı deyil.

```python
loss_per_item = nn.CrossEntropyLoss(reduction='none')(logits, target)
print(loss_per_item)  # Hər nümunənin ayrıca itkisi
```

**Diqqət:** Çəkili loss istifadə etdikdə “loss aşağı düşdü” nəticəsini əvvəlki çəkisiz loss ilə birbaşa müqayisə etmək düzgün deyil: məqsəd funksiyası dəyişib. Eyni validation metriki ilə yoxlayın.

---
## 4. Softmax və Cross-Entropy

### 4.1. Softmax nədir?

**Softmax** hər hansı real ədədlər vektorunu, bütün elementləri müsbət və cəmi 1 olan ehtimal paylanmasına çevirir:

$$\operatorname{Softmax}(z_i)=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}$$

Burada $C$ sinif sayıdır; $z_i$ isə $i$-ci sinfin **logit** dəyəridir. Logitlər **ehtimal deyil**: mənfi, müsbət və 1-dən böyük ola bilərlər.

Misal: `logits=[2.0,1.0,0.1]` üçün Softmax təxminən `[0.659, 0.242, 0.099]` qaytarır. Cəm **1.0** olur.

```python
import torch.nn.functional as F

scores = torch.tensor([[2.0, 1.0, 0.1],
                       [0.1, 0.2, 2.0]])
probs = F.softmax(scores, dim=1)  # 1-ci ölçü sinif sütunlarıdır
print(probs)
print(probs.sum(dim=1))  # tensor([1., 1.])
print(probs.argmax(dim=1))  # tensor([0, 2])
```

**Nə üçün lazımdır?** Bir-birini istisna edən siniflər arasında normallaşdırılmış seçim paylanması yaratmaq üçün.  
**Harada?** Şəkil təsnifatı, mətnin hansı mövzuya aid olması, bir oyunun üç mümkün nəticəsi və s.  
**Üstünlüklər:** Hesablanması sadədir, diferensiallana bilir, siniflər üzrə ehtimal cəmi 1-dir.  
**Mənfi tərəflər:** Bir sinfin payı artdıqda digərlərinin payı azalır; bu, siniflərin eyni anda doğru ola bildiyi məsələlərə uyğun deyil. Həmçinin yüksək Softmax ehtimalı **mütləq etibarlı/kalibrlənmiş əminlik** demək deyil.

**Softmax olmasa:** Çıxışdakı ən böyük logiti seçərək (`argmax`) yenə sinif müəyyən etmək olar; sıra dəyişmir. Amma istifadəçi üçün normallaşdırılmış paylanma əldə etmərik. Təlimdə `CrossEntropyLoss` artıq Softmax-ın riyazi mexanizmini əhatə etdiyindən ayrıca Softmax lazım deyil.

### 4.2. Niyə `dim=1` yazırıq?

Standart çoxsinifli batch formatı `[batch_size, num_classes]`-dir. `dim=1` **hər sətirdə siniflər üzrə** normallaşdırma aparır.

```python
batch = torch.randn(4, 3)
correct = torch.softmax(batch, dim=1)
assert torch.allclose(correct.sum(dim=1), torch.ones(4), atol=1e-6)
```

`dim=0` yazsaq, nümunələrin öz aralarında normallaşdırılmasına səbəb olarıq — bu, təsnifat modelinin məqsədi deyil.

### 4.3. Cross-Entropy ilə necə bağlıdır?

Hədəf sinif `k` olduqda:

$$L=-\log\big(\mathrm{Softmax}(z)_k\big)$$

Yəni praktik olaraq:

$$\boxed{\mathrm{CrossEntropyLoss}(z,y)=\mathrm{NLLLoss}(\mathrm{log\_softmax}(z),y)}$$

`log_softmax` / `CrossEntropyLoss` kimi birləşdirilmiş hesablamalar **ədədi sabitliyi** artırır. Böyük logitlər üzərində birbaşa `exp(z)` hesablamaq daşma (overflow) verə bilər.

```python
logits = torch.tensor([[2.0, 1.0, 0.1]])
y = torch.tensor([0])

built_in = nn.CrossEntropyLoss()(logits, y)
manual = -torch.log_softmax(logits, dim=1)[0, y[0]]
print(built_in.item(), manual.item())
assert torch.allclose(built_in, manual)
```

### 4.4. Ən mühüm səhv: iki dəfə Softmax

```python
# ❌ YANLIŞ: CrossEntropyLoss xam logit gözləyir
wrong_loss = nn.CrossEntropyLoss()(torch.softmax(logits, dim=1), y)

# ✅ DOĞRU
correct_loss = nn.CrossEntropyLoss()(logits, y)

# ✅ İNFERENS ZAMANI: ehtimal lazımdırsa
probability = torch.softmax(logits, dim=1)
```

İkiqat Softmax **ümumiyyətlə proqramı xətaya salmaya bilər**, buna görə təhlükəlidir: başqa bir optimallaşdırma məqsədi yaranır, səhv qradiyent siqnalı formalaşa və təlimin keyfiyyəti zəifləyə bilər.

### 4.5. Sigmoid yoxsa Softmax?

| Problem | Siniflər | Çıxış | Təlimdə loss | İnferens |
|---|---|---|---|---|
| **Binary** | İki mümkün cavabdan biri | Bir logit `(N,1)` | `BCEWithLogitsLoss` | `sigmoid(logit)` |
| **Multiclass** | `C` sinifdən **yalnız biri** | `C` logit `(N,C)` | `CrossEntropyLoss` | `softmax(logits, dim=1)` |
| **Multilabel** | Eyni anda bir neçə etiket ola bilər | Hər etiketə bir logit `(N,C)` | `BCEWithLogitsLoss` | Hər logitə ayrıca `sigmoid` |
| **Regression** | Ədədi dəyər | Bir və ya çox real ədəd | `MSELoss`, `L1Loss` və s. | Xam ədədi proqnoz |

**Misal:** Şəkildə eyni anda “avtomobil” və “insan” etiketləri ola bilər → multi-label → Sigmoid. Şəkil “pişik/it/quş” kateqoriyalarından **mütləq yalnız biridir** → multiclass → Softmax.

### 4.6. Numerik sabitlik təcrübəsi

```python
big_logits = torch.tensor([[1000.0, 1001.0, 999.0]])

# Riskli üsul: exp(1000) sonsuzluğa çevrilə bilər
naive = torch.exp(big_logits) / torch.exp(big_logits).sum(dim=1, keepdim=True)

# Sabit üsul
stable = torch.softmax(big_logits, dim=1)
print('Sadəlövh hesab:', naive)    # çox vaxt nan
print('Sabit hesab:', stable)     # sonlu ehtimallar
```

Sabit Softmax düsturu ən böyük elementi çıxmağa əsaslana bilər:

$$\mathrm{Softmax}(z_i)=\frac{e^{z_i-m}}{\sum_j e^{z_j-m}}, \quad m=\max(z)$$

Bu, ehtimalları dəyişdirmir, amma daşma riskini azaldır.

---

## 5. Optimizers — parametrlərin yenilənməsi

### 5.1. Optimizer nədir?

**Optimizer** — loss-a görə hesablanmış qradiyentləri istifadə edərək modelin öyrənilən parametrlərini yeniləyən alqoritmdir.

**Loss bildirir:** “Nə qədər və hansı istiqamətdə səhv edirik?”  
**`backward()` hesablayır:** “Bu səhv hər parametrdən nə qədər asılıdır?”  
**Optimizer edir:** “Parametrləri növbəti addım üçün necə dəyişək?”

**Optimizer olmasa:** `loss.backward()` nə qədər çağırılsa da, standart təlimdə parametrlər avtomatik yenilənməz; model eyni çəkilərlə qalacaq. Alternativ olaraq çəkiləri özümüz yeniləyə bilərik, amma bu halda optimizer mexanizmini əl ilə qurmuş olarıq.

### 5.2. Gradient Descent nədir?

Məqsəd loss funksiyasını azaltmaqdır. Bir parametr üçün əsas qayda:

$$\theta_{t+1}=\theta_t-\eta\frac{\partial L}{\partial\theta_t}$$

- $\theta$ — parametr;
- $\eta$ — **learning rate** (öyrənmə sürəti);
- $\partial L/\partial\theta$ — loss-un parametr üzrə törəməsi.

**Kiçik nümunə:** `L(w)=(w-3)²`, başlanğıc `w=0`, `lr=0.1`:

$$\frac{dL}{dw}=2(w-3)=-6$$

$$w_{new}=0-0.1(-6)=0.6$$

İtki `9`-dan `(0.6-3)²=5.76`-ya düşür.

```python
w = torch.tensor(0.0, requires_grad=True)
optimizer = torch.optim.SGD([w], lr=0.1)
for _ in range(5):
    optimizer.zero_grad()
    loss = (w - 3) ** 2
    loss.backward()
    optimizer.step()
    print(f'w={w.item():.4f}, loss={loss.item():.4f}')
```

### 5.3. SGD (Stochastic Gradient Descent)

SGD mini-batch üzrə hesablanmış qradiyentdən istifadə edərək sadə parametr yeniləməsi edir. `momentum` daxil edildikdə əvvəlki yeniləmə istiqamətlərindən də yararlanır.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
```

**Harada?** Şəkil modelləri və müxtəlif ML təcrübələri daxil olmaqla çox sayda tapşırıqda.  
**Üstünlüklər:** Sadədir, az əlavə yaddaş tələb edir; uyğun tənzimləmələrlə yaxşı ümumiləşdirmə verə bilər.  
**Mənfi tərəflər:** Learning rate və schedule seçiminə həssasdır; bəzi problemlərdə başlanğıcda yavaş öyrənə bilər.  
**Olmasa:** Adam/AdamW və digər optimizatorlar mövcuddur; SGD yeganə seçim deyil.

### 5.4. Adam (Adaptive Moment Estimation)

Adam qradiyentin **birinci momentini** (təqribi hərəkətli ortalama) və **ikinci momentini** (kvadratlarının hərəkətli ortalaması) izləyir; parametr üzrə adaptiv addımlar qurur.

Sadələşdirilmiş forma:

$$m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$$
$$v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$$
$$\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$$

Burada $\hat m_t$ və $\hat v_t$ başlanğıc qərəzinə görə korreksiya edilmiş qiymətlərdir.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

**Üstünlüklər:** Çox vaxt yaxşı başlanğıc seçimidir; hər parametr üzrə adaptiv sürət, səs-küylü/sey­rək qradiyentlərdə praktik rahatlıq.  
**Mənfi tərəflər:** SGD-dən daha çox optimizer vəziyyəti yaddaşı saxlayır; bəzi tapşırıqlarda ümumiləşdirmə və son nəticə baxımından SGD üstün ola bilər.  
**Olmasa:** SGD və digər adaptiv üsullar işləyə bilər; məqsəd problemə uyğun optimizator seçməkdir.

**AdamW nəyə görə ayrıca qeyd olunur?** AdamW `weight_decay`-i gradient əsaslı yeniləmədən **ayrılmış** (decoupled) şəkildə tətbiq edir. Müasir dərin öyrənmə modellərində tez-tez seçilir. Amma `weight_decay` hər tapşırıqda mütləq lazım deyil.

### 5.5. SGD və Adam müqayisəsi

| Meyar | SGD | Adam |
|---|---|---|
| Yeniləmə | Əsasən eyni LR əsasında | Adaptiv, parametrə görə fərqli |
| Yaddaş | Adətən daha az; momentum artırır | İki moment vektoru saxlayır |
| İlk addımlar | Bəzən daha həssas/yavaş | Çox vaxt sürətli başlanğıc |
| Tənzimləmə | LR/momentum/schedule əhəmiyyətlidir | LR və `betas` əhəmiyyətlidir |
| Ümumi qalib | **Yoxdur** | **Yoxdur** |

> **Doğru müqayisə:** Eyni dataset, eyni split, eyni arxitektura, eyni başlanğıc çəkilər, uyğun ayrılıqda tuning, bir neçə toxum (`seed`), validation metrikləri və hesablamaya sərf olunan vaxt nəzərə alınmalıdır. Tək run ilə “Adam həmişə daha yaxşıdır” demək olmaz.

### 5.6. Learning Rate (LR) nədir?

Learning rate optimizatorun **addım ölçüsünü** idarə edir:

- **Çox kiçik:** Model yavaş öyrənə və praktik vaxtda yaxşı nəticəyə çatmaya bilər.
- **Çox böyük:** Loss dalğalana, arta, hətta `nan` ola bilər.
- **Uyğun:** Təlim sabit gedər və validation metrikləri yaxşılaşar.

Başlanğıc təcrübəsi kimi `Adam(lr=1e-3)`, `SGD(lr=1e-2 .. 1e-1)` sınamaq olar; bunlar **universal qayda deyil**. Model, batch ölçüsü, normalizasiya və tapşırıq üzrə dəyişir.

**LR scheduler** — learning rate-in təlim boyu dəyişdirilməsi üsulu:

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(20):
    # burada train loop işləyir
    scheduler.step()  # bir epoch üçün qurulmuş StepLR nümunəsi
```

> Scheduler-ın `step()` çağırışının vaxtı istifadə olunan scheduler-dan asılıdır. Bu nümunədə **optimizer yeniləmələrindən sonra** epoch sonunda çağırılmalıdır. `ReduceLROnPlateau` kimi scheduler isə validation ölçüsünü qəbul edir.

### 5.7. `zero_grad()`, `backward()`, `step()` — düzgün sıra

```python
model.train()
for inputs, labels in train_loader:
    optimizer.zero_grad()         # (1) əvvəlki gradientləri təmizlə
    logits = model(inputs)        # (2) forward
    loss = criterion(logits, labels)  # (3) xətanı hesabla
    loss.backward()              # (4) gradientləri hesabla
    optimizer.step()             # (5) çəkiləri yenilə
```

| Funksiya | Nə edir? | Çağırmasaq? |
|---|---|---|
| `optimizer.zero_grad()` | Parametrlərin əvvəlki qradiyentlərini təmizləyir | Qradiyentlər toplanaraq istənməyən böyüklüyə çata bilər; qəsdən accumulation ayrıca texnikadır |
| `model(inputs)` | `forward` hesablamasını aparır | Proqnoz və loss üçün giriş çıxışı yaranmaz |
| `criterion(...)` | Səhvi ölçür | Öyrənmə məqsədi müəyyən edilməz |
| `loss.backward()` | Qradiyentləri hesablayır | Optimizer yeni loss-a uyğun qradiyent almaz |
| `optimizer.step()` | Parametrləri yeniləyir | Model çəkiləri dəyişməz |

**`zero_grad(set_to_none=True)` barədə:** Qradiyenti sıfır tensoru əvəzinə `None` etməyi seçmək yaddaş/performans üçün faydalı ola bilər. Optimizer bəzi hallarda `None` qradiyenti olan parametri tam atlayır, sıfır qradiyent üçün isə addım məntiqi fərqli ola bilər. Bu materialda standart `zero_grad()` kifayətdir.

### 5.8. Daha hansı problemlər ola bilər?

- **Vanishing gradients:** Dərin şəbəkədə qradiyentlər həddən artıq kiçilir; öyrənmə zəifləyir.
- **Exploding gradients:** Qradiyentlər hədsiz böyüyür; addımlar qeyri-sabit olur. Lazım gələrsə `torch.nn.utils.clip_grad_norm_` tətbiq edilə bilər.
- **Overfitting:** Train loss azalır, validation nəticəsi pisləşir; model təlim nümunələrinə həddən artıq uyğunlaşır.
- **Underfitting:** Model kifayət qədər öyrənə bilmir; həm train, həm validation nəticələri zəifdir.
- **Weight decay:** Çəkilərin çox böyüməsinin qarşısını almağa kömək edə bilən tənzimləyici üsul; təsiri optimizatorun tətbiqindən asılıdır.
- **Gradient accumulation:** Yaddaşa sığmayan böyük batch-i bir neçə kiçik batch üzrə qradiyent toplayaraq təqlid etməyə çalışır. Bu halda loss miqyasını düzgün tənzimləmək vacibdir.

---

## 6. Başdan sona praktiki model

### Məqsəd

İnternetsiz, kənardan dataset yükləmədən **iki xüsusiyyəti olan üç sinifli süni verilənlər** yaradıb PyTorch ilə təsnifat modeli öyrədəcəyik. Bu nümunədə bütün anlayışlar birləşir:

```text
X, y  →  nn.Linear → ReLU → nn.Linear → logits
                           │
                 CrossEntropyLoss(logits, y)
                           │
                       backward()
                           │
                       Adam.step()
                           │
           qiymətləndirmə: accuracy / Softmax
```

### 6.1. Verilənlərin yaradılması və train/validation bölgüsü

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(42)
centers = torch.tensor([[-2.0, -2.0], [2.0, -2.0], [0.0, 2.0]])
X = torch.cat([center + 0.7 * torch.randn(240, 2) for center in centers])
y = torch.cat([torch.full((240,), i, dtype=torch.long) for i in range(3)])

permutation = torch.randperm(len(X))
train_idx = permutation[:576]  # 80% (576/720)
valid_idx = permutation[576:]  # 20% (144/720)

train_loader = DataLoader(TensorDataset(X[train_idx], y[train_idx]),
                          batch_size=32, shuffle=True)
X_valid, y_valid = X[valid_idx], y[valid_idx]
print('Train:', len(train_idx), 'Validation:', len(valid_idx))
```

Verilənlərimiz demonstrasiya üçündür; real məsələ üçün məlumatın keyfiyyəti, etiket doğruluğu, məxfilik, qərəz, balans, normalizasiya və düzgün split ayrıca qiymətləndirilməlidir.

### 6.2. Model, loss və optimizer

```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.net(x)  # Logits!

model = Classifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
```

### 6.3. Təlim dövrəsi

```python
for epoch in range(40):
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)

    model.eval()
    with torch.no_grad():
        val_logits = model(X_valid)
        val_loss = criterion(val_logits, y_valid).item()
        val_acc = (val_logits.argmax(dim=1) == y_valid).float().mean().item()

    if epoch in (0, 9, 19, 39):
        print(f'Epoch {epoch+1:2d} | train loss={total_loss/len(train_idx):.4f}'
              f' | val loss={val_loss:.4f} | val accuracy={val_acc:.1%}')
```

**Nəyə fikir verin:** `CrossEntropyLoss`-a **logits** veririk, `labels` `long`-dur, batch boyu itki `len(xb)` ilə çəkilib orta hesablanır; validation-da `eval()` və `no_grad()` aktivdir.

### 6.4. Yeni nümunəyə proqnoz

```python
new_sample = torch.tensor([[0.0, 2.5]])
model.eval()
with torch.no_grad():
    new_logits = model(new_sample)
    new_probs = torch.softmax(new_logits, dim=1)
    predicted_class = int(new_probs.argmax(dim=1).item())

print('Ehtimallar:', new_probs[0].tolist())
print('Proqnoz sinif:', predicted_class)
```

### 6.5. Modeli saxlamaq və yükləmək

```python
# Saxlama:
torch.save(model.state_dict(), 'classifier_state.pt')

# Daha sonra eyni arxitekturada yükləmə:
loaded = Classifier()
loaded.load_state_dict(torch.load('classifier_state.pt', map_location='cpu', weights_only=True))
loaded.eval()
```

> **Təhlükəsizlik:** Tanımadığınız mənbədən gələn model fayllarını ehtiyatsız yükləməyin. Bu nümunədə yalnız `state_dict` saxlanılır və `weights_only=True` istifadə edilir. Tam təlimi davam etdirmək üçün adətən optimizer vəziyyəti, epoch, seed və konfiqurasiya da ayrıca saxlanır.

**Tam interaktiv variant:** Yanındakı `neyron_sebekeleri_praktika.ipynb` faylını Jupyter/VS Code/GitHub-da açın. Notebook-da ayrıca **SGD vs Adam** təcrübəsi, qradiyentlərin izlənməsi və mini-testlər var.

---

## 7. Real tətbiq ssenariləri və seçim qaydaları

### 7.1. Futbol nəticəsinin proqnozu — nümunə

Məsələn, oyundan **əvvəl** məlum olan xüsusiyyətlərlə 3 mümkün nəticəni proqnozlaşdıraq:

- Giriş: ev/qonaq komandalarının əvvəlki oyunlarda topladığı xallar, əvvəlki matçlarda vurulan/buraxılan qollar, əvvəlcədən məlum reytinq fərqi, istirahət günü və s.
- Çıxış: `0=ev qələbəsi`, `1=heç-heçə`, `2=qonaq qələbəsi`.
- Model: çoxsinifli neyron şəbəkəsi; son qat **3 logit**.
- Təlim: `nn.CrossEntropyLoss(logits, labels)`.
- İnferens: `torch.softmax(logits, dim=1)` → ehtimal paylanması.
- Qiymətləndirmə: Accuracy ilə yanaşı log loss (CE), kalibrlənmə və siniflər üzrə metriklər.

**Niyə ehtiyatlı olmalıyıq?** Futbol nəticələri qeyri-müəyyəndir; keçmiş göstəricilər nəticəni zəmanətləmir. Vaxt üzrə bölünmə, güc səviyyələrinin dəyişməsi, verilən sızması, bazis modellərlə müqayisə və ehtimalların kalibrlənməsi vacibdir. Bu konsept **Flashscore tipli canlı skor sisteminin özü deyil**; ehtimal proqnozlaşdıran ayrıca ML moduludur.

### 7.2. Tapşırığa uyğun seçim

| Tapşırıq | Son qat | Loss | İnferens addımı | Nə səhv gedə bilər? |
|---|---|---|---|---|
| Evin qiyməti | `Linear(...,1)` | MSE / Huber | Bir ədəd | Outlier MSE-ni şişirdir |
| Spam / deyil | `Linear(...,1)` | BCEWithLogits | Sigmoid + hədd | Balanssız siniflər accuracy-ni aldadır |
| Üç futbol nəticəsi | `Linear(...,3)` | CrossEntropy | Softmax / argmax | Yanlış `dim`, ikiqat Softmax |
| Şəkildə bir neçə obyekt | `Linear(...,C)` | BCEWithLogits | Ayrı-ayrı Sigmoid + hədd | Softmax bir neçə doğru etiketi sıxışdırır |
| Ədədi zaman sırası | Problemə görə | MSE / Huber və s. | Real ədəd(­lər) | Gələcək məlumatın train-ə sızması |

### 7.3. Model nə zaman uyğun seçim olmaya bilər?

Verilənlər çox azdırsa, problem qaydalarla dəqiq həll olunursa, izaholunma tələbi çox yüksəkdirsə və ya təlim/inferens xərci böyükdürsə, sadə statistik metodlar, qayda əsaslı yanaşma, xətti/loqistik reqressiya və ya ağac əsaslı modellər daha münasib ola bilər. **Neyron şəbəkəsi istifadə etmək məqsəd deyil; düzgün və səmərəli həll tapmaq məqsəddir.**

---

## 8. Ən çox rast gəlinən səhvlər

| Problem | Əlamət | Səbəb | Düzəliş |
|---|---|---|---|
| `CrossEntropyLoss`-dan əvvəl Softmax | Gözlənilməz təlim davranışı | Logit əvəzinə ehtimal ötürülür | Son qatı logit kimi saxlayın |
| `target` yanlış tipdə | `expected Long` tipli xəta | Sinif indeksləri `float` olur | `target.long()`; `(N,)` forma |
| BCE-də ölçü uyğunsuzluğu | `size mismatch` | `(N,1)` vs `(N,)` | Hədəf və çıxış formalarını eyniləşdirin |
| `optimizer.step()` yoxdur | Loss sabit qalır | Çəkilər yenilənmir | `backward()`-dan sonra `step()` |
| `zero_grad()` unudulub | Qeyri-stabil təlim | Qradiyent yığılır | Batch başında təmizləyin |
| `loss.backward()` yoxdur | Çəkilər öyrənmir | Yeni qradiyent yoxdur | Loss üçün `backward()` çağırın |
| Yanlış `dim` | Softmax cəmləri 1 deyil | Yanlış ox üzrə normalizasiya | `[N,C]` üçün `dim=1` |
| Çox böyük LR | Loss artar və ya `nan` olar | Addımlar həddən artıq böyükdür | LR azaldın, gradientləri izləyin |
| `model.eval()` unudulub | Eval nəticəsi dəyişkəndir | Dropout/BatchNorm train rejimində | Eval öncəsi `model.eval()` |
| `torch.no_grad()` yoxdur | Artıq yaddaş sərfi | Eval zamanı qraf qurulur | Eval/inferens üçün `no_grad()` |
| Data leakage | Şişirdilmiş metric | Test/gələcək məlumatı öyrənməyə qarışır | Düzgün split və pipeline |
| Yalnız train loss izlənir | Overfitting gizlənir | Validation yoxdur | Ayrı validation metrikləri izləyin |

**Problemi araşdırmaq üçün qısa kontrol siyahısı:** `shape`, `dtype`, `device`, etiket aralığı, NaN/Inf, dəyişən qradiyent və çəki qiymətləri, LR, train/validation fərqi, sinif balansı və təsadüfi seed yoxlanmalıdır.

---

## 9. Müqayisə cədvəlləri və yaddaş kartı

### 9.1. Dörd mövzunun bir-biri ilə əlaqəsi

| Hissə | Əsas sual | Əsas API | Onsuz nə baş verir? |
|---|---|---|---|
| **PyTorch / NN** | Proqnozu necə hesablayaq? | `nn.Module`, `nn.Linear`, `forward()` | Modeli başqa vasitə ilə qurmaq lazım gəlir |
| **Loss** | Nə qədər səhv etmişik? | `nn.CrossEntropyLoss`, `nn.MSELoss` | Optimallaşdırılan məqsəd olmaz |
| **Softmax** | Sinifləri ehtimala necə çevirək? | `torch.softmax` | `argmax` mümkündür, ehtimal paylanması alınmaz |
| **Optimizer** | Çəkiləri necə yeniləyək? | `torch.optim.SGD`, `torch.optim.Adam` | Standart təlimdə çəkilər öz-özünə dəyişməz |

### 9.2. Əzbərləmək üçün 10 qayda

1. **NN** — öyrənilən çəkili riyazi funksiyadır.
2. **`nn.Module`** — modeli/qatları sistemli təşkil edir.
3. **`nn.Linear`** — $xW^T+b$ hesablayır.
4. **`forward()`** — proqnoza aparan hesablamanı müəyyən edir.
5. **Autograd** — törəmələri avtomatik hesablayır.
6. **Loss** — optimallaşdırılan səhvi ölçür.
7. **Softmax** — bir-birini istisna edən sinif logitlərini paylanmaya çevirir.
8. **CrossEntropyLoss-a xam logit verilir**, əvvəlcədən Softmax yox.
9. **`backward()` qradiyent, `optimizer.step()` parametr yeniləməsidir.**
10. **Validation/test** — yalnız train loss-un deyil, real ümumiləşmənin ölçülməsi üçündür.

### 9.3. Kod şablonu

```python
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

model.train()
for xb, yb in train_loader:
    optimizer.zero_grad()
    logits = model(xb)
    loss = criterion(logits, yb)
    loss.backward()
    optimizer.step()

model.eval()
with torch.no_grad():
    logits = model(X_test)
    probs = torch.softmax(logits, dim=1)
    predictions = probs.argmax(dim=1)
```

---

## 10. Yoxlama sualları və praktiki tapşırıqlar

### 10.1. Özünü yoxla

<details>
<summary><strong>1) PyTorch nədir və yalnız GPU-da işləyirmi?</strong></summary>

PyTorch tensor əməliyyatları, autograd və NN təlimi üçün kitabxanadır. **Xeyr**, CPU-da da işləyir.
</details>

<details>
<summary><strong>2) `nn.Module` ilə `nn.Linear` fərqi nədir?</strong></summary>

`nn.Module` — ümumi modul/model interfeysi; `nn.Linear` — həmin interfeysdən istifadə edən konkret xətti qatdır.
</details>

<details>
<summary><strong>3) `forward()` nə edir və niyə adətən `model(x)` yazılır?</strong></summary>

Girişdən çıxışa hesablamanı təyin edir. `model(x)` modulun `__call__` mexanizmini (o cümlədən hook-ları) saxlayır.
</details>

<details>
<summary><strong>4) MSE ilə CrossEntropy arasında əsas fərq?</strong></summary>

MSE əsasən ədədi reqressiya səhvini, CE isə çoxsinifli təsnifatın logitlərinə görə doğru sinfin mənfi log-ehtimalını ölçür.
</details>

<details>
<summary><strong>5) Softmax nə edir, `dim=1` niyə vacibdir?</strong></summary>

Logitləri cəmi 1 olan ehtimal paylanmasına çevirir; `[N,C]` üçün sinif oxu 1-dir.
</details>

<details>
<summary><strong>6) `CrossEntropyLoss`-dan əvvəl Softmax etmək düzgündürmü?</strong></summary>

**Xeyr.** `CrossEntropyLoss` logitləri qəbul edir, daxildə ədədi sabit uyğun hesablamanı həyata keçirir.
</details>

<details>
<summary><strong>7) `loss.backward()` ilə `optimizer.step()` fərqi?</strong></summary>

Birincisi qradiyentləri hesablayır, ikincisi həmin qradiyentlərdən istifadə edib parametrləri dəyişir.
</details>

<details>
<summary><strong>8) LR çox böyük olsa nə olar?</strong></summary>

İtki dalğalana, arta və NaN yarana bilər; daha kiçik LR sınamaq lazımdır.
</details>

<details>
<summary><strong>9) Adam həmişə SGD-dən yaxşıdırmı?</strong></summary>

**Xeyr.** Məsələdən, hiperparametrlərdən, datasetdən və müqayisə metodundan asılıdır.
</details>

<details>
<summary><strong>10) `model.eval()` ilə `torch.no_grad()` eynidirmi?</strong></summary>

**Xeyr.** `eval()` bəzi qatların rejimini dəyişir; `no_grad()` autograd qeydiyyatını dayandırır.
</details>

### 10.2. Praktiki tapşırıqlar

- [ ] **T1.** `nn.Linear(5,10)` qatında neçə parametr var? Əl ilə hesablayıb `sum(p.numel() for p in layer.parameters())` ilə yoxlayın. **Cavab:** 60.
- [ ] **T2.** `x=3` üçün `f(x)=x²+2x` törəməsini `autograd` ilə hesablayın. **Cavab:** 8.
- [ ] **T3.** `[1,2,3]` logitlərinə Softmax tətbiq edin; cəmin 1 olduğunu yoxlayın.
- [ ] **T4.** CE-ni həm `nn.CrossEntropyLoss()`, həm `-log_softmax` ilə hesablayın; fərqin təqribən 0 olduğunu yoxlayın.
- [ ] **T5.** `optimizer.step()` sətrini müvəqqəti silin. Çəkilərin necə dəyişmədiyini müşahidə edin.
- [ ] **T6.** `Adam(lr=0.02)` əvəzinə `SGD(lr=0.05)` sınayın; eyni şəraitdə validation nəticələrini müqayisə edin.
- [ ] **T7.** Modeldən `ReLU` qatını çıxarın. Qeyri-xətti verilənlərdə qərar sərhədinin necə dəyişə biləcəyini izah edin.
- [ ] **T8.** 3 sinifli problemi 3 **müstəqil etiketli** problemə çevirərkən hansı loss-dan istifadə etməli olduğunuzu əsaslandırın. **Cavab:** `BCEWithLogitsLoss`.
- [ ] **T9.** Train accuracy yüksələrkən validation accuracy niyə düşə bilər? **Cavab:** mümkün səbəb overfitting və/və ya split/dağılım problemidir.
- [ ] **T10.** Yeni nümunələrdə Softmax ehtimalını hesablayın; ən böyük ehtimalın həmişə “zəmanət” olmadığını şərh edin.

**Təqdimat üçün tövsiyə:** Notebook-u ardıcıllıqla icra edin, nəticələrdən 2–3 ekran görüntüsü alın, SGD və Adam müqayisəsini, ikiqat Softmax səhvini və tətbiq ssenarisini öz sözlərinizlə izah etməyi məşq edin.

---

## 11. Terminlər lüğəti

| Termin | Azərbaycan dilində qısa izah |
|---|---|
| **Tensor** | Çoxölçülü ədədi verilənlər strukturu |
| **Feature** | Modelə verilən giriş əlaməti/xüsusiyyəti |
| **Label / target** | Öyrənmədə düzgün cavab |
| **Parameter** | Modelin öyrəndiyi `weight`/`bias` |
| **Hyperparameter** | Modelin təlimini idarə edən seçilmiş parametr (LR, batch və s.) |
| **Logit** | Ehtimala çevrilməmiş xam sinif skoru |
| **Activation** | Qat çıxışına tətbiq olunan funksiya (ReLU və s.) |
| **Forward pass** | Girişdən proqnoza doğru hesablama |
| **Backpropagation** | Törəmələrin arxaya yayılması |
| **Gradient** | Loss-un parametr üzrə törəməsi |
| **Learning rate** | Parametr yeniləməsinin addım ölçüsü |
| **Epoch** | Train məlumatının hamısından bir dəfə keçid |
| **Batch** | Bir addımda işlənən nümunələr toplusu |
| **Loss** | Optimizasiya üçün səhv göstəricisi |
| **Accuracy** | Doğru təsnifatların payı |
| **Overfitting** | Train-ə həddən artıq uyğunlaşma, zəif ümumiləşdirmə |
| **Inference** | Təlimdən sonra yeni məlumata proqnoz vermə |
| **Calibration** | Proqnoz ehtimallarının uzunmüddətli tezliklə nə qədər uyğun olması |

---

## 12. İstinadlar və daha çox oxu

Bu araşdırmanın izahları və nümunələri **PyTorch-un rəsmi sənədlərində göstərilən API davranışına** uyğun hazırlanıb (yoxlanma tarixi: **21.11.2026**). Sənəddəki açıqlamalar müəllifin öyrədici izahıdır; rəsmi mətnlərin hərfi tərcüməsi deyil.

1. [PyTorch — Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — Tensor, DataLoader, model, optimizasiya axını.
2. [PyTorch — Quickstart](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html) — `nn.Module` və tam təlim dövrəsi.
3. [PyTorch — `CrossEntropyLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) — Logitlər, sinif indeksləri, `LogSoftmax + NLLLoss` əlaqəsi.
4. [PyTorch — `Softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Softmax.html) — Softmax riyaziyyatı və `dim`.
5. [PyTorch — `BCEWithLogitsLoss`](https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html) — Birgə Sigmoid+BCE və ədədi sabitlik.
6. [PyTorch — `torch.optim`](https://docs.pytorch.org/docs/stable/optim.html) — SGD, Adam və scheduler-lər.
7. [PyTorch — `SGD`](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html) və [PyTorch — `Adam`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html) — optimizator parametrləri və davranışı.
8. [PyTorch — Installation](https://pytorch.org/get-started/locally/) — uyğun CPU/GPU qurulumu.

### Əlavə praktik fayl

[`neyron_sebekeleri_praktika.ipynb`](./neyron_sebekeleri_praktika.ipynb) — interaktiv kod, hesablama yoxlamaları, təlim qrafikləri, SGD/Adam müqayisəsi və tapşırıqlar.

---

> **Yekun fikir:** Güclü neyron şəbəkəsi tək bir alqoritmdən ibarət deyil. **Düzgün data + uyğun arxitektura + düzgün loss + stabil optimizasiya + qərəzsiz qiymətləndirmə** birlikdə real nəticə yaradır. Bu dörd mövzunun əlaqəsini başa düşmək daha mürəkkəb AI modellərini öyrənmək üçün möhkəm təməldir.
