
import csv
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field, field_validator
from filelock import FileLock, Timeout

import logging
import threading
from datetime import datetime, timezone

import joblib
import pandas as pd

from train_model import (
    FEATURES,
    train_model,
    save_model_atomically,
)


# FastAPI tətbiqi
app = FastAPI(
    title="AI 10 - Housing Price Prediction",
    description="Mənzil qiymətlərinin idarə edilməsi",
    version="1.0.0"
)


# Layihənin əsas AI 10 qovluğu
BASE_DIR = Path(__file__).resolve().parents[2]

# CSV faylının ünvanı
CSV_PATH = BASE_DIR / "data" / "housing_az_realistic.csv"

# Fayl əməliyyatlarını qorumaq üçün kilid
CSV_LOCK = FileLock(str(CSV_PATH) + ".lock")



# Öyrədilmiş modelin saxlanacağı yer
MODEL_PATH = BASE_DIR / "models" / "models_az.pkl"
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

# Eyni vaxtda iki Retrain prosesini bloklayır
RETRAIN_LOCK = FileLock(str(MODEL_PATH) + ".lock")

# Modelin yaddaşda təhlükəsiz dəyişdirilməsi
MODEL_GUARD = threading.Lock()
PIPELINE = None

# Server başladıqda əvvəlki modeli yüklə
if MODEL_PATH.is_file():
    try:
        loaded = joblib.load(MODEL_PATH)
        PIPELINE = loaded["pipeline"]
        print("Əvvəlki AI modeli yükləndi")
    except Exception:
        logging.exception("Əvvəlki model yüklənə bilmədi")
else:
    print("Model yoxdur. İlk Retrain tələb olunur.")




# CSV sütunları
COLUMNS = [
    "PriceAZN",
    "Bedrooms",
    "Bathrooms",
    "Sqm",
    "City"
]


# İstifadəçinin daxil etdiyi mənzil məlumatları
class ApartmentIn(BaseModel):
    bedrooms: int = Field(strict=True, gt=0)
    bathrooms: int = Field(strict=True, gt=0)
    sqm: float = Field(gt=0, allow_inf_nan=False)
    city: str
    priceAZN: float = Field(gt=0, allow_inf_nan=False)

    @field_validator("city")
    @classmethod
    def check_city(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Şəhər boş ola bilməz")

        return value


# Əsas səhifə
@app.get("/")
def home():
    return {
        "message": "AI 10 Housing API işləyir!",
        "status": "success"
    }


# Serverin vəziyyəti
@app.get("/health")
def health():
    return {"status": "healthy"}


# Yeni mənzil əlavə edilməsi
@app.post("/apartments", status_code=201)
def add_apartment(apartment: ApartmentIn):

    if not CSV_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="CSV faylı tapılmadı"
        )

    temp_path = None

    try:
        # Paralel CSV əməliyyatlarını bloklayırıq
        with CSV_LOCK.acquire(timeout=10):

            # Mövcud məlumatları oxuyuruq
            with CSV_PATH.open(
                "r", encoding="utf-8-sig", newline=""
            ) as file:

                reader = csv.DictReader(file)

                if reader.fieldnames != COLUMNS:
                    raise HTTPException(
                        status_code=500,
                        detail="CSV sütunları düzgün deyil"
                    )

                old_rows = list(reader)

            # Yeni mənzilin CSV formatı
            new_row = {
                "PriceAZN": apartment.priceAZN,
                "Bedrooms": apartment.bedrooms,
                "Bathrooms": apartment.bathrooms,
                "Sqm": apartment.sqm,
                "City": apartment.city
            }

            # Əvvəlcə müvəqqəti fayl yaradırıq
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=CSV_PATH.parent,
                suffix=".tmp",
                delete=False
            ) as temp_file:

                temp_path = Path(temp_file.name)

                writer = csv.DictWriter(
                    temp_file,
                    fieldnames=COLUMNS
                )

                writer.writeheader()
                writer.writerows(old_rows)
                writer.writerow(new_row)

                temp_file.flush()
                os.fsync(temp_file.fileno())

            # Hazır faylı əsas CSV ilə əvəz edirik
            os.replace(temp_path, CSV_PATH)
            temp_path = None

            return {
                "message": "Mənzil uğurla əlavə edildi",
                "added": 1,
                "total_count": len(old_rows) + 1
            }

    except Timeout:
        raise HTTPException(
            status_code=503,
            detail="CSV hazırda başqa əməliyyatda istifadə olunur"
        )

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="CSV faylına yazmaq mümkün olmadı"
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)



@app.post("/apartments/import", status_code=201)
async def import_apartments(file: UploadFile = File(...)):
    # 1. Faylın formatını yoxlayırıq
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Yalnız CSV faylı yükləmək olar"
        )

    # 2. Faylın ölçüsünü məhdudlaşdırırıq (5 MB)
    max_size = 5 * 1024 * 1024
    content = await file.read(max_size + 1)

    if len(content) > max_size:
        raise HTTPException(
            status_code=413,
            detail="CSV faylı 5 MB-dan böyük ola bilməz"
        )

    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="CSV UTF-8 formatında olmalıdır"
        )

    # 3. CSV faylını oxuyuruq
    reader = csv.DictReader(
        io.StringIO(text, newline=""),
        strict=True
    )

    new_rows = []

    try:
        headers = reader.fieldnames

        if not headers:
            raise HTTPException(
                status_code=400,
                detail="CSV faylı boşdur"
            )

        if len(headers) != len(set(headers)):
            raise HTTPException(
                status_code=400,
                detail="CSV başlığında təkrar sütunlar var"
            )

        missing = [
            column for column in COLUMNS
            if column not in headers
        ]

        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Çatışmayan sütunlar: {', '.join(missing)}"
            )

        # 4. Bütün mənzilləri əvvəlcədən yoxlayırıq
        for row in reader:
            line_number = reader.line_num

            try:
                if None in row:
                    raise ValueError(
                        "Sətirdə artıq sütun var"
                    )

                apartment = ApartmentIn(
                    bedrooms=int((row["Bedrooms"] or "").strip()),
                    bathrooms=int((row["Bathrooms"] or "").strip()),
                    sqm=float((row["Sqm"] or "").strip()),
                    city=(row["City"] or "").strip(),
                    priceAZN=float((row["PriceAZN"] or "").strip())
                )

            except (ValueError, TypeError) as error:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"CSV sətir {line_number}: "
                        f"{error}"
                    )
                )

            new_rows.append({
                "PriceAZN": apartment.priceAZN,
                "Bedrooms": apartment.bedrooms,
                "Bathrooms": apartment.bathrooms,
                "Sqm": apartment.sqm,
                "City": apartment.city
            })

    except csv.Error as error:
        raise HTTPException(
            status_code=400,
            detail=(
                f"CSV format xətası, "
                f"sətir {reader.line_num}: {error}"
            )
        )

    if not new_rows:
        raise HTTPException(
            status_code=400,
            detail="CSV-də əlavə ediləcək mənzil yoxdur"
        )

    # 5. Əsas CSV mövcuddurmu?
    if not CSV_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail="Əsas mənzil CSV faylı tapılmadı"
        )

    temp_path = None

    try:
        # 6. Paralel yazılmaların qarşısını alırıq
        with CSV_LOCK.acquire(timeout=10):

            with CSV_PATH.open(
                "r", encoding="utf-8-sig", newline=""
            ) as existing_file:

                existing_reader = csv.DictReader(existing_file)

                if existing_reader.fieldnames != COLUMNS:
                    raise HTTPException(
                        status_code=500,
                        detail="Əsas CSV sütunları düzgün deyil"
                    )

                old_rows = list(existing_reader)

            # 7. Yeni və köhnə məlumatları hazırlayırıq
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=CSV_PATH.parent,
                suffix=".tmp",
                delete=False
            ) as temp_file:

                temp_path = Path(temp_file.name)

                writer = csv.DictWriter(
                    temp_file,
                    fieldnames=COLUMNS
                )

                writer.writeheader()
                writer.writerows(old_rows)
                writer.writerows(new_rows)

                temp_file.flush()
                os.fsync(temp_file.fileno())

            # 8. Yalnız tam hazır faylı yerləşdiririk
            os.replace(temp_path, CSV_PATH)
            temp_path = None

            total_count = len(old_rows) + len(new_rows)

        return {
            "message": "CSV uğurla idxal edildi",
            "added": len(new_rows),
            "total_count": total_count
        }

    except Timeout:
        raise HTTPException(
            status_code=503,
            detail="CSV hazırda başqa əməliyyatda istifadə olunur"
        )

    except OSError:
        raise HTTPException(
            status_code=500,
            detail="CSV faylına yazmaq mümkün olmadı"
        )

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)



@app.post("/retrain")
def retrain():
    global PIPELINE

    try:
        # Eyni anda yalnız bir model öyrədilə bilər
        with RETRAIN_LOCK.acquire(timeout=0):

            try:
                # CSV-ni digər yazma əməliyyatlarından qoruyuruq
                with CSV_LOCK.acquire(timeout=10):

                    if not CSV_PATH.is_file():
                        raise HTTPException(
                            status_code=500,
                            detail="Əsas CSV faylı tapılmadı"
                        )

                    # Ən son məlumatların surətini oxuyuruq
                    data = pd.read_csv(
                        CSV_PATH,
                        encoding="utf-8-sig"
                    )

            except Timeout:
                raise HTTPException(
                    status_code=503,
                    detail="CSV hazırda istifadə olunur"
                )

            # Yeni modeli öyrədirik
            try:
                bundle, metrics, row_count = train_model(data)
            except ValueError as error:
                raise HTTPException(
                    status_code=422,
                    detail=str(error)
                )

            # Yeni modeli təhlükəsiz fayla yazırıq
            saved_bundle = save_model_atomically(
                bundle, MODEL_PATH
            )

            # FastAPI yaddaşındakı modeli yeniləyirik
            with MODEL_GUARD:
                PIPELINE = saved_bundle["pipeline"]

            # Təlimin bitmə vaxtı
            finished_at = datetime.now(
                timezone.utc
            ).isoformat(timespec="seconds")

            return {
                "message": "Model uğurla yenidən öyrədildi",
                "used_rows": row_count,
                "metrics": metrics,
                "finished_at": finished_at,
            }

    except Timeout:
        raise HTTPException(
            status_code=409,
            detail="Hazırda başqa Retrain prosesi işləyir"
        )

    except HTTPException:
        raise

    except Exception:
        logging.exception("Retrain zamanı xəta")

        raise HTTPException(
            status_code=500,
            detail=(
                "Retrain uğursuz oldu. "
                "Əvvəlki işlək model saxlanıldı."
            )
        )





class PredictIn(BaseModel):
    bedrooms: int = Field(strict=True, gt=0)
    bathrooms: int = Field(strict=True, gt=0)
    sqm: float = Field(gt=0, allow_inf_nan=False)
    city: str

    @field_validator("city")
    @classmethod
    def check_predict_city(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Şəhər boş ola bilməz")
        return value

@app.post("/predict")
def predict(apartment: PredictIn):
    # Yaddaşdakı aktiv modeli götürürük
    with MODEL_GUARD:
        current_model = PIPELINE

    if current_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model yüklənməyib. Əvvəl Retrain edin."
        )

    features = pd.DataFrame([{
        "Bedrooms": apartment.bedrooms,
        "Bathrooms": apartment.bathrooms,
        "Sqm": apartment.sqm,
        "City": apartment.city,
    }], columns=FEATURES)

    try:
        result = current_model.predict(features)[0]

        return {
            "priceAZN": float(result)
        }

    except Exception:
        logging.exception("Prediction xətası")
        raise HTTPException(
            status_code=500,
            detail="Qiymət proqnozlaşdırıla bilmədi"
        )