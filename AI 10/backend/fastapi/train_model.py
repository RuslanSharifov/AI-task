
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


FEATURES = ["Bedrooms", "Bathrooms", "Sqm", "City"]
CSV_COLUMNS = ["PriceAZN", *FEATURES]


def train_model(data: pd.DataFrame):
    # 1. CSV strukturunu yoxlayırıq
    if list(data.columns) != CSV_COLUMNS:
        raise ValueError(
            f"CSV sütunları belə olmalıdır: {', '.join(CSV_COLUMNS)}"
        )

    if len(data) < 10:
        raise ValueError(
            "Təlim üçün ən azı 10 mənzil lazımdır"
        )

    data = data.copy()

    # 2. Rəqəmlərin düzgünlüyünü yoxlayırıq
    for column in [
        "PriceAZN", "Bedrooms", "Bathrooms", "Sqm"
    ]:
        try:
            data[column] = pd.to_numeric(
                data[column], errors="raise"
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{column} yalnız rəqəm olmalıdır"
            ) from exc

        values = data[column].to_numpy(dtype=float)

        if (
            not np.isfinite(values).all()
            or (data[column] <= 0).any()
        ):
            raise ValueError(
                f"{column} müsbət və sonlu rəqəm olmalıdır"
            )

    for column in ["Bedrooms", "Bathrooms"]:
        if (data[column] % 1 != 0).any():
            raise ValueError(
                f"{column} müsbət tam ədəd olmalıdır"
            )

    # 3. Şəhər məlumatını yoxlayırıq
    if (
        data["City"].isna().any()
        or (
            data["City"].astype(str).str.strip() == ""
        ).any()
    ):
        raise ValueError("City boş ola bilməz")

    data["City"] = data["City"].astype(str).str.strip()

    # 4. Rəqəmsal məlumatların hazırlanması
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    # 5. Şəhər adlarının kodlaşdırılması
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        ),
    ])

    # 6. Preprocessing pipeline
    preprocessor = ColumnTransformer([
        (
            "numeric",
            numeric_pipeline,
            ["Bedrooms", "Bathrooms", "Sqm"]
        ),
        (
            "categorical",
            categorical_pipeline,
            ["City"]
        ),
    ])

    # 7. RandomForest modeli
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    # 8. Məlumatları train və test hissələrinə ayırırıq
    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data["PriceAZN"],
        test_size=0.2,
        random_state=42
    )

    # 9. Preprocessing və model yalnız train-də öyrədilir
    pipeline.fit(X_train, y_train)

    # 10. Test proqnozları
    predictions = pipeline.predict(X_test)

    # 11. Metrikalar
    mse = mean_squared_error(y_test, predictions)

    metrics = {
        "mae": round(
            float(mean_absolute_error(y_test, predictions)),
            2
        ),
        "mse": round(float(mse), 2),
        "rmse": round(float(np.sqrt(mse)), 2),
        "r2": round(
            float(r2_score(y_test, predictions)),
            4
        ),
    }

    # 12. FastAPI-nin oxuya biləcəyi model paketi
    bundle = {
        "pipeline": pipeline,
        "featured_order": FEATURES,
    }

    return bundle, metrics, len(data)


def save_model_atomically(bundle: dict, model_path: Path):
    model_path.parent.mkdir(
        parents=True, exist_ok=True
    )

    # Əvvəlcə müvəqqəti model faylı yaradılır
    fd, temp_name = tempfile.mkstemp(
        dir=model_path.parent,
        suffix=".pkl"
    )
    os.close(fd)

    temp_path = Path(temp_name)

    try:
        joblib.dump(bundle, temp_path, compress=3)

        # Saxlanmış modelin oxuna bildiyini yoxlayırıq
        validated = joblib.load(temp_path)

        if (
            not isinstance(validated, dict)
            or "pipeline" not in validated
        ):
            raise ValueError(
                "Yaradılmış model düzgün deyil"
            )

        # Yalnız uğurlu model köhnəsini əvəz edir
        os.replace(temp_path, model_path)

        return validated

    finally:
        temp_path.unlink(missing_ok=True)
