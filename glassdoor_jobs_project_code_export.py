from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "glassdoor_jobs.csv"
TARGET = "avg_salary_k"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def parse_salary_estimate(value: str) -> float:
    if pd.isna(value):
        return np.nan

    numbers = re.findall(r"\d+", str(value))
    if len(numbers) < 2:
        return np.nan
    return (float(numbers[0]) + float(numbers[1])) / 2


def build_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame[TARGET] = frame["Salary Estimate"].apply(parse_salary_estimate)
    frame["company_name_clean"] = frame["Company Name"].astype(str).str.split("\n").str[0]
    frame["company_age"] = np.where(frame["Founded"].between(1900, 2025), 2026 - frame["Founded"], np.nan)
    frame["is_headquarters_same_city"] = (
        frame["Location"].fillna("").str.split(",").str[0].str.strip().str.lower()
        == frame["Headquarters"].fillna("").str.split(",").str[0].str.strip().str.lower()
    ).astype(int)
    frame = frame.dropna(subset=[TARGET]).reset_index(drop=True)
    return frame


def dataset_overview(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "rows": [len(df)],
            "columns": [df.shape[1]],
            "missing_values": [int(df.isna().sum().sum())],
            "duplicates": [int(df.duplicated().sum())],
        }
    )


def prepare_features(frame: pd.DataFrame):
    columns = [
        "Job Title",
        "Rating",
        "Location",
        "Headquarters",
        "Size",
        "Industry",
        "Sector",
        "Revenue",
        "Type of ownership",
        "company_name_clean",
        "company_age",
        "is_headquarters_same_city",
    ]
    usable = frame[columns + [TARGET]].copy()
    x = usable.drop(columns=[TARGET])
    y = usable[TARGET]

    numeric_cols = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [column for column in x.columns if column not in numeric_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    return x, y, preprocessor


def extract_feature_importance(pipeline: Pipeline, top_n: int = 12) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()

    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_)
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    importance = pd.DataFrame({"feature": feature_names, "importance": values})
    return importance.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)


def run_modeling(random_state: int = 42):
    raw = load_data()
    frame = build_model_frame(raw)
    x, y, preprocessor = prepare_features(frame)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=random_state
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=random_state,
            min_samples_leaf=2,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=random_state),
    }

    rows = []
    fitted = {}
    for name, model in models.items():
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )
        pipe.fit(x_train, y_train)
        pred = pipe.predict(x_test)
        rows.append(
            {
                "model": name,
                "mae": round(mean_absolute_error(y_test, pred), 3),
                "rmse": round(mean_squared_error(y_test, pred, squared=False), 3),
                "r2": round(r2_score(y_test, pred), 3),
            }
        )
        fitted[name] = pipe

    metrics = pd.DataFrame(rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    best_model = metrics.loc[0, "model"]
    best_pipeline = fitted[best_model]

    return {
        "raw_frame": raw,
        "model_frame": frame,
        "metrics": metrics,
        "best_model": best_model,
        "best_pipeline": best_pipeline,
        "feature_importance": extract_feature_importance(best_pipeline),
        "sector_salary": frame.groupby("Sector")[TARGET].median().sort_values(ascending=False).head(10),
    }


def project_summary(results: dict):
    metrics = results["metrics"]
    return [
        f"Best model: {results['best_model']}",
        f"Lowest RMSE: {metrics.loc[0, 'rmse']}",
        f"Top sector by median salary: {results['sector_salary'].index[0] if not results['sector_salary'].empty else 'n/a'}",
    ]


if __name__ == "__main__":
    output = run_modeling()
    print(dataset_overview(output["raw_frame"]))
    print(output["metrics"])
    print(output["feature_importance"])