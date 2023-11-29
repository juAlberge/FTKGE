from pathlib import Path
from time import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import HistGradientBoostingRegressor as HGBR
from sklearn.model_selection import (
    cross_val_score,
    RepeatedKFold,
    GridSearchCV,
)


root_dir = Path(__file__).absolute().parent
results_file = root_dir / "scores.parquet"


targets = {
        "tables/us_elections.parquet": "US elections",
        "tables/datasets/evaluation/housing_prices/target_log.parquet": "Housing prices",
        "tables/us_accidents.parquet": "US accidents",
        "tables/movie_revenues.parquet": "Movie revenues",
        "tables/company_employees.parquet": "Company employees",
    }



def prediction_scores(
    embeddings,
    id,
    target_file,
    data,
    n_repeats,
    scoring,
    results_file,
    tune_hyperparameters=True,
):
    # Load target dataframe
    target = pd.read_parquet(root_dir / target_file)

    # Load previously stored results
    if Path(results_file).is_file():
        df_res = pd.read_parquet(results_file)
    else:
        df_res = None
    
    print("Embeddings scores on target: ", targets[target_file])

    # Replace entity names by their idx
    target["col_to_embed"] = target["col_to_embed"].map(dict(data["ent2idx"])).astype(int)
    
    X_emb = embeddings[target["col_to_embed"]]
    y = target["target"]
    model = HGBR()
    cv = RepeatedKFold(n_splits=5, n_repeats=n_repeats)

    if tune_hyperparameters:
        param_grid = {
            "max_depth": [2, 4, 6, None],
            "min_samples_leaf": [4, 6, 10, 20],
        }
        model = GridSearchCV(
            model,
            param_grid=param_grid,
            scoring=scoring,
            cv=3,
        )
    start_time = time()
    cv_scores = cross_val_score(model, X_emb, y, cv=cv, scoring=scoring, n_jobs=15)
    duration = time() - start_time

    X_shape = X_emb.shape

    # Save results to a dataframe
    results = {
        "id": id,
        "target_file": str(target_file),
        "scoring": scoring,
        "duration": duration,
        "n_samples": X_shape[0],
        "n_features": X_shape[1],
        "scores": cv_scores,
    }
    new_df_res = pd.DataFrame([results])
    if Path(results_file).is_file():
        df_res = pd.read_parquet(results_file)
        df_res = pd.concat([df_res, new_df_res]).reset_index(drop=True)
    else:
        df_res = new_df_res
    df_res.to_parquet(results_file, index=False)

    return


def evaluate_embeddings(embeddings, data, name="fine-tuned"):
    embeddings = np.array(embeddings)
    for target_file in tqdm(targets.keys(), desc="Evaluating embeddings"):
            prediction_scores(
                embeddings=embeddings,
                id=name,
                target_file=target_file,
                data=data,
                n_repeats=5,
                scoring="r2",
                results_file=root_dir / "scores.parquet",
            )
    return


