import pandas as pd


def load_dataset(file_path):
    return pd.read_csv(file_path)


def summarize_dataset(dataset):
    return {
        "total_runs": dataset.shape[0],
        "total_parameters": dataset.shape[1],
        "average_yield": round(dataset["product_yield_g_l"].mean(), 2),
        "highest_yield": round(dataset["product_yield_g_l"].max(), 2),
        "lowest_yield": round(dataset["product_yield_g_l"].min(), 2),
    }


def best_process_condition(dataset):
    best_run = dataset.loc[dataset["product_yield_g_l"].idxmax()]
    return best_run
