
import pickle as pkl
import pandas as pd
from evaluate import evaluate_embeddings
import numpy as np
import os

with open(r"data/yago3_2022.pickle", "rb") as input_file:
    YAGO_DATA = pkl.load(input_file)


def read_and_process_embddings(path, data = YAGO_DATA):
    embeddings = pd.read_parquet(path)
    embeddings['ent_idx'] = embeddings["Entity"].map(dict(data["ent2idx"]))
    embeddings = embeddings[~embeddings["ent_idx"].isna()]
    embeddings.sort_values("ent_idx", ascending=True, inplace=True)
    return embeddings.iloc[:, :-2].to_numpy()
    
if __name__=="__main__":
    #embed_values = read_and_process_embddings("data/emb_mure_yago3_2022_full.parquet")
    #evaluate_embeddings(embed_values, YAGO_DATA, name='ken')

    num_entities = len(YAGO_DATA["ent2idx"])
    embed_values = np.zeros((num_entities, 200))
    for table in os.listdir('gat_embed'): 
        with open("gat_embed/" + table , "rb") as input_file:
            d = pkl.load(input_file)
            X_data, yago_idx= d["X_data_new"], d["yago_idx"]
            for i in range(len(yago_idx)):

                embed_values[yago_idx[i]] = X_data[i]
    evaluate_embeddings(embed_values, YAGO_DATA, name='KEN+GAT')