# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from evaluate import TARGETS
import torch
import pickle as pkl
import igraph as ig

with open(r"data/yago3_2022.pickle", "rb") as input_file:
    YAGO_DATA = pkl.load(input_file)

# %%
## MAKE BARPLOT RESULTS TO SHOW R2 SCORES ON EACH DOWNSTREAM TASK

def make_barplot(path="scores.parquet", name_fig="ken_embeddings_results"):
    scores_targets = pd.read_parquet(path)
    scores_targets["mean_scores"]=scores_targets["scores"].apply(np.mean)
    scores_targets["target_name"] = scores_targets["target_file"].map(TARGETS)
    fig= plt.figure(figsize=(10, 5))
    sns.barplot(data=scores_targets,x="target_name", y="mean_scores", hue="id")
    score_ids = scores_targets.groupby("id").mean_scores.sum()
    print(score_ids)
    plt.ylabel("Mean Cross Validation R2 score")
    plt.xlabel("Downstream task")
    plt.show()
    plt.savefig(name_fig)
    return

#%%
make_barplot("scores.parquet")




# %%
# Le numéro de Bambi <3
i = 2158558
mask = (YAGO_DATA["edge_index"][0] == i) | (YAGO_DATA["edge_index"][1] == i)

edge_index = YAGO_DATA["edge_index"].T[mask].T
edge_type = YAGO_DATA["edge_type"][mask]
edge_type

# %%
## PLOT GRAPHLET WITH ATTENTION MECHANISM

edge_index = torch.Tensor([[0, 2, 3, 4, 5], 
                            [1, 0, 0, 0, 0]])

attention = torch.Tensor([[0.1], [0.4], [0.8], [0.2], [0.6], [0.05], [0.1]])

entity_mapping = torch.Tensor([
    [0, 2158558], 
    [1, 20042],
    [2, 1009473], 
    [3, 652141],
    [4, 1626574], 
    [5, 616140]
])

edge_type_mapping = torch.Tensor([
    [4], 
    [17], 
    [17],
    [28], 
    [28],
])

idx_to_ent = {index:entity for entity,index in YAGO_DATA["ent2idx"]}

idx_to_rel = {index:relation for relation,index in YAGO_DATA["rel2idx"]}

edges_index_numpy= edge_index.to(torch.long).numpy()
edge_type_mapping_numpy = edge_type_mapping.to(torch.long).numpy().flatten()
#%%
entity_mapping_numpy = entity_mapping.to(torch.long).numpy()
attention_numpy = attention.numpy() * 10

entity_mapping_dict= {t_ind: ind for t_ind, ind in entity_mapping_numpy}

nodes_values = np.unique(edges_index_numpy.flatten())
nodes_entities = [idx_to_ent[entity_mapping_dict[edge_val]] for edge_val in nodes_values]
edges_entities = [idx_to_rel[edge_val] for edge_val in edge_type_mapping_numpy]

# %% 
nodes_entities = [ent[1:-1] for ent in nodes_entities]
edges_entities = [ent[1:-1] for ent in edges_entities]

# %%
g = ig.Graph(edges=np.transpose(edges_index_numpy))

g.vs["label"] = nodes_entities
g.vs["weights"] = attention_numpy




# %%
fig, ax = plt.subplots(1, 1,figsize=(10, 5))

import seaborn as sns
alpha = .5
colors = sns.color_palette("hls", 3)
color_dict = {4: colors[0], 17: colors[1], 28: colors[2]}

ig.plot(g, 
        layout="fruchterman_reingold", 
        vertex_size = 32,
        vertex_frame_width = 0.5,
        vertex_color = 'tomato',
        edge_color = [color_dict[relation] + (alpha,) for relation in edge_type_mapping_numpy],
        edge_width=attention_numpy,
        edge_label=edges_entities,
        edge_curved=0, 
        target=ax
        )

plt.savefig('plot1.png', pad_inches='layout')


# %%
target = pd.read_parquet("tables/movie_revenues.parquet")
target["col_to_embed"] = target["col_to_embed"].map(dict(YAGO_DATA["ent2idx"]))
target = target[~target["col_to_embed"].isna()]
target["col_to_embed"] = target["col_to_embed"].astype(int)

# %%
entities, degrees = np.unique(YAGO_DATA["edge_index"], return_counts=True)
mask = np.isin(entities, target["col_to_embed"])

target_degrees = degrees[mask]
target_entities = entities[mask]

idx2idx = {v:k for k,v in dict(YAGO_DATA["ent2idx"]).items()}
#[idx2idx[idx] for idx in target_entities[np.isin(target_degrees, [5,6])]]

# %%


# %%
