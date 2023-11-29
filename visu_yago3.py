# %%
import networkx as nx
import pickle as pkl
from graphlet_construction import Graphlet
import pandas as pd
from evaluate import evaluate_embeddings


# %%
with open(r"data/yago3_2022.pickle", "rb") as input_file:
    yago_data = pkl.load(input_file)

#G = nx.Graph() 
#edges = list(zip(yago_data["edge_index"][0].numpy(), yago_data["edge_index"][1].numpy()))
#G.add_edges_from(edges)


# %%
graphlet = Graphlet(yago_data, num_hops=2, max_nodes=100)
# %%
data = graphlet.extract_graph_data(1125)
# %%
data
# %%
from torch_geometric.utils import to_networkx

G = to_networkx(data[0])
# %%
nx.draw(G)
# %%
data[0].mapping

# %%

embeddings= pd.read_parquet("data/emb_mure_yago3_2022_full.parquet")
embeddings['ent_idx'] = embeddings["Entity"].map(dict(yago_data["ent2idx"]))
embeddings = embeddings[~embeddings["ent_idx"].isna()]
embeddings.sort_values("ent_idx", ascending=True, inplace=True)
# %%
embed_values = embeddings.iloc[:, :-2].to_numpy()
# %%
embed_values
# %%

evaluate_embeddings(embed_values, yago_data, name='ken')
# %%
yago_data["ent2idx"]
# %%
