import data_layer
import modeling

df_loaded = data_layer.load()
df_not_normalized = data_layer.clean(df_loaded)
df_clean = data_layer.process(df_not_normalized.copy())
df_not_normalized['cluster'] = modeling.cluster(df_clean)
df_not_normalized.to_csv("../data/final/clustered_data.csv", index=False)
