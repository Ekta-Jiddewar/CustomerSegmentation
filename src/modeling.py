from sklearn.cluster import KMeans


def cluster(df_clean):
    best_model = 3
    kmeans = KMeans(n_clusters=best_model, random_state=0, n_init="auto").fit(df_clean)
    return kmeans.labels_
