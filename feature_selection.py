from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# This class is just a wrapper around the classes we use for feature selection
class FeatureSelector:

    def __init__(self, model) -> None:
        models = {
            "PCA" : PCA(n_components=0.8, svd_solver='full'),
            "LDA" : LDA()
        }
        self.model = models[model]

    def transform(self, x_data):
        return self.model.transform(x_data)
    
    def fit_transform(self, x_data, y_data):
        return self.model.fit_transform(x_data, y_data)