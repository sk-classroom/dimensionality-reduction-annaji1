import numpy as np

class PrincipalComponentAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray):
        self.mean = np.mean(X, axis=0)
        cen_mean = X - self.mean
        cov = np.cov(cen_mean, rowvar=False)
        vals, vecs = np.linalg.eig(cov)
        idx = np.argsort(vals)[::-1]
        self.components = vecs[:, idx[:self.n_components]]

    def transform(self, X: np.ndarray) -> np.ndarray:
        cen_mean = X - self.mean
        return np.dot(cen_mean, self.components)


class LinearDiscriminantAnalysis:
    def __init__(self, n_components: int) -> None:
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.mean = np.mean(X, axis=0)
        w = np.zeros((n_features, n_features))
        b = np.zeros((n_features, n_features))

        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            w += np.dot((X_c - mean_c).T, (X_c - mean_c))
            b += len(X_c) * np.outer((mean_c - self.mean), (mean_c - self.mean))

        vals, vecs = np.linalg.eig(np.linalg.inv(w).dot(b))
        idx = np.argsort(vals.real)[::-1]
        self.components = vecs[:, idx[:self.n_components]].real

    def transform(self, X: np.ndarray) -> np.ndarray:
        cen_mean = X - self.mean
        return np.dot(cen_mean, self.components)

class AdversarialExamples:
    def __init__(self) -> None:
        pass

    def pca_adversarial_data(self, n_samples, n_features):
        
        c1_mean = np.array([0, 0])
        c2_mean = np.array([5, 5])
        c_cov = np.array([[1, 0.5], [0.5, 1]])

        c1_sam = np.random.multivariate_normal(c1_mean, c_cov, n_samples // 2)
        c2_sam = np.random.multivariate_normal(c2_mean, c_cov, n_samples // 2)

        X = np.concatenate([c1_sam, c2_sam])
        y = np.concatenate([np.zeros(n_samples // 2), np.ones(n_samples // 2)]).astype(int)

        i = np.arange(n_samples)
        np.random.shuffle(i)
        X = X[i]
        y = y[i]

        pca = PrincipalComponentAnalysis(n_components=2)
        X_pca = pca.fit_transform(X)

        return X_pca, y
