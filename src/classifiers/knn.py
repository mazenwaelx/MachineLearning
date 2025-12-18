import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

class KNNClassifier:

    def __init__(self, n_neighbors: int = 5, weights: str = 'distance'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric='euclidean',
            n_jobs=-1
        )
        self._is_trained = False

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self._is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return self.model.predict_proba(X)

    def tune_k(
        self,
        X: np.ndarray,
        y: np.ndarray,
        k_range: range = None,
        cv: int = 5
    ) -> int:
        if k_range is None:
            k_range = range(3, 21, 2)

        best_k = self.n_neighbors
        best_score = 0.0
        cv_results = {}

        for k in k_range:
            knn = KNeighborsClassifier(
                n_neighbors=k,
                weights=self.weights,
                metric='euclidean',
                n_jobs=-1
            )
            scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
            mean_score = scores.mean()
            cv_results[k] = {
                'mean_score': mean_score,
                'std_score': scores.std()
            }

            if mean_score > best_score:
                best_score = mean_score
                best_k = k

        self.n_neighbors = best_k
        self.model = KNeighborsClassifier(
            n_neighbors=best_k,
            weights=self.weights,
            metric='euclidean',
            n_jobs=-1
        )

        self.model.fit(X, y)
        self._is_trained = True

        return best_k

    def get_params(self) -> dict:
        return {
            'n_neighbors': self.n_neighbors,
            'weights': self.weights
        }
