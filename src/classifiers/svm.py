import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score

class SVMClassifier:

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42
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

    def tune_hyperparameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        c_range: list = None,
        gamma_range: list = None,
        cv: int = 5
    ) -> dict:
        if c_range is None:
            c_range = [0.1, 1, 10, 100]
        if gamma_range is None:
            gamma_range = ['scale', 'auto', 0.01, 0.1]

        param_grid = {
            'C': c_range,
            'gamma': gamma_range,
            'kernel': [self.kernel]
        }

        grid_search = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            refit=True
        )

        grid_search.fit(X, y)

        self.C = grid_search.best_params_['C']
        self.gamma = grid_search.best_params_['gamma']
        self.model = grid_search.best_estimator_
        self._is_trained = True

        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }

    def get_params(self) -> dict:
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma
        }
