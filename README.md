# Benchmarking Machine Learning and Deep Learning Architectures for Financial Time Series Data

---

## **1. Dataloader Module**

*Foundational components for loading raw data.*

* **`class DataLoader`**
  A utility class for loading and wrapping data.
---

## **2. Datawrapper Module**

*Core data structures for managing features, labels, and predictions in memory.*

* **`class DataHandler`**
  Utility class for managing features, labels, and predictions within a pandas DataFrame. Provides methods for accessing, modifying, slicing, and copying the underlying data.

* **`class DataWrapper`**
  Higher-level interface for interacting with a `DataHandler`. Designed to be extended by specialized wrappers that add domain-specific logic.

---

## **3. Learner Module**

*The training engine. Coordinates the flow of data into models and generates predictions.*

* **`class Learner`**
  Coordinates training and testing of a model using a `Trainer` and `Tester`, operating on data segments defined by a `DataSelector` (scope).

* **`class Trainer`**
  Handles training loop logic. Initializes model parameters from the `DataWrapper` and fits the model using features and labels.

* **`class Tester`**
  Handles inference and prediction generation for a trained model on datasets wrapped by `DataWrapper`.

---

## **4. Model Module**

*Definitions of predictive architectures.*

* **`class Model`**
  Base class for machine learning models. Defines common attributes such as input columns (`in_cols`) and the underlying estimator or neural model.

* **`class NeuralModel`**
  Intermediate base class for neural network implementations, handling delegation of fitting and prediction logic.

* **`class ScikitModel`**
  Wrapper for scikit-learn estimators. Integrates sklearn models into the unified modeling interface, handling label preprocessing and dynamic selection between `predict` and `predict_proba`.

* **`class TorchModule`**
  Base class for PyTorch models. Handles training and prediction loops and is intended to be subclassed with concrete implementations of `forward()`, batch handling, and loss computation.

---

## **5. Optimizer Module**

*Hyperparameter tuning and optimization logic.*

* **`class Optimizer`**
  Manages hyperparameter optimization using Optuna. Coordinates the learner, data wrapper, evaluation metrics, and search space across multiple trials.

---

## **6. Simulation Module**

*Strategy execution and backtesting logic (easily adaptable to trading systems).*

* **`class Simulation`**
  Abstract base class for simulation-based strategy evaluation. Defines a standardized interface for evaluating strategies using predictions, actual outcomes, and external signals such as odds.
---

## **7. Transformer Module**

*Data handling for time-series iteration, windowing, and scoping.*

* **`class DataSelector`**
  Manages synchronized iteration over training and testing scopes. Controls updates, validation logic, and reset behavior across scope transitions.

* **`class Scope`**
  Abstract base class for data segmentation. Defines the interface and shared initialization logic for all scope implementations.

* **`class BaseTransformer`**
    Base transformer class.

* **`class ScopeSelector`**
    Base class responsible for selecting subsets of data according to the current state of a `Scope`,
    including update and reset logic.

* **`class Transformer`**
    Class to apply the transformation from the dict of transformation.

---

## **8. Utils Module**

*General-purpose utilities for metrics, caching, and experiment tracking.*

* **`class Cache`**
  Static utility for caching and loading Python objects using pickle serialization.

* **`class MLFlowTracker`**
  Static MLflow tracking utility that manages experiment runs, parameters, and metrics using class-level state.

* **`class Merger`**
  Utility for merging multiple `DataWrapper` instances into a single consolidated wrapper.

* **`class Evaluation` (Functions)**
  Collection of evaluation functions (e.g., `evaluate_metrics`) that compute and summarize classification metrics such as accuracy, Brier score, and Ranked Probability Score (RPS).
