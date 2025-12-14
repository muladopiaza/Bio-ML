# Bio-ML
Sci-kit learn like Machine learning library from scratch but also comes with bioinformatics specific machine learning algorithms as well such a K-mer based classification
classDiagram
    %% Base Classes
    class BaseEstimator {
        <<abstract>>
        -parameters
        -fitted
        +fit(X, y)
        +predict(X)
        +score(X, y)
        +get_params()
        +set_params()
    }

    class Optimizer {
        <<Singleton, Strategy>>
        -learning_rate
        -max_iterations
        -tolerance
        +optimize(loss_function, parameters)
    }

    class LossFunction {
        <<Strategy>>
        -name
        +compute(y_true, y_pred)
        +gradient(y_true, y_pred)
    }

    class Dataset {
        -X
        -y
        -feature_names
        -target_names
        +train_test_split()
        +normalize()
        +shuffle()
    }

    class ModelFactory {
        <<Factory>>
        +create_model(model_type, **params)
    }

    class Metrics {
        <<Strategy>>
        +accuracy(y_true, y_pred)
        +precision(y_true, y_pred)
        +recall(y_true, y_pred)
        +f1_score(y_true, y_pred)
        +roc_auc(y_true, y_pred)
    }

    class ModelManager {
        -models
        +train_all(datasets)
        +evaluate_all(datasets)
    }

    %% Relationships
    BaseEstimator o-- Optimizer : "has-a"
    BaseEstimator o-- LossFunction : "has-a"
    BaseEstimator --> Dataset : "uses"
    BaseEstimator --> Metrics : "evaluates with"
    ModelManager o-- BaseEstimator : "manages"
    ModelFactory --> BaseEstimator : "creates"

    %% Inheritance example for future models
    BaseEstimator <|-- LinearRegression
    BaseEstimator <|-- RandomForest
    BaseEstimator <|-- KMeans
