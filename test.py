
from bohby import optimize_hyperparameters

optimize_hyperparameters(
    None,
    {
        "beta": {
            "lower": 1.0,
            "upper": 250.0,
            "type": float
        },
        "lr": {
            "lower": 1e-7,
            "upper": 1e-3,
            "type": float
        },
    },
    lambda train_and_validate_fn: None,
    num_iterations = 100
)
