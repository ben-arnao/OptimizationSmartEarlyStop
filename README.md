# OptimizationSmartEarlyStop
Algorithm to determine when optimization will not reach a previously established best score

When doing hyperparameter optimization, it can be very time consuming needing to train a model for the full amount of iterations each trial. This algorithm analayzes from a short history of steps, if the current opimization trajectory will surpass the "best" trial value. This allows us to stop training a model if it will not outperform the best trial.
