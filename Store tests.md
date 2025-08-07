{
    "model": "fixmatch",                # or "supervised", "fed_supervised", etc.
    "dataset": "cifar10",
    "non_iid_alpha": 0.1,               # Dirichlet alpha
    "labeled_fraction": 0.01,
    "lambda_u": 1.0,
    "clients": 10,
    "rounds": 20,
    "final_accuracy": 0.73,
    "final_f1_macro": 0.69,
    "loss": 0.55,
    "round_acc": [0.5, 0.55, ..., 0.73], # Optional: learning curve
    "round_loss": [...],                # Optional
    "pseudo_label_retention": True,     # If you do this in your extended method
    "adaptive_threshold": True,         # Same here
    "run_id": "fixmatch_alpha0.1_lf0.01_lu1.0",
}
