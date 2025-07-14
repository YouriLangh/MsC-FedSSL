#utils 

import optuna

def objective(trial):
    # Sample hyperparameters
    lr_backbone = trial.suggest_loguniform("lr_backbone", 1e-5, 1e-2)
    lr_head = trial.suggest_loguniform("lr_head", 1e-4, 1e-1)
    lambda_u = trial.suggest_uniform("lambda_u", 0.5, 5.0)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])

    # Modify args
    args = arg_parse()
    load_config(args, "Dataset", args.dataset, args.data_config_dir)
    load_config(args, "Hyperparameters", args.model, args.hyperparam_config_dir)

    args.lambda_u = lambda_u  # Update FixMatch param
    args.optimizer_name = optimizer_name
    args.lr_backbone = lr_backbone
    args.lr_head = lr_head

    # Create and train
    federated_system = FedAvg(args)
    federated_system.train()

    # Evaluate
    test_data = load_dataset(args.dataset, args.data_dir, transform=transforms.ToTensor(), train=False)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    test_log = federated_system.evaluate(test_loader)

    # Return a score: higher is better (e.g., test accuracy)
    return test_log["accuracy"]
