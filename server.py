import flwr as fl

def weighted_average(metrics):
    # Estrazione delle metriche di loss 
    losses = [m[2]["loss"] for m in metrics if len(m) == 3 and isinstance(m[2], dict) and "loss" in m[2]]
    examples = [m[1] for m in metrics if len(m) >= 2] # Almeno 2 elementi devono essere presenti

    # Calcolo della media ponderata della loss (se presenti le metriche)
    if losses:
        weighted_loss_sum = sum([loss * num for loss, num in zip(losses, examples)])
        total_examples = sum(examples)
        return weighted_loss_sum / total_examples
    else:
        return {}  # Restituisci un dizionario vuoto se non ci sono metriche

# Definire la strategia federata
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=5,
    min_available_clients=5,
    fit_metrics_aggregation_fn=weighted_average,  
    evaluate_metrics_aggregation_fn=weighted_average,
)

# Avviare il server
fl.server.start_server(
    server_address="localhost:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy
)
