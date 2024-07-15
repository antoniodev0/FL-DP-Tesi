import flwr as fl

def weighted_average(metrics):
    # Estrazione delle metriche (loss o accuracy)
    values = [m[2].get("loss") if "loss" in m[2] else m[2].get("accuracy") for m in metrics if len(m) == 3 and isinstance(m[2], dict)]
    weights = [m[1] for m in metrics if len(m) >= 2] # Almeno 2 elementi devono essere presenti

    # Calcolo della media ponderata (se presenti le metriche)
    if values:
        weighted_sum = sum([v * w for v, w in zip(values, weights)])
        total_weight = sum(weights)
        return weighted_sum / total_weight
    else:
        return {}

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
