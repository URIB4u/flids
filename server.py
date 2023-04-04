import flwr as fl
import sys
import numpy as np

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        rnd,
        results,
        failures
    ):
        aggregated_weights = super().aggregate_fit(rnd, results, failures)
        if aggregated_weights is not None:
            # Save aggregated_weights
            print(f"Saving round {rnd} aggregated_weights...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_weights)
        return aggregated_weights


# Define an evaluation metrics aggregation function
#def evaluate_fn(history):
        total_accuracy = 0.0
        total_count = 0
        for metrics in history:
            total_accuracy += metrics["accuracy"] * metrics["count"]
            total_count += metrics["count"]
        if total_count == 0:
            return {"accuracy": 0.0}
        else:
            return {"accuracy": total_accuracy / total_count}
# Create strategy and run server
strategy = SaveModelStrategy()

# Start Flower server for three rounds of federated learning
fl.server.start_server(
        server_address = 'localhost:'+str(sys.argv[1]) , 
        config=fl.server.ServerConfig(num_rounds=3) ,
        grpc_max_message_length = 1024*1024*1024,
        strategy = strategy
)
