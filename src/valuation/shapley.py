import numpy as np
from update import test_inference
from utils import average_weights, initialize_model, get_device

def compute_shapley(args, global_weights, client_weights, test_dataset):
    """Estimate Shapley values for participants in a round using permutation sampling."""
    device = get_device()
    model = initialize_model(args) 
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()

    m = len(client_weights)
    epsilon, delta, r = 0.1, 0.05, 1  # allow 10% error at 95% confidence, r = 1 since accuracy in [0, 1]  
    t = int((2 * r**2 / epsilon**2) * np.log(2 * m / delta))
    
    base_acc = test_inference(model, test_dataset)[0] 
    shapley_updates = np.zeros(m)

    for _ in range(t):
        permutation = np.random.permutation(m)
        prev_acc = base_acc

        model.load_state_dict(global_weights)
        current_weights = []

        for i in permutation:
            current_weights.append(client_weights[i])
            model.load_state_dict(average_weights(current_weights))
            curr_acc = test_inference(model, test_dataset)[0]
            shapley_updates[i] += curr_acc - prev_acc
            prev_acc = curr_acc
    
    shapley_updates /= t
    return shapley_updates