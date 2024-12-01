import numpy as np
from update import test_inference
from utils import average_weights, initialize_model, get_device

def compute_shapley(args, global_weights, client_weights, test_dataset):
    """Estimate Shapley values for participants in a round using permutation sampling."""
    device = get_device()
    model = initialize_model(args) 
    print("here1")
    model.load_state_dict(global_weights)
    model.to(device)
    model.train()
    print("here2")

    m = len(client_weights)
    epsilon, delta, r = 0.1, 0.05, 1  # allow 10% error at 95% confidence, r = 1 since accuracy in [0, 1]  
    t = int((2 * r**2 / epsilon**2) * np.log(2 * m / delta))
    
    print("here3")
    base_acc = test_inference(model, test_dataset)[0] 
    shapley_updates = np.zeros(m)

    print("here4")
    for _ in range(t):
        permutation = np.random.permutation(m)
        prev_acc = base_acc
        print("here5")

        model.load_state_dict(global_weights)
        current_weights = []
        print("here6")
        for i in permutation:
            current_weights.append(client_weights[i])
            print(current_weights)
            print("here7")
            avg_weights = average_weights(current_weights)
            print(avg_weights)
            model.load_state_dict(avg_weights)
            print("here9")
            curr_acc = test_inference(model, test_dataset)[0]
            print("here10")
            shapley_updates[i] += curr_acc - prev_acc
            print("here11")
            prev_acc = curr_acc
    
    shapley_updates /= t
    return shapley_updates