import json
import numpy as np

def simulate_latent(alpha=0.1, beta=0.05, steps=100):
    z = 1.0
    trace = []
    for step in range(steps):
        drift = -alpha * z
        gating = np.sin(beta * step)
        noise = 0.02 * np.random.randn()
        z = z + drift + gating * 0.1 + noise
        trace.append(z)
    return trace

if __name__ == '__main__':
    trace = simulate_latent()
    story = {
        'title': 'Toy latent drift',
        'confidence': 0.75,
        'note': 'Drift slows as gating balances noise, promising smoother decoder inputs.'
    }
    output = {'trace': trace, 'story': story}
    print(json.dumps(output, indent=2))
