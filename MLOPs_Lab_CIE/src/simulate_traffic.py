import requests
import json
import os

# Test /status
status = requests.get('http://127.0.0.1:8500/status').json()

# Test /estimate
test_input = {'gpu_memory_gb': 40, 'batch_size': 64, 'model_params_millions': 1500, 'queue_depth': 8}
pred = requests.post('http://127.0.0.1:8500/estimate', json=test_input).json()

output = {
    'health_endpoint': '/status',
    'predict_endpoint': '/estimate',
    'port': 8500,
    'health_response': status,
    'test_input': test_input,
    'prediction': pred['prediction']
}

os.makedirs('results', exist_ok=True)
with open('results/step2_serving.json', 'w') as f:
    json.dump(output, f, indent=2)

print('Task 2 done!')
print(json.dumps(output, indent=2))