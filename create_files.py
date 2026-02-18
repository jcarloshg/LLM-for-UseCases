import json
import os

# Create data JSON files
os.makedirs('data/examples', exist_ok=True)
os.makedirs('data/validation', exist_ok=True)

with open('data/examples/user_stories.json', 'w') as f:
    json.dump([], f)

with open('data/validation/test_dataset.json', 'w') as f:
    json.dump([], f)

print("Files created successfully")
