# utils.py

import json

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)

def nice(msg):
    return f"*** {msg} ***"

def compute_position_size(balance, risk_percent):
    return balance * risk_percent / 100
