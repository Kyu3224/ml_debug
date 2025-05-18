# ml_debug

This repository contains a collection of utility functions designed to facilitate debugging and troubleshooting of machine learning models.

## Installation

You can install the package via `pip`:

```bash
pip install git+git@github.com:Kyu3224/ml_debug.git
```

Alternatively, install locally from the source code:
```bash
pip install git+https://github.com/Kyu3224/ml_debug.git
cd ml_debug
pip install .
```

## Usage Examples
```bash
from ml_debug import elapsed_time

with elapsed_time(mode="seconds", precision=3):
    # Your ML model training or inference code here
    train_model()
```
This will print the elapsed time in seconds with 3 decimal places.

## Future plans
Currently, ml_debug provides the elapsed_time utility to measure execution duration easily.
I am actively developing and planning to add many more useful debugging and monitoring tools for machine learning workflows in future releases.
Stay tuned for upcoming features and improvements!
