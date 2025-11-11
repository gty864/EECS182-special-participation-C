# SGD vs Momentum Demo

```bash
pip install -r requirements.txt
pip install -e .

# GD only
python -m src.train optim.momentum=0.0

# Momentum
python -m src.train optim.momentum=0.9

# Open the plot
jupyter nbconvert --execute notebooks/demo.ipynb