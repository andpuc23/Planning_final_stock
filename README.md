# Planning_final_stock
Repo for final project at Planning Algorithms course at My University.

## Description
We want to build an reinforcement learning algorithm to trade stocks with maximum efficiency. To do this, we use:
- Rapidly exploring random tree (RRT)
- Value Iteration (VI)

### If we are good:
- Deep Q-learing
- Temporal difference
- Model-predictive control

## Structure of our project
- `data` folder holds the data we train and test on
- `utils` contains some useful functions, i.e. to prepare data for models
- `models` folder has subfolders dedicated to each algorithm we use
- `pipeline.ipynb` is the main file which runs the whole stuff

## Authors
Todo, see collaborators :D


### For authors:

1. Our actions are: sell, buy, do nothing
2. Our environment is (vector of prices, vector of pct_changes, current day & day of week)
3. Observation is (10 last prices, 10 last pct_changes (9 mb?), current day of week, number of stocks on agent's balance, money he has)
3. Reward function: total cost (n_stocks*stock_price + curr_money) - start_money
