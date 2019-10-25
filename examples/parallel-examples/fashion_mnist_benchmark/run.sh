#!/bin/bash
for i in {1..5}
do
  python runner_random_search.py "$@"
done

for i in {1..5}
do
  python runner_gpyopt.py "$@"
done

for i in {1..5}
do
  python runner_successive_halving.py "$@"
done

for i in {1..5}
do
  python runner_population_based_training.py "$@"
done

