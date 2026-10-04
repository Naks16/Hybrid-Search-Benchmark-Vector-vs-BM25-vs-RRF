# Random forests

A random forest trains many decision trees on bootstrap samples of the data.
The parameter n_estimators sets how many trees the forest contains. More trees
reduce variance but make training slower. Each tree also looks at a random
subset of features at every split, controlled by max_features.
