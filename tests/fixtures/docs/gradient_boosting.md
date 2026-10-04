# Gradient boosting

Gradient boosting adds trees one at a time. Every new tree is fitted to the
errors of the trees before it. The learning rate shrinks the contribution of
each tree, so lower learning rates usually need a larger n_estimators value.
