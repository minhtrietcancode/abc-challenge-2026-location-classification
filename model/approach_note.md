### Data Splitting
- Day 1 + 2 for training 
- Day 3 for testing

### Approach 1: Starting pipeline
- Window size = 1
- 25 beacons x 3 features (mean, std, count)
- Gradient Boosting Algorithm: XGBoost

### Approach 2: Funny trick applied on approach 1
- just penalty XGBoost more for minority class / increase the weight of minority class 3x times, however obviously this lord trick cannot be used, it just means pay attention 3x times to shjt ...

### Approach 3: SMOTE applied to handle class imbalance compared to approach 1
- we try to use SMOTE technique to handle class imbalance here, however this still does not make any big difference

### Approach 4: more features introduced - top beacons features and some related things 
- we try to use this idea applied to improve approach 1, which is the starting pipeline but this still does not work. We can now conclude that the problem is with 2 things: class imbalance which destroy our macro f1 score and the misunderstading of classification between adjacent rooms (we need some way to filter noise in data to handle this problem)

### Approach 5: 