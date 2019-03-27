# Cencus 1994 USA dataset
Prediction task is to determine whether a person makes over 50K a year.<br />
Reference to dataset: [archive.ics.uci.edu](https://archive.ics.uci.edu/ml/datasets/adult)
## Version 1
We use Logistic Regression to predict. We use data from ```adult.test``` without ```adult.test```<br />
Features:
- One-Hot-Encoding categorial variables
- Splitting dataset using ```train_test_split()``` function
- Used features: ```['age', 'workclass', 'education', 'gender', 'hours-per-week', 'occupation', 'income']```
