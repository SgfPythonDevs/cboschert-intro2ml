__author__ = 'Chad'
import pandas as pd
import numpy as np

# Pandas makes data exploration and manipulation easier and more readable
# Pandas has its own read_csv method; it's smart enough to infer data types
train_df = pd.read_csv('.\\data\\train.csv', header=0)

# To prepare our data we must first convert text values to numeric values
# The Series object provides a map() function for easily translating values
sex_map = {'female': 0, 'male': 1}
train_df['SexNum'] = train_df.Sex.map(sex_map).astype(int)

# map() also supports more complex logic; here we're crudely parsing the title out of the Name column
train_df['Title'] = train_df.Name.map(lambda x: x.split(',')[1].strip().split(' ')[0])

# ...but we still need to convert titles to a number
title_map = {}
for i, v in enumerate(sorted(train_df.Title.unique())):  # this does allow invalid titles to be mapped
    title_map[v] = i
train_df['TitleNum'] = train_df.Title.map(title_map)

# Let's fill in missing Age values by Pclass and Title
num_class_values = len(train_df.Pclass.unique())
num_title_values = len(title_map)
train_df['AgeFill'] = train_df.Age
age_stats = np.zeros((num_class_values, num_title_values))
for c in range(num_class_values):
    default_age = train_df[train_df.Pclass == c + 1].Age.dropna().median()
    for t in range(num_title_values):
        age_stats[c, t] = train_df[(train_df.Pclass == c + 1) & (train_df.TitleNum == t)].Age.dropna().median()
        if np.isnan(age_stats[c, t]):
            age_stats[c, t] = default_age

        train_df.loc[(train_df.Age.isnull()) & (train_df.Pclass == c + 1) & (train_df.TitleNum == t), 'AgeFill'] = age_stats[c, t]

# We can also "engineer" new features by combining others
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch']


# With our training data prepped, we now load the test data
test_df = pd.read_csv('.\\data\\test.csv', header=0)

#test_df.info()
# Use loc[] to update DataFrame values in-place; for simplicity we'll replace missing Fares with the population avg Fare
test_df.loc[test_df.Fare.isnull(), 'Fare'] = train_df.Fare.mean()

# Apply Same transformations to SexNum, Title, TitleNum, and AgeFill
test_df['SexNum'] = test_df.Sex.map(sex_map).astype(int)
test_df['Title'] = test_df.Name.map(lambda x: x.split(',')[1].strip().split(' ')[0])
test_df['TitleNum'] = test_df.Title.map(title_map)
test_df['AgeFill'] = test_df.Age
for c in range(num_class_values):
    for t in range(num_title_values):
        test_df.loc[(test_df.Age.isnull()) & (test_df.Pclass == c + 1) & (test_df.TitleNum == t), 'AgeFill'] = age_stats[c, t]

# ...and FamilySize
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch']

# Lastly, if new Titles are present in the test set just set them to the most popular title by Sex
test_df.loc[(test_df.TitleNum.isnull()) & (test_df.Sex == 'female'), 'TitleNum'] = train_df[train_df.Sex == 'female'].TitleNum.value_counts().index[0]
test_df.loc[(test_df.TitleNum.isnull()) & (test_df.Sex == 'male'), 'TitleNum'] = train_df[train_df.Sex == 'male'].TitleNum.value_counts().index[0]

# Now we guess which numeric columns will produce the best results; drop the unnecessary columns from each set
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title', 'Age', 'SibSp', 'Parch'], axis=1)
# dropna() removes any row with NaN value and .values returns the data as an ndarray
train_data = train_df.dropna().values

# ...do the same thing to test_df
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Title', 'Age', 'SibSp', 'Parch'], axis=1)
test_data = test_df.dropna().values


# And FINALLY, we train our model and make predictions
# A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the
# dataset and use averaging to improve the predictive accuracy and control over-fitting
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier

# Create an RFC with 100 decision trees
forest = RandomForestClassifier(n_estimators=100)

# Build a forest of trees using:
#  X = the training data features (except PassengerId and Survived)
#  y = the training data outputs (just Survived)
forest = forest.fit(train_data[0::, 2::], train_data[0::, 1])

# Compute the predicted output (survival outcomes) for each test row's features (except PassengerId)
# In this process each tree votes for an outcome weighted by its probability estimates. The predicted class is the one
# with the highest mean probability estimate across all trees in the forest
output = forest.predict(test_data[0::, 1::])

# Combine PassengerId and predicted Survived values
result = pd.DataFrame([test_data[0::, 0], output], index=['PassengerId', 'Survived'], dtype='int').T
# ...and write them to a .csv file
result.to_csv('.\\data\\randomforestmodel.csv', index=False)
