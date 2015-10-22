__author__ = 'Chad'
import csv as csv
import numpy as np
# NumPy is short for "Numerical Python". It's the fundamental package required for high
# performance scientific computing and data analysis

training_file_obj = csv.reader(open('.\\data\\train.csv', 'rU'))
header = next(training_file_obj)
print(header)

# Load training rows into Python array
training_data = []
for r in training_file_obj:
    training_data.append(r)

# Convert Python array to a numpy ndarray
# ndarray is a fast and space-efficient multidimensional array providing vectorized arithmetic operations and
# sophisticated broadcasting capabilities. Also, provides tools for integrating with C, C++, and Fortran
training_data = np.array(training_data)

# Survived is the 2nd column; Compute the survival rate for the training set; convert from string to float
passenger_count = np.size(training_data[0::, 1].astype(np.float))
survivor_count = np.sum(training_data[0::, 1].astype(np.float))
survival_rate = survivor_count / passenger_count

# Create a bool indexes for the Sex column (5th column)
female_index = training_data[0::, 4] == 'female'
male_index = training_data[0::, 4] != 'female'

# Lookup the Survived (2nd column) values by gender
female_survival = training_data[female_index, 1].astype(np.float)
male_survival = training_data[male_index, 1].astype(np.float)

female_survival_rate = np.sum(female_survival) / np.size(female_survival)
male_survival_rate = np.sum(male_survival) / np.size(male_survival)

print('Overall Survival Rate:  %s' % survival_rate)
print(' Female Survival Rate:  %s' % female_survival_rate)
print('   Male Survival Rate:  %s' % male_survival_rate)


print('--  Let''s make some predictions using a simple gender heuristic ------')
# Open the test data file
test_file_obj = csv.reader(open('.\\data\\test.csv', 'rU'))
header = next(test_file_obj)
print(header)

# Create a file to store our predictions
prediction_file_obj = csv.writer(open('.\\data\\gendermodel.csv', 'w', newline = ''))

# Write the file headers expected by Kaggle
prediction_file_obj.writerow(['PassengerId', 'Survived'])

# For each test record write the PassengerId and 1 (survived) for females, otherwise 0 (didn't survived) for males
for r in test_file_obj:
    if r[3] == 'female':
        prediction_file_obj.writerow([r[0], '1'])
    else:
        prediction_file_obj.writerow([r[0], '0'])


print('-- Add some sophistication with a survival table ---------------------------------------------------')
# We'll compute the average survival rate for all combinations of gender, class, and fare
# Let's start by cleaning up the source data some; first sex (5th column) values should be numeric
training_data[female_index, 4] = 0
training_data[male_index, 4] = 1

# Then enforce a max fare (10th column) value and remove outliers...
fare_ceiling = 40
outlier_index = training_data[0::, 9].astype(np.float) >= fare_ceiling
training_data[outlier_index, 9] = fare_ceiling - 1

# ...and bucketize (binning) fares to group similar fare values
fare_bucket_size = 10
number_of_fare_buckets = int(fare_ceiling / fare_bucket_size)
training_data[0::, 9] = (training_data[0::, 9].astype(np.float) / fare_bucket_size).astype(np.int)

# Initialize the survival table
number_of_genders = len(np.unique(training_data[0::, 4]))
number_of_classes = len(np.unique(training_data[0::, 2]))
survival_table = np.zeros((number_of_genders, number_of_classes, number_of_fare_buckets))
print(survival_table)

print('-- Compute statistics for survival_table --------------------------------------------------------')
for g in range(number_of_genders):
    for c in range(number_of_classes):
        for f in range(number_of_fare_buckets):
            mask = ((training_data[0::, 4].astype(np.float) == g) &
                (training_data[0::, 2].astype(np.float) == c + 1) &  # +1 because PClass is 1-3, and survival table is 0-2
                (training_data[0::, 9].astype(np.float) == f))

            stats = training_data[mask, 1]

            survival_table[g, c, f] = np.mean(stats.astype(np.float))

print(survival_table)

# Remove NaN values
survival_table[survival_table != survival_table] = 0
print(survival_table)

# Predict survival for any category of passenger with 50% or better survival rate
survival_table[survival_table < 0.5] = 0
survival_table[survival_table >= 0.5] = 1

print(survival_table)


print('--  Write out the predictions with this new model ------------------------------------------')
# Open the test data file
test_file_obj = csv.reader(open('.\\data\\test.csv', 'rU'))
header = next(test_file_obj)

# Create a file to store our predictions
prediction_file_obj = csv.writer(open('.\\data\\genderclassfaremodel.csv', 'w', newline = ''))

# Write the file headers expected by Kaggle
prediction_file_obj.writerow(['PassengerId', 'Survived'])

# For each test record write the PassengerId and 1 (survived) for females, otherwise 0 (didn't survived) for males
for r in test_file_obj:
    #print(r)
    if r[3] == 'female':
        g = 0
    else:
        g = 1

    c = int(r[1]) - 1
    f = r[8]
    if f == '':
        f = np.mean(training_data[0::, 9].astype(np.float))  # assume avg fare if fare is empty
    else:
        f = float(f)

    if f > fare_ceiling:
        f = fare_ceiling - 1
    f = int(f / fare_bucket_size)

    # Perform the survival lookup
    survived = survival_table[g, c, f]
    prediction_file_obj.writerow([r[0], int(survived)])