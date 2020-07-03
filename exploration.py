#First import the training and testing data from kaggle. Note that the test dataset does not have a 'Survived' label column
import pandas as pd
train_df = pd.read_csv('/kaggle/input/titanic/train.csv')
test_df = pd.read_csv('/kaggle/input/titanic/test.csv')
print(train.head())
print(test.head())

#For simplicity moving forward, let's put the both datasets together
test_df['Survived'] = np.nan
full_df = train_df.append(test_df)

print(full_df.shape)
print(full_df.info())

#Beginning with nulls. Do we have any gaps in the data?
print(full_df.isna().sum())

#Nulls in Age, Fare, Cabin, and Embarked. What to do

#With only 2 nulls in Embarked we fill this with the most common value
full_df.Embarked.fillna(full_df.Embarked.value_counts(ascending = False)[0], inplace = True)

#Take a look at the correlation matrix to see what other variables are good indicators of Age and Fare
print(full_df.corr())

#notice the high correlation these features have with 'Pclass'
full_df['Age'] = full_df.groupby('Pclass')['Age'].transform(lambda x: x.fillna(x.median()))
full_df['Fare'] = full_df.groupby('Pclass')['Fare'].transform(lambda x: x.fillna(x.median()))

#Let's take the time to build some new features. Namely Family size, Cabin, and Title
full_df['Cabin'] = full_df.Cabin.str[0]
full_df.Cabin.fillna('U', inplace = True)

full_df['Family_size'] = full_df['Parch'] + full_df['SibSp'] + 1

import re
full_df['Title'] = full_df.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
full_df['Title'] = full_df.Title.replace(['Master', 'Dr', 'Rev', 'Col', 'Major', 'Ms', 'Mlle', 'Countess', 'Lady', 'Dona', 'Sir', 'Jonkheer', 'Don', 'Capt', 'Mme'], 'Other')

#With these new features on board we can drop those they were created from to avoid redundancy
full_df = full_df.drop(['Name', 'Parch', 'SibSp'], axis = 1)

#Now to deal with skewed distributions. Observe histograms of the numerical features.
import matplotlib.pyplot as plt
plt.hist(full_df.Fare)
plt.show()
plt.hist(full_df.Age)
plt.show()
plt.hist(full_df.Family_size)
plt.show()
plt.hist(full_df.Pclass)
plt.show()

#Fare, Age, and Family size are all quite skewed. Since none of these features have negative values we will use a log transform to deal with distributions
full_df['Fare_log'] = np.log(full_df.Fare+1)
full_df['Age_log'] = np.log(full_df.Age)
full_df['Family_size'] = np.log(full_df.Family_size)

#Again we will drop redundant columns and split back into the training and testing data
full_df = full_df.drop(['Fare', 'Age', 'Family_size'], axis = 1)
train_df = full_df[full_df['Survived'].notna()]
test_df = full_df[full_df['Survived'].isna()]
