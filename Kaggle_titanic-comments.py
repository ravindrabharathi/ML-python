import numpy as np
# Import 'tree' from scikit-learn library
from sklearn import tree
import pandas as pd
pd.options.mode.chained_assignment = None

#load data

train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

full_data = [train, test]

# Passengers that survived vs passengers that passed away
#print(train.Survived.value_counts())

# As proportions
#print(train.Survived.value_counts(normalize=True))

# Males that survived vs males that passed away
#print(train["Survived"][train["Sex"] == 'male'].value_counts())

# Females that survived vs Females that passed away
#print(train["Survived"][train["Sex"] == 'female'].value_counts())

# Normalized male survival
#print(train["Survived"][train["Sex"] == 'male'].value_counts(normalize=True))

# Normalized female survival
#print(train["Survived"][train["Sex"] == 'female'].value_counts(normalize=True))


#-----------------
train["Age"] = train["Age"].fillna(train["Age"].median())
# Create the column Child and assign to 'NaN'
train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
train['Child'][train['Age']<16]=1
train['Child'][train['Age']>=16]=0
#print(train['Child'])

# Print normalized Survival Rates for passengers under 18
#print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
#print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))

#----------------------------

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna('S')

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
#Print the Sex and Embarked columns
#print(train.Sex)
#print(train.Embarked)

#---------------------------------------

# Print the train data to see the available features
#print(train)

# Create the target and features numpy arrays: target, features_one
target = train['Survived'].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values
print(features_one)
# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

# Look at the importance and score of the included features
#print(my_tree_one.feature_importances_)
#print(my_tree_one.score(features_one, target))

#--------------------------------------------------------
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
# Impute the missing value with the median
test.Fare[152] = train.Fare.median()
test.Age = test.Age.fillna(test.Age.median())
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

test["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
test['Child'][test['Age']<16]=1
test['Child'][test['Age']>=16]=0

# has cabin ?

train['Has_Cabin'] = train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)
test['Has_Cabin'] = test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

#print(test.Sex)
# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Write your solution to a csv file with the name my_solution.csv
my_solution.to_csv("D://kaggle//titanic//my_solution_one.csv", index_label = ["PassengerId"])

#------------------------------------------------------------

# Create a new array with the added features: features_two
features_two = train[["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']].values

#Control overfitting by setting "max_depth" to 10 and "min_samples_split" to 5 : my_tree_two
max_depth = 10
min_samples_split =5
my_tree_two = tree.DecisionTreeClassifier(max_depth = max_depth, min_samples_split = min_samples_split, random_state = 1)
my_tree_two = my_tree_two.fit(features_two,target)

#Print the score of the new decison tree
#print(my_tree_two.score(features_two,target))

test_features_two = test[["Pclass","Age","Sex","Fare", 'SibSp', 'Parch', 'Embarked']].values
my_prediction_2=my_tree_two.predict(test_features_two)
my_solution_2 =pd.DataFrame(my_prediction_2,PassengerId,columns=['Survived'])
my_solution_2.to_csv("D://kaggle//titanic//my_solution_two.csv", index_label = ["PassengerId"])


#----------------------------------------------

# Create train_two with the newly defined feature
train_two = train.copy()
train_two["family_size"] = train_two['SibSp']+train_two['Parch']+1

# Create a new feature set and add the new feature
features_three = train_two[["Pclass", "Sex", "Age", "Fare", "Parch", 'family_size','Embarked']].values

# Define the tree classifier, then fit the model
my_tree_three = tree.DecisionTreeClassifier()
my_tree_three = my_tree_three.fit(features_three,target)

# Print the score of this decision tree
#print(my_tree_three.score(features_three, target))
test_3=test.copy()
test_3['family_size']=test_3['SibSp']+test_3['Parch']+1
#test_features_three = test_3[["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", 'family_size','Embarked','Child']].values
test_features_three = test_3[["Pclass", "Sex", "Age", "Fare", "Parch", 'family_size','Embarked']].values
predict3=my_tree_three.predict(test_features_three)
sol3=pd.DataFrame(predict3,PassengerId,columns=['Survived'])
sol3.to_csv("D://kaggle//titanic//my_solution_three.csv", index_label = ["PassengerId"])


#-----------------------

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
#my_forest = forest.fit(features_forest, target)

my_forest = forest.fit(features_three, target)

# Print the score of the fitted random forest
#print(my_forest.score(features_forest, target))

# Compute predictions on our test set features then print the length of the prediction vector
#test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
pred_forest = my_forest.predict(test_features_three)
print(len(pred_forest))
sol4=pd.DataFrame(pred_forest,PassengerId,columns=['Survived'])
sol4.to_csv("D://kaggle//titanic//my_solution_four.csv", index_label = ["PassengerId"])

#-------------------------------------------