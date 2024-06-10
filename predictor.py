import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

def main():
	train_data = pd.read_csv("data/train.csv");
	test_data = pd.read_csv("data/test.csv");

	y = train_data["Survived"];

	features = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
	X = pd.get_dummies(train_data[features]);
	X_test = pd.get_dummies(test_data[features]);

	model = RandomForestClassifier(n_estimators=100, max_depth=5, max_features="sqrt",random_state=1234);
	model.fit(X,y);
	predictions = model.predict(X_test)

	output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions});
	output.to_csv('data/submission.csv',index=False);
	print("This model scored: ",model.score(X,y)," on the training data");
	print("submission saved!");


if __name__ == '__main__':
	main()