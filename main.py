import joblib
import pandas as pd
import json

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate



def main():
    print('Scikit-learn models tuning and comparison')

    with open('data/grid_search_dict.json', 'r') as file:
        grid_search_dict = json.load(file)

    df = pd.read_csv('data/feed.csv')

    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1.0 if x > 0.0 else 0.0)

    train, test, target, target_test = train_test_split (X, y, test_size=.2, random_state=34)

    models = (
        LogisticRegression(),
        SGDClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        GradientBoostingClassifier(),
        ExtraTreesClassifier(),
        AdaBoostClassifier(),
        SVC(),
        GaussianNB(),
        MLPClassifier()
    )

    best_score = .0
    best_model = None
    for model in models:
        GS = GridSearchCV(
            estimator=model,
            param_grid=grid_search_dict[type(model).__name__],
            scoring='accuracy',
            cv=5,
            verbose=4
        )

        GS.fit(train, target)
        print (f'model: {type(model).__name__} best score{GS.best_score_} ')

        if GS.best_score_() > best_score:
            best_score = GS.best_score_()
            best_model = GS.best_estimator_

    print(f'best model: {type(best_model).__name__}, accuracy: {best_score:.4f}')

    best_model_trained = best_model.fit(X,y)
    joblib.dump(best_model_trained, 'best_model_trained.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()