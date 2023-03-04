import pandas as pd
import joblib
import json
import logging

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split, GridSearchCV


def main():
    print('Scikit-learn models GridSearchCV tuning and comparison')

    with open('data/grid_search_params.json', 'r') as file:
        grid_search_params = json.load(file)

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
    best_params = None
    df_grid_search_logs = pd.DataFrame()
    logging.basicConfig(filename='data/info.log', encoding='utf-8', level=logging.INFO)

    for model in models:
        GS = GridSearchCV(
            estimator=model,
            param_grid=grid_search_params[type(model).__name__],
            scoring='accuracy',
            cv=5,
            verbose=4
        )

        GS.fit(train, target)
        df_logs = pd.DataFrame(GS.cv_results_)
        df_grid_search_logs = pd.concat([df_grid_search_logs, df_logs])

        logging.info (f'model: {type(model).__name__} best score: {GS.best_score_}')

        if GS.best_score_ > best_score:
            best_score = GS.best_score_
            best_model = GS.best_estimator_
            best_params = GS.best_params_

    logging.info(f'best model: {type(best_model).__name__}, '
                 f'accuracy: {best_score:.4f}'
                 f'best parameters: {best_params}')

    df_grid_search_logs.to_csv('data/grid_search_logs.csv')
    best_model.fit(X,y)
    joblib.dump(best_model,'data/best_model_trained.pkl')


if __name__ == '__main__':
    main()