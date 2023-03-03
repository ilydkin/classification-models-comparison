import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate


def main():
    print('Scikit-learn models tuning and comparison')

    df = pd.read_csv('data/feed.csv')

    X = df.drop('income', axis=1)
    y = df['income'].apply(lambda x: 1.0 if x > 0.0 else 0.0)

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
        score = cross_val_score (model, X, y, cv=5, scoring= 'accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')

        if score.mean() > best_score:
            best_score = score.mean()
            best_model = model

    print(f'best model: {type(best_model).__name__}, accuracy: {best_score:.4f}')
    # joblib.dump(best_pipe, 'loan_pipe.pkl')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()