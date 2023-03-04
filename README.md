# Scikit-learn models comparison and autotune

The project was born in a way of studying Scikit-learn library and came as an idea to automate search 
for the best estimator for a given dataset and problem. It occurred to me to loop tuning of 10 classification
models to find the best estimator in its best configuration, then fit and serialize it in one run.
And this as an alternative to manual work in Jupytor Notebook. So, this what this project is about.

### Models
- LogisticRegression(),
- SGDClassifier(),
- DecisionTreeClassifier(),
- RandomForestClassifier(),
- GradientBoostingClassifier(),
- ExtraTreesClassifier(),
- AdaBoostClassifier(),
- SVC(),
- GaussianNB(),
- MLPClassifier()

### EDA.ipynb
Exploratory data analysis. An almost clean data set describes several features correlated with the target parameter "income".
The "income" variable may take one of the two values - over 50K or less, which makes its prediction a problem of classification.
Check for ProfileReport, a powerful tool for EDA applied in the notebook: https://github.com/ydataai/ydata-profiling
Features standardization and encoding are also here, whereas the resulting 'feed.csv' file is a fully prepared feed for the models.

### grid_search.ipynb
This notebook is to make a dictionary with parameters for tuning the models.
The resulting dictionary is exported as 'grid_search_params.json'

### data/
- best_model_trained.pkl - the outcome product of main.py
- feed.csv - EDA.ipynb product
- grid_search_logs.csv - if you fancy to look at parameters beyond the best accuracy score (e.g. fit_time) 
and decide for yourself what your best configuration is.
- grid_search_params.json - grid_search.ipynb product
- info.log - best scores reached by each model, the very best model, its ultimate score and configuration
- raw_data.csv - this is where it all started.

### main.py
Run it and behold the magic