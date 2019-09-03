import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, RFE, SelectFromModel
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


player_df = pd.read_csv('data/fifa19_player.csv')

numcols = ['Overall', 'Crossing', 'Finishing',  'ShortPassing',  'Dribbling', 'LongPassing', 'BallControl',
           'Acceleration', 'SprintSpeed', 'Agility',  'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
           'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']
player_df = player_df[numcols+catcols]

traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
features = traindf.columns
traindf = traindf.dropna()
traindf = pd.DataFrame(traindf, columns=features)
y = traindf['Overall'] >= 87
X = traindf.copy()
del X['Overall']
# print(X.head())
# print(len(X.columns))

feature_name = list(X.columns)
num_feats = 30


# Pearson correlation
def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature


X_norm = MinMaxScaler().fit_transform(X)

# Pearson
cor_support, cor_feature = cor_selector(X, y, num_feats)


## Chi-Square Features
chi_selector = SelectKBest(chi2, k=num_feats)
chi_selector.fit(X_norm, y)
chi_support = chi_selector.get_support()
chi_feature = X.loc[:, chi_support].columns.tolist()


## Recursive Feature Elimination
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats,
                   step=10, verbose=5)
rfe_selector.fit(X_norm, y)
rfe_support = rfe_selector.get_support()
rfe_feature = X.loc[:, rfe_support].columns.tolist()


## Lasso: SelectFromModel
embeded_lr_selector = SelectFromModel(LogisticRegression(penalty='l1'), max_features=num_feats)
embeded_lr_selector.fit(X_norm, y)
embeded_lr_support = embeded_lr_selector.get_support()
embeded_lr_feature = X.loc[:, embeded_lr_support].columns.tolist()


## Tree-based: SelectFromModel
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
embeded_rf_selector.fit(X, y)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_feature = X.loc[:, embeded_rf_support].columns.tolist()


feature_selection_df = pd.DataFrame({'Feature': feature_name, 'Pearson': cor_support,
                                     'Chi-2': chi_support,
                                     'RFE': rfe_support,
                                     'Logistics': embeded_lr_support,
                                     'Random Forest': embeded_rf_support})
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
print(feature_selection_df.head(num_feats))
