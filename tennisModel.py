import pandas as pd
import numpy as np
import openpyxl
import random

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier

df = pd.read_csv("df_atp.csv")


objs = df.select_dtypes("object").columns
print(", ".join(objs))



from collections import defaultdict

player_stats = defaultdict(lambda: [0, 0])

def calc_elo(stats):
    g, w = stats
    c = 10
    base_rating = 1500
    p = (w + c)/(g + 2*c)
    rating = base_rating + 400 *np.log10(p/(1-p))
    return rating

def calch2h(w, g):
    return (w + 1)/(g + 2)

# ensure columns exist
df["player0Elo"] = 0.0
df["player1Elo"] = 0.0
df["player0Rank"] = 0.0
df["player1Rank"]  = 0.0
df["player0Pts"] = 0.0
df["player1Pts"]  = 0.0
df["player0Max"] = 0.0
df["player1Max"] = 0.0
df["won"] = 0
df["h2hPlayer0"] = 0.0
df["h2hPlayer1"] = 0.0

random.seed(43)

h2h = {}
df["WRank"] = pd.to_numeric(df["WRank"].replace("NR", np.nan), errors="coerce")
df["LRank"] = pd.to_numeric(df["LRank"].replace("NR", np.nan), errors="coerce")

df["WRank"] = df["WRank"].astype("Int64")
df["LRank"] = df["LRank"].astype("Int64")

for i in range(len(df)):
    w = df.iloc[i, df.columns.get_loc("Winner")]
    l = df.iloc[i, df.columns.get_loc("Loser")]
    players = frozenset([w, l])

    if players not in h2h:
        h2h[players] = {w: 0, l: 0}
    

    w_elo = calc_elo(player_stats[w])
    l_elo = calc_elo(player_stats[l])

    wNum = random.randint(0, 1)
    lNum = 1 - wNum

    df.at[df.index[i], "h2hPlayer" + str(wNum)] = calch2h(h2h[players][w],h2h[players][w] + h2h[players][l])
    df.at[df.index[i], "h2hPlayer" + str(lNum)] = calch2h(h2h[players][l],h2h[players][w] + h2h[players][l])
    df.at[df.index[i], "player" + str(wNum) + "Elo"] = w_elo
    df.at[df.index[i], "player" + str(lNum) + "Elo"] = l_elo
    df.at[df.index[i], "player" + str(wNum) + "Rank"] = (df.iloc[i, df.columns.get_loc("WRank")])
    df.at[df.index[i], "player" + str(lNum) + "Rank"] = (df.iloc[i, df.columns.get_loc("LRank")])
    df.at[df.index[i], "player" + str(wNum) + "Pts"] = df.iloc[i, df.columns.get_loc("WPts")]
    df.at[df.index[i], "player" + str(lNum) + "Pts"] = df.iloc[i, df.columns.get_loc("LPts")]
    df.at[df.index[i], f"player{wNum}Max"] = df.iloc[i, df.columns.get_loc("MaxW")]
    df.at[df.index[i], f"player{lNum}Max"] = df.iloc[i, df.columns.get_loc("MaxL")]
    df.at[df.index[i], "won"] = wNum

    h2h[players][w] += 1

    player_stats[w][0] += 1
    player_stats[w][1] += 1
    player_stats[l][0] += 1



df[["Day", "Month", "Year"]] = df["Date"].str.split("/", expand=True).astype(int)

actualDf = df[["won", "player0Elo", "player1Elo", "Day", "Month", "Year",
                "Surface", "Location", "Series", "player0Rank", "player1Rank",
                "player0Pts", "player1Pts", "player0Max", "player1Max", "h2hPlayer0", "h2hPlayer1"]].copy()

actualDf = pd.get_dummies(actualDf, columns=["Location", "Series", "Surface"])

actualDf.head()

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import xgboost as xgb
from packaging import version
import numpy as np

from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, make_scorer
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import numpy as np
from packaging import version

import numpy as np, xgboost as xgb, sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from xgboost import XGBRegressor



# X = actualDf.drop(columns=["won"])
# y = actualDf["won"].astype(int)

# X_train, X_valid, y_train, y_valid = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# param_grid = {
#     'max_depth': [5, 6, 7],
#     'learning_rate': [0.1, 0.01, 0.05],
#     'n_estimators': [500, 1000, 1500]
# }

# xgb = XGBRegressor(n_estimators=1000, objective='binary:logistic', random_state=42)

# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# grid_search.fit(X_train, y_train)

# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Best score: {grid_search.best_score_}")

# best_model = grid_search.best_estimator_


# y_proba = best_model.predict_proba(X_test)[:, 1]

# from sklearn.metrics import roc_auc_score
# test_auc = roc_auc_score(y_test, y_proba)
# print("Test AUC:", test_auc)

# split once








X = actualDf.drop(columns=["won"])
y = actualDf["won"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from xgboost import XGBRegressor

model = XGBRegressor(
    max_depth=5,
    learning_rate=0.01,
    n_estimators=500,
    objective="reg:squarederror",  # regression
    random_state=42
)

model.fit(X_train, y_train)

# continuous predictions (e.g. 0.73 means 73% chance player1 wins)
y_pred_continuous = model.predict(X_test)

# threshold at 0.5 if you want a binary outcome
y_pred_binary = (y_pred_continuous >= 0.5).astype(int)

y_pred_continuous = model.predict(X_test)   # continuous ~ probability

print(y_pred_continuous)

auc = roc_auc_score(y_test, y_pred_continuous)
print("Regressor AUC:", auc)
print(X_test)
for i in X_test:
    print(i)
# newData = {
#     "player0Elo": calc_elo(player_stats["Medvedev D."]),
#     "player1Elo": calc_elo(player_stats["Zverev A."]),
#     "Day": 1,
#     "Month": 10,
#     "Year": 2020,
#     "Surface"

# }

# actualDf = df[["won", "player0Elo", "player1Elo", "Day", "Month", "Year",
#                 "Surface", "Location", "Series", "player0Rank", "player1Rank",
#                 "player0Pts", "player1Pts", "player0Max", "player1Max", "h2hPlayer0", "h2hPlayer1"]].copy()