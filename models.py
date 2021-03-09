from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import csv
from preprocess import list_of_features, outcome, treatments, dummies
import pickle


def get_propensity_score(data: pd.DataFrame):
    treat = list(data["T"])
    cols = list(data.columns)
    cols.remove("T")
    cols.remove("Y")
    data = data[cols]
    data_points = []
    for index, row in data.iterrows():
        data_points.append(row)
    model = LogisticRegression(max_iter=1000)
    model.fit(data_points, treat)
    pred = model.predict_proba(data_points)
    pred = pred[:, 1].tolist()
    return model, pred


def calc_ipw(data, prop_score):
    treat_sum = 0
    y_sum = 0
    right_side_numerator = 0
    right_side_denominator = 0
    treat = list(data["T"])
    y_list = list(data["Y"])
    cols = list(data.columns)
    cols.remove("T")
    cols.remove("Y")
    data = data[cols]
    data_points = []
    for index, row in data.iterrows():
        data_points.append(row)
    for t, y, x in zip(treat, y_list, data_points):
        y_sum += t*y
        treat_sum += t
        p_list = prop_score.predict_proba([x])
        p_list = p_list[0]
        right_side_numerator += (1-t)*y*(p_list[1]/p_list[0])
        right_side_denominator += (1-t)*(p_list[1]/p_list[0])
    ipw = y_sum/treat_sum - right_side_numerator/right_side_denominator
    return ipw


def s_learner(data: pd.DataFrame):
    cols = list(data.columns)
    cols.remove("Y")
    points = data[cols]
    data_points = []
    for index, row in points.iterrows():
        data_points.append(list(row))
    y_list = list(data["Y"])
    model = LinearRegression()
    model.fit(data_points, y_list)
    treated = points[points["T"] == 1]
    zero_t = [0] * len(treated)
    counter_factual = treated.copy()
    counter_factual["T"] = zero_t
    treated = treated.to_numpy()
    counter_factual = counter_factual.to_numpy()
    treat_predict = model.predict(treated)
    not_treat_predict = model.predict(counter_factual)
    tp = sum(treat_predict)/len(treat_predict)
    ntp = sum(not_treat_predict)/len(not_treat_predict)
    return tp - ntp


def t_learner(data: pd.DataFrame):
    treated = data[data["T"] == 1]
    treated_y = list(treated["Y"])

    not_treated = data[data["T"] == 0]
    not_treated_y = list(not_treated["Y"])

    cols = list(treated.columns)
    cols.remove("Y")
    cols.remove("T")
    treated = treated[cols]
    not_treated = not_treated[cols]

    treat_model = LinearRegression()
    treat_model.fit(treated.to_numpy(), treated_y)
    not_treated_model = LinearRegression()
    not_treated_model.fit(not_treated.to_numpy(), not_treated_y)

    treat_model_predict = treat_model.predict(treated.to_numpy())
    not_treat_model_predict = not_treated_model.predict(treated.to_numpy())

    tp = sum(treat_model_predict)/len(treat_model_predict)
    ntp = sum(not_treat_model_predict)/len(not_treat_model_predict)

    return tp - ntp


def matching(data: pd.DataFrame):
    treated = data[data["T"] == 1]
    treated_y = list(treated["Y"])

    not_treated = data[data["T"] == 0]
    not_treated_y = list(not_treated["Y"])

    cols = list(treated.columns)
    cols.remove("Y")
    cols.remove("T")
    treated = treated[cols]
    not_treated = not_treated[cols]

    model = KNeighborsRegressor(n_neighbors=1)
    model.fit(not_treated, not_treated_y)

    att = 0
    for index, data_point in enumerate(treated.values):
        pred = model.predict([data_point])[0]
        ite = treated_y[index] - pred
        att += ite
    att = att/len(treated)
    return att
    pass


def calc_for_dataframe(data_name, outcome, treatment=None):
    data = pd.read_csv(data_name)
    cols_list = [a.lower().replace(" ", "_") for a in list_of_features] + [outcome]
    data = data[cols_list]
    dummy_cols = dummies
    if treatment in dummy_cols:
        dummy_cols.remove(treatment)
    data = pd.get_dummies(data, columns=dummies)

    if treatment == "1st_point_of_impact":
        data[treatment] = data[treatment].replace(to_replace=[0, 2, 3, 4], value=5)
        data[treatment] = data[treatment].replace(to_replace=1, value=0)
        data[treatment] = data[treatment].replace(to_replace=5, value=1)
    if treatment == "car_passenger":
        data[treatment] = data[treatment].replace(to_replace=[2, 1], value=2)
        data[treatment] = data[treatment].replace(to_replace=0, value=1)
        data[treatment] = data[treatment].replace(to_replace=2, value=0)

    data = data.rename(columns={treatment: "T", outcome: "Y"})

    for col in data.columns:
        if col not in ["Y", "T"]:
            col_lst = list(data[col])
            min_max_scaler = preprocessing.StandardScaler()
            col_lst = np.array(col_lst).reshape(-1, 1)
            scaled = min_max_scaler.fit_transform(col_lst)
            data[col] = scaled
    prop_score, propensity_list = get_propensity_score(data)
    ipw = calc_ipw(data, prop_score)
    print(ipw)
    s = s_learner(data)
    print(s)
    t = t_learner(data)
    print(t)
    # match = matching(data)
    # print(match)
    print("\n")
    # return [ipw, s, t, match], propensity_list
    return [ipw, s, t], propensity_list


if __name__ == '__main__':
    file = open("Basic_results.txt", "w")
    string_to_write = ""
    prop_dict = {}
    for treat in treatments:
        print(treat)
        ls, prop_list = calc_for_dataframe("full_data.csv", outcome, treat)
        string_to_write += treat + ": " + str(ls) + "\n\n"
        prop_dict[treat] = prop_list
    file.write(string_to_write)
    file.close()
    pickle.dump(prop_dict, open("prop_pickle.pkl", "wb"))
    pass


