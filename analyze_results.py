import ast
import pandas as pd
from matplotlib import pyplot as plt


def get_data_from_txt(txt_path):
    data_ate = {}
    data_att = {}
    experiment_lst = ["IPW", "S Learner", "Inception", "T_Learner", "Matching"]
    pretty_treat = {'1st_point_of_impact': "1st Point \n Of Impact", 'car_passenger': "Car \n Passenger",
                    'sex_of_driver': "Sex of \n Driver", 'road_surface_conditions': "Road Surface\nConditions"}
    with open(txt_path, "r") as f:
        for line in f:
            if len(line.strip()) == 0:
                continue
            treat, res = line.split(":")
            treat = treat.strip()
            att, ate, _ = res.split("]")
            ate = ate.strip() + "]"
            att = att.strip() + "]"
            ate = ast.literal_eval(ate)
            att = ast.literal_eval(att)
            ate = [round(x, 3) for x in ate]
            att = [round(x, 3) for x in att]
            ate = {exp: val for exp, val in zip(experiment_lst, ate)}
            att = {exp: val for exp, val in zip(experiment_lst, att)}
            treat = pretty_treat[treat]
            data_ate[treat] = ate
            data_att[treat] = att

    data_ate = pd.DataFrame.from_dict(data_ate, orient="index")
    data_att = pd.DataFrame.from_dict(data_att, orient="index")

    data_ate.plot.bar(rot=0)
    plt.ylim([-0.1, 0.15])
    plt.title("ATE Of Different Treatments")
    plt.grid(axis='y')
    plt.savefig("graphs/ate_graph.png")
    plt.show()

    data_att.plot.bar(rot=0)
    plt.ylim([-0.1, 0.15])
    plt.title("ATT Of Different Treatments")
    plt.grid(axis='y')
    plt.savefig("graphs/att_graph.png")

    plt.show()


if __name__ == '__main__':
    get_data_from_txt("Basic_results.txt")
