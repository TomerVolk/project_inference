import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

list_of_features = ["Day of Week", "Road Type", "Junction Detail", "Junction Control",
                    "Light Conditions", "weather_conditions", "road_surface_conditions", "Urban or Rural area", "Vehicle Type",
                    "Vehicle Manoeuvre", "vehicle_location-restricted_lane", "Skidding and overturning",
                    "vehicle_leaving_carriageway", "1st Point of impact", "Sex of Driver", "age_of_driver", "age_of_vehicle",
                    "Car Passenger", "Casualty Type", "speed_limit", "number_of_vehicles"
                    ]

dummies = ["Day of Week", "Road Type", "Junction Detail", "Junction Control",
                    "Light Conditions", "weather_conditions", "Vehicle Type",
                    "Vehicle Manoeuvre", "vehicle_location-restricted_lane", "Skidding and overturning",
                    "vehicle_leaving_carriageway", "1st Point of impact",
                    "Car Passenger", "Casualty Type"]
dummies = [a.lower().replace(" ", "_") for a in dummies]

treatments = ["1st Point of impact", "car Passenger", "Sex of Driver", "road_surface_conditions"]
treatments = [a.lower().replace(" ", "_") for a in treatments]

hidden_confounders = [
                    "Special Conditions at Site", "Carriageway Hazards", "Towing and Articulation",
                      "Junction Location", "Hit Object in Carriageway", "Journey Purpose", "Vehicle Propulsion Code",
                      "engine_capacity_(cc)"
                      ]

post_treatment = ["Police Officer Attend", "Casualty Class"]

outcome = "Accident Severity"
outcome = outcome.lower().replace(" ", "_")
potential_outcome = "number_of_casualties"


def clean_dataset():
    lof = [a.lower().replace(" ", "_") for a in list_of_features]
    lof += [outcome.lower().replace(" ", "_"), 'number_of_casualties']
    print(len(lof))
    df = pd.read_csv("archive/Kaagle_Upload.csv")
    df = df[lof]

    df["road_surface_conditions"] = df["road_surface_conditions"].replace(to_replace=[i for i in range(3, 8)], value=None)
    df["road_surface_conditions"] = df["road_surface_conditions"].replace(to_replace=2, value=0)
    df["sex_of_driver"] = df["sex_of_driver"].replace(to_replace=3, value=None)
    df["sex_of_driver"] = df["sex_of_driver"].replace(to_replace=2, value=0)
    df["urban_or_rural_area"] = df["urban_or_rural_area"].replace(to_replace=3, value=None)
    df["urban_or_rural_area"] = df["urban_or_rural_area"].replace(to_replace=2, value=0)
    df["weather_conditions"] = df["weather_conditions"].replace(to_replace=9, value=None)

    df = df.replace(to_replace=-1, value=None)
    df = df.dropna()
    print(len(df))

    df[outcome] = df[outcome].replace(to_replace=2, value=1)
    df[outcome] = df[outcome].replace(to_replace=1, value=0)
    df[outcome] = df[outcome].replace(to_replace=3, value=1)
    df1 = df[df[outcome] == 0]
    df3 = df[df[outcome] == 1]
    df3 = df3.sample(n=60000)
    full_df = pd.concat([df1, df3])
    full_df.to_csv("full_data.csv")


def get_graphs(df: pd.DataFrame):
    # days of week bar plot
    severe_labels = [i+0.15 for i in range(1, 8)]
    not_severe_labels = [i-0.15 for i in range(1, 8)]

    severe_df = df[df[outcome] == 0]
    not_severe_df = df[df[outcome] == 1]
    severe_list = list(severe_df["day_of_week"])
    not_severe_list = list(not_severe_df["day_of_week"])

    severe_counter = Counter(severe_list)
    not_severe_counter = Counter(not_severe_list)
    severe_list = [severe_counter[i]/len(severe_list) for i in range(1, 8)]
    not_severe_list = [not_severe_counter[i]/len(not_severe_list) for i in range(1, 8)]

    plt.bar(severe_labels, severe_list, color="red", label="severe", width=0.3)
    plt.bar(not_severe_labels, not_severe_list, color="blue", label="not severe", width=0.3)
    plt.title("Days Distribution")
    plt.legend()
    plt.xticks(range(1, 8), ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"])
    plt.savefig("Days Distribution.jpeg")
    plt.show()

    # age box plots
    severe_age_of_driver = list(severe_df["age_of_driver"])
    severe_age_of_car = list(severe_df["age_of_vehicle"])
    not_severe_age_of_driver = list(not_severe_df["age_of_driver"])
    not_severe_age_of_car = list(not_severe_df["age_of_vehicle"])

    plt.boxplot([severe_age_of_driver, not_severe_age_of_driver])
    plt.title("driver_age_boxplot")
    plt.legend()
    plt.xticks(range(1, 3), ["severe", "not severe"])
    plt.savefig("driver_age_boxplot.jpeg")
    plt.show()

    plt.boxplot([severe_age_of_car, not_severe_age_of_car])
    plt.title("car_age_boxplot")
    plt.legend()
    plt.xticks(range(1, 3), ["severe", "not severe"])
    plt.savefig("car_age_boxplot.jpeg")
    plt.show()

    # severity hists
    severe_num = severe_df["number_of_casualties"]
    not_severe_num = not_severe_df["number_of_casualties"]
    plt.hist(severe_num, color="red", label="severe", alpha=0.5, bins=20)
    plt.hist(not_severe_num, color="blue", label="not severe", alpha=0.5, bins=20)
    plt.legend()
    plt.xlabel("number of casualties")
    plt.title("Casualties Distribution")
    plt.savefig("casualties.jpeg")
    plt.show()

    # speed limit
    severe_speed = list(severe_df["speed_limit"])
    not_severe_speed = list(not_severe_df["speed_limit"])

    severe_labels = [i + 1 for i in range(10, 80, 10)]
    not_severe_labels = [i - 1 for i in range(10, 80, 10)]

    severe_counter = Counter(severe_speed)
    not_severe_counter = Counter(not_severe_speed)
    severe_list = [severe_counter[i] / len(severe_speed) for i in range(10, 80, 10)]
    not_severe_list = [not_severe_counter[i] / len(not_severe_speed) for i in range(10, 80, 10)]

    plt.bar(severe_labels, severe_list, color="red", label="severe", width=2)
    plt.bar(not_severe_labels, not_severe_list, color="blue", label="not severe", width=2)
    plt.title("Speed Limit Distribution")
    plt.legend()
    # plt.xticks(range(1, 10), ["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"])
    plt.savefig("Speed Limit Distribution.jpeg")
    plt.show()


def numeric_data(df: pd.DataFrame):
    severe_df = df[df[outcome] == 0]
    not_severe_df = df[df[outcome] == 1]

    severe_area = list(severe_df["urban_or_rural_area"])
    not_severe_area = list(not_severe_df["urban_or_rural_area"])

    counter = 0
    for e in severe_area:
        if e == 1:
            counter += 1
    print(f"{counter/len(severe_area)*100}% of the severe accidents are urban")

    counter = 0
    for e in not_severe_area:
        if e == 1:
            counter += 1
    print(f"{counter/len(not_severe_area)*100}% of the not severe accidents are urban")

    # impact
    severe_impact = list(severe_df["1st_point_of_impact"])
    not_severe_impact = list(not_severe_df["1st_point_of_impact"])


    counter = 0
    for e in severe_impact:
        if e == 1:
            counter += 1
    print(f"{counter / len(severe_impact) * 100}% of the severe accidents are front impact")

    counter = 0
    for e in not_severe_impact:
        if e == 1:
            counter += 1
    print(f"{counter / len(not_severe_impact) * 100}% of the not severe accidents are front impact")

    # car passenger
    severe_passenger = list(severe_df["car_passenger"])
    not_severe_passenger = list(not_severe_df["car_passenger"])

    severe_counter = Counter(severe_passenger)
    not_severe_counter = Counter(not_severe_passenger)

    for key in severe_counter.keys():
        print(f"{severe_counter[key]/len(severe_passenger)*100}% of the severe accidents are {key}")
        print(f"{not_severe_counter[key] / len(not_severe_passenger) * 100}% of the not severe accidents are {key}")


if __name__ == '__main__':
    clean_dataset()
    df = pd.read_csv("full_data.csv")
    ls = df.speed_limit.unique()
    print(ls)
    get_graphs(df)
    numeric_data(df)
    pass


