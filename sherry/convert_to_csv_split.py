import json
import csv

with open("../data/train.json") as file:
    data = json.load(file)

with open("data/train.csv", "w") as file:
    output = csv.writer(file)

    output.writerow(data[0].keys())

    for row in data:
        for ingredient in row["ingredients"]:
            output.writerow([row["id"], row["cuisine"], ingredient])
