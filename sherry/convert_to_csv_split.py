import json
import csv
import math

with open("../data/train.json") as file:
    data = json.load(file)

max_number_per_file = math.ceil(len(data) / 5)
print("Recipes per file: ", max_number_per_file)
number_of_recipes = 0
current_json = []

# Not label, just know that the columns are ["id", "cuisine", "ingredients"]
# output.writerow(data[0].keys())
for row in data:
    number_of_recipes = number_of_recipes + 1
    file_number = math.floor(number_of_recipes/max_number_per_file)
    if number_of_recipes % max_number_per_file == max_number_per_file - 1:
        with open("data/train_" + str(file_number) + ".json", "w") as outfile:
            json.dump(current_json, outfile)
        current_json = []
        print("Writing JSON: ", file_number)

    current_json.append(row)
    with open("data/train_" + str(file_number) + ".csv", "a") as file:
        output = csv.writer(file)
        for ingredient in row["ingredients"]:
            output.writerow([row["id"], row["cuisine"], ingredient])
