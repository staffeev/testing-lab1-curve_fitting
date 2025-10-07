import pandas as pd
from source.solver import get_solutions


if __name__ == "__main__":
    data = pd.read_csv("ExampleData.csv", sep=";")
    x_data, y_data, z_data = data["X"][:100], data["Y"][:100], data["Z"][:100]
    print(get_solutions({}, x_data, y_data, None, use_only_max_dimension=True)[-10:])