import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

# Load the California housing dataset
data = fetch_california_housing(as_frame=True)

# Convert the dataset to a DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Display the first few rows of the dataset
print(df.head())

# Visualize the dataset (scatter plots, histograms, etc.)
# You can use Matplotlib, Seaborn, or any other visualization library.
