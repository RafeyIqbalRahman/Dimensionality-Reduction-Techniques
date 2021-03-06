from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA

# Load the data
df = read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/oil-spill.csv',
              header=None,)

# Show the first 5 rows of the data
df.head()

# Define X (Predictor variables) and y (Target variable)
dt = df.values
ix = [i for i in range(dt.shape[1]) if i != 49]
X, y = dt[:, ix], dt[:, 49]

# Dimensionality Reduction
pc = PCA(n_components=2)

# Fit and transform PCA on the dataset
Xtrans = pc.fit_transform(X)

# Convert NumPy array to Pandas DataFrame
Xtrans = DataFrame(data=Xtrans)

# Show the first 5 rows of the data with scaled values
Xtrans.head()
