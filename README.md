Principal Component Analysis (PCA) Implementation
This project implements Principal Component Analysis (PCA) from scratch using NumPy to reduce the dimensionality of a dataset. PCA is a statistical method used to simplify complex datasets while preserving as much variance as possible. It is widely applied in areas like machine learning, data analysis, and image processing.

Dataset
The project uses the Education in General.csv dataset, which contains features related to education indicators. Each feature represents a measurable attribute such as literacy rates, enrollment levels, or access to educational resources, which can help analyze patterns and relationships within the education sector.

Implementation Steps
The notebook walks through the full PCA process step by step: The project uses the Education in General.csv dataset, which contains features related to education indicators. Each feature represents a measurable attribute such as literacy rates, enrollment levels, or access to educational resources, which can help analyze patterns and relationships within the education sector.

1. Load and Standardize the Data
The dataset is loaded using numpy.genfromtxt.
Each feature is standardized by subtracting the mean and dividing by the standard deviation.
Standardization ensures all features contribute equally to the PCA, regardless of scale.
2. Calculate the Covariance Matrix
The covariance matrix is computed using numpy.cov.
It shows how features vary with respect to one another.
3. Perform Eigendecomposition
numpy.linalg.eig is used to find the eigenvalues and eigenvectors of the covariance matrix.
Eigenvectors represent the principal components (directions of maximum variance).
Eigenvalues represent how much variance is explained by each component.
4. Sort Principal Components
Eigenvectors are sorted in descending order of their eigenvalues.
The explained variance ratio is calculated to show the proportion of total variance captured by each component.
5. Project Data onto Principal Components
The standardized data is projected onto the top principal components (in this case, the first 4).
This step reduces the dataset’s dimensions while retaining most of the important information.
6. Output the Reduced Data
The shape of the reduced dataset confirms dimensionality reduction.
The first few rows are displayed to preview the transformed data.
7. Visualize Before and After PCA
Two scatter plots (using matplotlib.pyplot) visualize:
The original data (first two features)
The reduced data (first two principal components)
This visual comparison shows how PCA simplifies data while preserving structure and variance.
Dependencies
-NumPy -Matplotlib

-Ensure you have the heart.csv file in the same directory as the notebook or provide the correct path.

Author
Chinemerem Judith Ugbo
