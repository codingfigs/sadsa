# SADSA - Software Application for Data Science and Analytics

**Version:** Full Version  
**Developer:** Dr. M. Kamakshaiah / AMCHIK SOLUTIONS  
**Website:** [http://codingfigs.com](http://codingfigs.com)  
**Contact:** contact@codingfigs.com

---

## Overview

SADSA (Software Application for Data Science and Analytics) is a comprehensive desktop application for statistical analysis, data visualization, and machine learning. Built with Python, SADSA provides an intuitive graphical interface for performing advanced analytics without writing code.

---

## Installation Instructions

### The open source alternative to this is available at https://github.com/codingfigs/sadsa-os

### Prerequisites
- Windows 10 or later (64-bit)
- Administrator privileges required for installation

### Step-by-Step Installation

1. **Download the Installer**
   - Download `sadsa.exe` from the releases section of this repository
   - Save the file to your Downloads folder or preferred location

2. **Run as Administrator**
   - Right-click on `sadsa.exe`
   - Select "Run as administrator" from the context menu
   - Click "Yes" when prompted by User Account Control (UAC)

3. **Follow Installation Wizard**
   - Click "Next" on the welcome screen
   - Read and accept the License Agreement
   - Choose installation directory (default: `C:\Program Files\SADSA`)
   - Select Start Menu folder (default: SADSA)
   - Click "Install" to begin installation

4. **Complete Installation**
   - Wait for files to be copied (approximately 1-2 minutes)
   - Click "Finish" to complete installation
   - A desktop shortcut will be created automatically

5. **First Launch**
   - Double-click the SADSA icon on your desktop
   - The application will initialize (first launch may take 10-15 seconds)
   - You're ready to start analyzing data!

### Troubleshooting Installation
- **Permission Error:** Ensure you run the installer as administrator
- **Antivirus Warning:** Add SADSA to your antivirus exceptions list
- **Installation Fails:** Check that you have at least 500MB free disk space
- **Cannot Launch:** Install Microsoft Visual C++ Redistributable (included with installer)

---

## Features & Menu Structure

### File Menu

#### Open Data File
- **Description:** Import data from various file formats
- **Supported Formats:** CSV, Excel (.xlsx, .xls), JSON, TSV, Parquet, LibreOffice
- **Usage:** Select file → Data loads into grid view

#### Input Data File
- **Description:** Alternative data import with advanced options
- **Features:** Column selection, data type inference, delimiter options

#### Save Data
- **Description:** Export current dataset to file
- **Formats:** CSV, Excel, JSON

#### Data View
- **Description:** Interactive spreadsheet view of loaded data
- **Features:** Sort, filter, edit cells, column operations

---
## 💻 **Python Console**
- Interactive Python shell integrated into the application.
- Direct access to loaded DataFrame via `df` variable.
- Execute custom Python code without leaving the application.
- Full access to all data analysis libraries (pandas, numpy, scipy, scikit-learn, etc.).
- Multi-line code support with Ctrl+Return execution.
- Call any app function directly from the console.
- Perfect for exploratory data analysis and custom operations.

### Console Methods & Features

#### Core Methods
- **`__init__(parent, app_instance, title="Python Console")`** - Initializes the Python console window with parent window reference and application instance.
- **`setup_ui()`** - Creates the console user interface with output area, input area, buttons, and information panel.
- **`setup_console_environment()`** - Sets up the Python environment with pre-imported libraries and displays welcome message with available objects and examples.

#### Code Execution Methods
- **`execute_code(event=None)`** - Executes Python code from input area with error handling and output capture via Ctrl+Return or Execute button.
- **`_import_module(module_name)`** - Imports a module by name and handles import errors gracefully.

#### Output & Display Methods
- **`print_output(text, tag="info")`** - Prints text to the output area with color coding (info=blue, error=red, success=green).
- **`clear_output()`** - Clears all text from the output display area.
- **`clear_input()`** - Clears all text from the code input area.
- **`show_help()`** - Displays a comprehensive help window with available objects, commands, and usage examples.

#### Pre-imported Libraries & Objects
- **`df`** - Current DataFrame from your loaded data, accessible for analysis and manipulation.
- **`app`** - Reference to the SADSA application instance for accessing app methods and data.
- **`pd`** - Pandas library for data manipulation and analysis.
- **`np`** - NumPy library for numerical computations and array operations.
- **`plt`** - Matplotlib.pyplot library for creating plots and visualizations.
- **`sns`** - Seaborn library for statistical data visualization.
- **`stats`** - SciPy.stats library for statistical functions and distributions.

##   Python Console

SADSA includes an **interactive Python console** for advanced users who want to perform custom data analysis, write custom code, or explore data interactively.

### How to Access
- Navigate to **Help** → **Python Console** from the menu bar
- A new window will open with the Python shell

### Key Features
- **Full Python Environment**: Execute any Python code directly
- **DataFrame Access**: Access your loaded data via the `df` variable
- **App Integration**: Reference the application via `app` to call any app function
- **Pre-imported Libraries**: 
  - `pandas` (pd)
  - `numpy` (np)
  - `scipy`, `matplotlib`, `scikit-learn`
  - `statsmodels`, `networkx` and more

### Common Operations
```python
# View data
df.head()
df.describe()
df.info()

# Data manipulation
df['new_col'] = df['col1'] + df['col2']
df_filtered = df[df['col'] > 100]
df_grouped = df.groupby('category').sum()

# Statistical analysis
correlation = df.corr()
mean_val = df['col'].mean()
std_val = df['col'].std()

# Call app functions from console
app.perform_correlation_analysis()  # Performs correlation analysis on the dataset
app.data  # Access current data from app

# File Operations
app.open_csv()                       # Opens a file dialog to load CSV data
app.save_data()                      # Saves current data in memory
app.save_data_file()                 # Saves data to a CSV file
app.clear_data()                     # Clears all loaded data
app.display_data()                   # Refreshes the data display in the grid

# Edit Menu Operations
app.rename_columns()                 # Renames one or more columns in the dataset
app.compute_variable()               # Creates new computed variables from existing columns
app.recode_variable()                # Recodes/transforms existing variables
app.missing_data_treatment()         # Handles missing values using various methods
app.set_values()                     # Sets fixed values in the dataset

# Transformations Menu
app.data_simulations()               # Generates simulated/synthetic data
app.generate_multivariate_normal()   # Generates multivariate normal distribution data
app.perform_decomposition("cholesky") # Performs Cholesky decomposition on numeric data
app.perform_decomposition("qr")      # Performs QR decomposition on numeric data
app.perform_decomposition("svd")     # Performs Singular Value Decomposition (SVD)
app.perform_decomposition("eig")     # Performs Eigenvalue decomposition
app.perform_standardization("Min-Max") # Applies Min-Max scaling to data
app.perform_standardization("Z-Score") # Applies Z-Score standardization to data
app.perform_standardization("Decimal Scaling") # Applies Decimal Scaling transformation
app.perform_standardization("Log")   # Applies Log transformation to data
app.perform_standardization("Log-Normal") # Applies Log-Normal transformation

# Data Analytics - Descriptive Statistics
app.show_frequencies()               # Displays frequency tables for categorical variables
app.show_summary_statistics()        # Shows summary statistics (mean, std, min, max, etc.)
app.show_about_dataset()             # Displays dataset metadata (rows, columns, types, missing values)

# Data Analytics - Inferential Statistics
app.perform_ttest()                  # Performs t-test (one-sample, two-sample, paired)
app.perform_chisquare()              # Performs Chi-Square test for independence
app.perform_normality_tests()        # Tests for data normality (Shapiro-Wilk, Kolmogorov-Smirnov)
app.perform_anova()                  # Performs ANOVA (Analysis of Variance) test
app.perform_manova()                 # Performs MANOVA (Multivariate ANOVA) test

# Data Analytics - Factor Analysis
app.perform_efa()                    # Performs Exploratory Factor Analysis
app.perform_cfa()                    # Performs Confirmatory Factor Analysis

# Data Analytics - Correlation & Regression
app.perform_regression_analysis()    # Performs linear and multiple regression analysis

# Data Analytics - Cluster Analysis
app.perform_kmeans()                 # Performs K-Means clustering on the data
app.perform_hierarchical()           # Performs Hierarchical clustering on the data

# Data Analytics - Time Series Analysis
app.perform_stationarity_tests()     # Tests for stationarity in time series data
app.perform_seasonal_decomposition() # Decomposes time series into trend, seasonal, and residual
app.perform_holt_winters()           # Applies Holt-Winters exponential smoothing
app.perform_moving_averages()        # Calculates moving averages for time series smoothing

# Machine Learning Models
app.perform_logistic_regression()    # Builds logistic regression classification model
app.perform_decision_tree()          # Builds decision tree classification model
app.perform_random_forest()          # Builds random forest classification/regression model
app.perform_naive_bayes()            # Builds Naive Bayes classification model
app.perform_knn()                    # Builds K-Nearest Neighbors classification model
app.perform_svm()                    # Builds Support Vector Machine classification model
app.perform_neural_network()         # Builds neural network classification/regression model

# Visualization & Display
app.generate_plot()                  # Opens plot generation interface with various plot types
app.show_report_window(data, title)  # Displays data/results in a report window
app.show_message(title, message)     # Shows a message dialog with custom title and content

# Console & Help
app.open_python_console()            # Opens the interactive Python console window
```

### Multi-line Code
- Write multiple lines of code
- Press **Ctrl+Return** to execute
- Use proper Python indentation for blocks

---

### Edit Menu

#### Copy
- **Description:** Copy selected cells to clipboard
- **Shortcut:** Ctrl+C

#### Paste
- **Description:** Paste data from clipboard into grid
- **Shortcut:** Ctrl+V

#### Delete Row
- **Description:** Remove selected row(s) from dataset
- **Usage:** Select row → Delete → Confirm

#### Add Column
- **Description:** Insert new column with specified name and default value
- **Options:** Column name, data type, position

---

### Transformations Menu

#### Standardization
- **Z-Score Standardization**
  - Transforms data to mean=0, std=1
  - Use for: Comparing variables on different scales
  
- **Min-Max Normalization**
  - Scales data to range [0, 1]
  - Use for: Neural networks, distance-based algorithms

- **Robust Scaling**
  - Uses median and IQR (resistant to outliers)
  - Use for: Data with extreme values

#### Data Generation
- **Multivariate Normal Distribution**
  - Generate synthetic data from normal distribution
  - Options: Variable names, means, covariance matrix, sample size

#### Matrix Decomposition
- **Cholesky Decomposition**
  - Decomposes positive-definite matrix
  - Use for: Linear algebra operations, optimization

- **QR Decomposition**
  - Orthogonal-triangular factorization
  - Use for: Solving linear systems, eigenvalue problems

- **SVD (Singular Value Decomposition)**
  - Decomposes matrix into U, Σ, V components
  - Use for: Dimensionality reduction, recommender systems

- **Eigenvalue Decomposition**
  - Computes eigenvalues and eigenvectors
  - Use for: PCA, spectral analysis

---

### Data Analytics Menu

#### Descriptive Statistics

##### Univariate Statistics
- **Description:** Summary statistics for single variables
- **Outputs:** Mean, median, mode, std dev, variance, skewness, kurtosis, min, max, quartiles
- **Plots:** Histogram, box plot, Q-Q plot

##### Bivariate Statistics
- **Description:** Relationships between two variables
- **Outputs:** Cross-tabulation, correlation, contingency tables
- **Plots:** Scatter plots, grouped bar charts

##### Multivariate Statistics
- **Description:** Analysis of multiple variables simultaneously
- **Outputs:** Correlation matrix, covariance matrix, partial correlations
- **Plots:** Correlation heatmap, pair plot, 3D scatter

#### Inferential Statistics

##### T-Test
- **Independent Samples T-Test**
  - Compares means of two independent groups
  - Outputs: T-statistic, p-value, confidence interval
  - Plot: Box plots comparing groups

- **Paired Samples T-Test**
  - Compares means of related groups
  - Use for: Before/after studies, matched pairs

- **One-Sample T-Test**
  - Tests if sample mean differs from population mean
  - Outputs: T-statistic, p-value

##### Chi-Square Test
- **Pearson Chi-Square**
  - Tests independence in contingency tables
  - Outputs: χ² statistic, p-value, degrees of freedom

- **Fisher's Exact Test**
  - Exact test for 2×2 tables (small sample sizes)
  - Use for: Small cell counts (<5)

- **McNemar's Test**
  - Tests marginal homogeneity (paired data)
  - Use for: Before/after categorical data

- **Yates' Correction**
  - Continuity correction for chi-square
  - Use for: 2×2 tables with small samples

- **Likelihood Ratio Chi-Square**
  - Alternative to Pearson chi-square
  - Better for small expected frequencies

- **Mantel-Haenszel Chi-Square**
  - Tests association controlling for confounders
  - Use for: Stratified 2×2 tables

- **G-Test (Alternative)**
  - Likelihood ratio test for independence
  - More accurate for small samples

##### Normality Tests
- **Shapiro-Wilk Test**
  - Most powerful normality test
  - Best for: Sample sizes < 2000

- **Kolmogorov-Smirnov Test**
  - Tests fit to any distribution
  - Use for: Large samples, known distribution

- **Anderson-Darling Test**
  - Gives more weight to tails
  - Use for: Detecting outliers

- **D'Agostino K² Test**
  - Based on skewness and kurtosis
  - Use for: Larger samples (n > 20)

- **Jarque-Bera Test**
  - Tests skewness and kurtosis
  - Use for: Financial data, regression residuals

- **Lilliefors Test**
  - Modified K-S test (parameters estimated)
  - Use for: When parameters unknown

##### ANOVA (Analysis of Variance)
- **Description:** Compares means across multiple groups
- **Outputs:** F-statistic, p-value, group means, post-hoc tests
- **Assumptions:** Normality, homogeneity of variance
- **Plot:** Box plots, means plot

##### MANOVA (Multivariate ANOVA)
- **Description:** Tests differences on multiple dependent variables
- **Outputs:** Wilks' Lambda, Pillai's trace, Roy's largest root
- **Use for:** Multiple outcome variables

#### Exploratory Analysis

##### Correspondence Analysis
- **Simple Correspondence Analysis**
  - Analyzes relationships in contingency tables
  - Outputs: Row/column coordinates, inertia, chi-square
  - Plot: Biplot of rows and columns

- **Multiple Correspondence Analysis**
  - Extension for multiple categorical variables
  - Outputs: Variable coordinates, contributions
  - Plot: Category space visualization

- **Canonical Correspondence Analysis**
  - Relates two sets of variables
  - Outputs: Canonical correlations, coefficients
  - Plot: Canonical variates plot

##### Multidimensional Scaling (MDS)
- **Description:** Visualizes dissimilarities in low dimensions
- **Options:** 
  - Metric MDS (preserves distances)
  - Non-metric MDS (preserves rank order)
- **Distance Metrics:** Euclidean, Manhattan, Cosine, Correlation
- **Outputs:** Stress, RSQ, coordinates
- **Plot:** 2D/3D point plot

##### Principal Components Analysis (PCA)
- **Description:** Reduces dimensionality while preserving variance
- **Outputs:** 
  - Principal components
  - Eigenvalues
  - Explained variance ratio
  - Component loadings
- **Plots:** Scree plot, biplot, component loadings

#### Correlation & Regression

##### Correlation Analysis
- **Pearson Correlation**
  - Linear relationship between continuous variables
  - Range: -1 to +1
  - Outputs: Correlation coefficient, p-value

- **Spearman Correlation**
  - Monotonic relationship (rank-based)
  - Use for: Non-linear relationships, ordinal data

- **Kendall Correlation**
  - Rank correlation (tau)
  - More robust than Spearman for small samples

- **Canonical Correlation**
  - Relationship between two sets of variables
  - Outputs: Canonical correlations, variates
  - Plot: Canonical variates

**Options:**
- P-values and significance testing
- Confidence intervals
- Explained variance

##### Covariance Analysis
- **Description:** Analyzes variance-covariance matrices
- **Tests Available:**
  - Box M Test (homogeneity of covariance matrices)
  - Bartlett's Test (sphericity)
  - Levene's Test (homogeneity of variance)
  - Permutation Test (non-parametric)

- **Fit Measures:**
  - Covariance Matrix
  - Eigenvalues
  - Determinant
  - Trace
  - Condition Number

##### Regression Analysis
- **Simple Linear Regression**
  - One predictor, one outcome
  - Outputs: Coefficients, R², p-values, residuals
  - Plot: Scatter with regression line

- **Multiple Linear Regression**
  - Multiple predictors
  - Outputs: Coefficients, adjusted R², F-statistic
  - Plot: Residual plots, Q-Q plot

- **Generalized Linear Model (GLM)**
  - Non-normal outcomes (binomial, Poisson, etc.)
  - Outputs: Coefficients, deviance, AIC
  - Link functions: Logit, log, identity

#### Factor Analysis

##### Exploratory Factor Analysis (EFA)
- **Description:** Identifies latent factors
- **Tests:**
  - Bartlett's Test of Sphericity
  - Kaiser-Meyer-Olkin (KMO) test
- **Outputs:**
  - Factor loadings
  - Communalities
  - Eigenvalues
- **Rotation:** Varimax, Promax
- **Plots:** Scree plot, factor loadings heatmap

##### Confirmatory Factor Analysis (CFA)
- **Description:** Tests hypothesized factor structure
- **Features:**
  - Specify factor structure
  - Define relationships between factors
  - Estimate raw or standardized coefficients
- **Outputs:**
  - Factor loadings
  - Fit indices (CFI, TLI, RMSEA, SRMR)
  - Modification indices
- **Plots:** Path diagram with estimates

##### Mediation Analysis
- **Description:** Tests indirect effects through mediator
- **Model:** X → M → Y
- **Outputs:**
  - Direct effects
  - Indirect effects
  - Total effects
  - Sobel test
- **Plot:** Mediation model diagram

#### Cluster Analysis

##### K-Means Clustering
- **Description:** Partitions data into K clusters
- **Options:**
  - Number of clusters (k)
  - Initialization method
  - Max iterations
- **Outputs:**
  - Cluster assignments
  - Cluster centers
  - Within-cluster sum of squares
  - Silhouette score
- **Plot:** Cluster scatter plot, elbow plot

##### Hierarchical Clustering
- **Description:** Creates tree of clusters
- **Methods:** Ward, complete, average, single linkage
- **Outputs:**
  - Dendrogram
  - Cluster assignments at cut height
  - Cophenetic correlation
- **Plot:** Dendrogram

#### Time Series Analysis

##### Stationarity Tests
- **Augmented Dickey-Fuller (ADF) Test**
  - Tests for unit root (non-stationarity)
  - Outputs: Test statistic, p-value, critical values
  - Interpretation: p < 0.05 suggests stationarity

- **KPSS Test**
  - Tests null hypothesis of stationarity
  - Complementary to ADF test
  - Use both for robust conclusion

##### Seasonal Decomposition
- **Description:** Separates time series into components
- **Components:**
  - Trend
  - Seasonal pattern
  - Residual (irregular)
- **Models:** Additive, multiplicative
- **Plot:** Decomposition plot (4 subplots)

##### Holt-Winters Method
- **Description:** Exponential smoothing with trend and seasonality
- **Options:**
  - Trend type (additive, multiplicative, none)
  - Seasonal type (additive, multiplicative, none)
  - Seasonal period
- **Outputs:**
  - Forecasts
  - Fitted values
  - MSE, MAE, RMSE
- **Plot:** Actual vs. forecast

##### Moving Averages
- **Description:** Smooths data using rolling window
- **Options:** Window size
- **Outputs:**
  - Smoothed series
  - MSE, MAE, RMSE
- **Plot:** Original vs. smoothed

---

### Machine Learning Menu

#### Supervised Learning

##### Logistic Regression
- **Description:** Binary classification using logistic function
- **Options:**
  - Dependent variable (binary)
  - Independent variables (multiple)
  - Train/test split ratio
  - Regularization (L1, L2)
  - Max iterations
- **Outputs:**
  - Coefficients
  - Confusion matrix
  - Accuracy, precision, recall, F1-score
  - Classification report
  - ROC curve and AUC
- **Plots:**
  - ROC curve
  - Confusion matrix heatmap
  - Predicted vs. actual
- **Prediction:** Paste new data or upload CSV for predictions

##### Decision Tree
- **Description:** Tree-based classifier
- **Options:**
  - Max depth
  - Min samples split
  - Criterion (gini, entropy)
- **Outputs:**
  - Tree visualization
  - Feature importance
  - Confusion matrix
  - Accuracy metrics
- **Plot:** Decision tree diagram

##### Random Forest
- **Description:** Ensemble of decision trees
- **Options:**
  - Number of trees
  - Max depth
  - Feature subset size
- **Outputs:**
  - Feature importance
  - Out-of-bag score
  - Confusion matrix
  - Accuracy metrics
- **Plot:** Feature importance bar chart

##### Naive Bayes
- **Description:** Probabilistic classifier based on Bayes' theorem
- **Types:** Gaussian, Multinomial, Bernoulli
- **Outputs:**
  - Prior probabilities
  - Confusion matrix
  - Accuracy metrics

##### K-Nearest Neighbors (KNN)
- **Description:** Instance-based classifier
- **Options:**
  - Number of neighbors (k)
  - Distance metric
  - Weight function
- **Outputs:**
  - Confusion matrix
  - Accuracy metrics
- **Plot:** Decision boundaries (2D)

##### Support Vector Machine (SVM)
- **Description:** Maximum margin classifier
- **Options:**
  - Kernel (linear, RBF, polynomial)
  - C parameter (regularization)
  - Gamma (kernel coefficient)
- **Outputs:**
  - Support vectors
  - Confusion matrix
  - Accuracy metrics
- **Plot:** Decision boundaries with support vectors

##### Neural Network
- **Description:** Multi-layer perceptron classifier
- **Options:**
  - Hidden layer sizes
  - Activation function
  - Learning rate
  - Max iterations
- **Outputs:**
  - Loss curve
  - Confusion matrix
  - Accuracy metrics
- **Plot:** Training loss over iterations

#### Unsupervised Learning

##### K-Means Clustering
- **Description:** Partitions data into K clusters
- **Options:**
  - Number of clusters
  - Distance metric
  - Initialization method
- **Outputs:**
  - Cluster assignments
  - Cluster centers
  - Silhouette score
  - Inertia (within-cluster sum of squares)
- **Plot:** Cluster scatter plot
- **Prediction:** Assign new data points to nearest cluster

##### Hierarchical Clustering
- **Description:** Builds hierarchy of clusters
- **Options:**
  - Linkage method (ward, average, complete, single)
  - Distance metric
- **Outputs:**
  - Dendrogram
  - Cluster assignments
  - Cophenetic correlation
- **Plot:** Dendrogram tree
- **Prediction:** Assign new points to nearest cluster centroid

##### DBSCAN Clustering
- **Description:** Density-based clustering
- **Options:**
  - Epsilon (neighborhood radius)
  - Min samples (core point threshold)
- **Outputs:**
  - Cluster assignments
  - Core samples
  - Noise points (-1 label)
- **Plot:** Cluster scatter with noise points
- **Prediction:** Label new points based on neighborhood density

##### Principal Component Analysis (PCA)
- **Description:** Linear dimensionality reduction
- **Options:**
  - Number of components
  - Standardization
- **Outputs:**
  - Transformed data
  - Explained variance ratio
  - Component loadings
  - Eigenvalues
- **Plots:** Scree plot, biplot, component loadings
- **Prediction:** Transform new data into principal component space

##### Factor Analysis
- **Description:** Latent variable analysis
- **Options:**
  - Number of factors
  - Rotation method
- **Outputs:**
  - Factor loadings
  - Communalities
  - Factor scores
- **Plot:** Factor loadings heatmap
- **Prediction:** Compute factor scores for new observations

##### Gaussian Mixture Models (GMM)
- **Description:** Probabilistic clustering using mixture of Gaussians
- **Options:**
  - Number of components
  - Covariance type
  - Max iterations
- **Outputs:**
  - Cluster assignments
  - Cluster probabilities
  - BIC, AIC scores
  - Means and covariances
- **Plot:** Cluster scatter with probability contours
- **Prediction:** Predict cluster probabilities for new data

---

### NLP (Natural Language Processing) Menu

#### Text Analysis
- **Word Frequency**
  - Count and visualize most common words
  - Remove stopwords
  - Plot: Bar chart of top words

- **Word Cloud**
  - Visual representation of word frequencies
  - Customizable colors and shapes

#### Sentiment Analysis
- **Description:** Classifies text sentiment (positive, negative, neutral)
- **Methods:** VADER, TextBlob
- **Outputs:** Sentiment scores, classification

#### Topic Modeling
- **Latent Dirichlet Allocation (LDA)**
  - Discovers topics in document collection
  - Outputs: Topic distributions, top words per topic
  - Plot: Topic word clouds

#### Text Preprocessing
- **Tokenization**
- **Lemmatization**
- **Stopword Removal**
- **Named Entity Recognition**

---

### Bibliometrics Menu

#### Citation Analysis
- **Description:** Analyzes citation patterns
- **Outputs:**
  - Citation counts per author/paper
  - H-index
  - Citation network
- **Plot:** Citation network graph

#### Co-Authorship Analysis
- **Description:** Examines collaboration patterns
- **Outputs:**
  - Co-authorship network
  - Collaboration statistics
- **Plot:** Network graph of authors

#### Bibliometric Coupling
- **Description:** Measures similarity based on shared references
- **Outputs:**
  - Coupling strength matrix
  - Document clusters
- **Plot:** Coupling network

#### Word Count Analysis
- **Description:** Analyzes keyword frequencies
- **Outputs:** Word frequencies, trending terms
- **Plot:** Bar chart, word cloud

#### Keyword Analysis
- **Description:** Extracts and analyzes keywords from documents
- **Methods:** TF-IDF, keyword extraction
- **Plot:** Keyword cloud

---

### Plots Menu

#### Basic Plots
- **Histogram:** Distribution of continuous variables
- **Box Plot:** Five-number summary with outliers
- **Scatter Plot:** Relationship between two variables
- **Line Plot:** Trends over time/sequence
- **Bar Chart:** Comparisons across categories
- **Pie Chart:** Part-to-whole relationships

#### Advanced Plots
- **Heatmap:** Correlation or confusion matrix
- **Pair Plot:** Pairwise relationships (scatter matrix)
- **Violin Plot:** Distribution shape with quartiles
- **3D Scatter:** Three-variable relationships
- **Contour Plot:** 2D density or function values
- **Q-Q Plot:** Tests normality assumption

---

### Help Menu

#### About SADSA
- **Description:** Version information and credits
- **Developer:** AMCHIK SOLUTIONS
- **License:** Full Version

#### User Guide
- **Description:** Opens comprehensive documentation
- **Contents:** 
  - Getting started tutorial
  - Menu reference
  - Analysis examples
  - FAQ

#### License Information
- **Description:** View license status and validity
- **Actions:** 
  - Check license
  - Activate license
  - View machine ID
  - License agreement

#### Contact Support
- **Email:** contact@codingfigs.com
- **Website:** http://codingfigs.com
- **Response Time:** 24-48 hours

---

## Data Management

### Supported File Formats

#### Import Formats
- **CSV** (Comma-separated values)
- **Excel** (.xlsx, .xls)
- **JSON** (JavaScript Object Notation)
- **TSV** (Tab-separated values)
- **Parquet** (Apache Parquet)
- **LibreOffice** (.ods)

#### Export Formats
- **CSV**
- **Excel** (.xlsx)
- **JSON**
- **HTML** (for reports)
- **PDF** (via report generation)

### Data Grid Features
- **Editable Cells:** Double-click to edit
- **Sort:** Click column headers
- **Filter:** Right-click column header
- **Add/Delete Rows:** Context menu
- **Add Columns:** Edit → Add Column
- **Copy/Paste:** Standard keyboard shortcuts

---

## Analysis Workflow

### General Steps
1. **Load Data:** File → Open Data File
2. **Explore Data:** View in data grid, check descriptive statistics
3. **Prepare Data:** Apply transformations if needed
4. **Select Analysis:** Choose from menu
5. **Configure Options:** Select variables, parameters, tests
6. **Run Analysis:** Click "Run Analysis"
7. **View Results:** Multiple tabs with tables and plots
8. **Export Results:** Export to CSV or download report

### Tips for Best Results
- **Check Assumptions:** Use normality tests, correlation checks
- **Handle Missing Data:** Remove or impute before analysis
- **Scale Variables:** Use standardization for distance-based methods
- **Validate Results:** Check p-values, confidence intervals, fit indices
- **Visualize:** Always inspect plots to understand patterns

---

## Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open File | Ctrl+O |
| Save | Ctrl+S |
| Copy | Ctrl+C |
| Paste | Ctrl+V |
| Delete Row | Delete |
| Undo | Ctrl+Z |
| Redo | Ctrl+Y |
| Find | Ctrl+F |
| Select All | Ctrl+A |

---

## System Requirements

### Minimum Requirements
- **OS:** Windows 10 (64-bit)
- **RAM:** 4 GB
- **Storage:** 500 MB free space
- **Display:** 1280x720 resolution

### Recommended Requirements
- **OS:** Windows 11 (64-bit)
- **RAM:** 8 GB or more
- **Storage:** 1 GB free space
- **Display:** 1920x1080 resolution
- **Processor:** Multi-core CPU for faster computations

---

## License & Activation

SADSA uses a license-based activation system:
- **Trial Version:** 30 days with full features
- **Full Version:** Requires license key
- **Activation:** Help → License Information → Activate License

To obtain a license key:
1. Visit http://codingfigs.com
2. Purchase license
3. Receive license key via email
4. Activate in SADSA using Help menu

#### ✅ During Trial Period (Days 1-30):
- ✅ **File** - Supports CSV, Excel, and few other data formats
- ✅ **Transformations** - Data Transformations such as variable recoding, computing etc.
- ✅ **Data Simulations** - Data simulations for quit testing learning purposes (supports cholsky, SVD, QR etc).
- ✅ **Data Analytics** - Uni, bi and multivariate analysis (including CA, PCS, MDS, EFA, CFA, Time Series Forecasting etc.) 
- ✅ **Machine Learning** - All supervised & unsupervised algorithms
- ✅ **NLP** - Document-term matrix generation & feature extraction
- ✅ **Meta Analysis** - Fixed/random effects, heterogeneity tests
- ✅ **Bibliometrics** - Citation analysis, co-authorship networks
- ✅ **Advanced Plots** - Multi-plot generator with customization
- ✅ **Report Download** - Export to PDF/DOCX
- ✅ **Data Export** - All formats (CSV, Excel, JSON, Parquet, ODS)
- ✅ **File Import** - All supported formats

**Trial starts automatically** on first launch - no registration required!

#### ⚠️ After Trial Expires (Day 31+):
- ❌ Machine Learning menu - **Disabled** (requires license)
- ❌ NLP menu - **Disabled** (requires license)
- ❌ Meta Analysis menu - **Disabled** (requires license)
- ❌ Bibliometrics menu - **Disabled** (requires license)
- ❌ Plots menu - **Disabled** (requires license)
- ❌ Download Reports - **Disabled** (requires license)
- ⚠️ File Import - **CSV only** (other formats blocked)
- ✅ Basic data viewing, editing and analysis - **Still available**

### Full License Activation

Activate a **FULL LICENSE** for:
- ✅ **Permanent Access** - Never expires
- ✅ **All Features** - No restrictions
- ✅ **All File Formats** - Import/export everything
- ✅ **Priority Support** - Direct email assistance
- ✅ **Free Updates** - Receive new features and improvements

#### How to Activate:

1. **Get Your Machine ID**:
   - Help → Machine ID Information
   - Copy either "Computer Name" OR "Machine ID"
   - Example: `39FBD9ACCC7D0618`

2. **Request License**:
   - Email: **contact@codingfigs.com**
   - Subject: "SADSA License Request"
   - Include: Your Machine ID or Computer Name
   - Specify: License type needed (Full/Extended Trial)

3. **Activate**:
   - Receive license key via email
   - Help → Activate License
   - Paste the license key
   - Click "Activate License"
   - **Restart SADSA**

4. **Verify**:
   - Title bar shows: "SADSA - Full Version"
   - Help → License Information: "ACTIVE (PERMANENT)"
   - All menus and features enabled

### License Status Display

Your license status is visible in multiple locations:
- **Title Bar**: `SADSA - Trial (X days remaining)` or `SADSA - Full Version`
- **Status Bar**: Shows daily countdown and available features
- **Help → License Information**: Detailed license status, expiration date, Machine ID

---
## License Agreement

Copyright © 2025 AMCHIK SOLUTIONS. All rights reserved.

This software is licensed, not sold. By installing and using SADSA, you agree to the terms specified in the License Agreement accessible via Help → License Agreement.

**Key Terms:**
- Personal/Academic use permitted
- Commercial use requires appropriate license
- Redistribution prohibited
- Reverse engineering prohibited

For full license terms, see LICENSE.txt or Help → License Agreement within the application.

---

## Support & Contact

### Technical Support
- **Email:** contact@codingfigs.com
- **Website:** http://codingfigs.com
- **Response Time:** 24-48 hours on business days

### Report Issues
When reporting issues, please include:
- SADSA version number
- Windows version
- Error message (screenshot if possible)
- Steps to reproduce
- Sample data (if applicable)

### Feature Requests
We welcome suggestions for new features! Email us at contact@codingfigs.com with:
- Feature description
- Use case
- Expected benefit

---

## Updates & Versions

SADSA receives regular updates with:
- New statistical methods
- Bug fixes
- Performance improvements
- UI enhancements

**Check for Updates:** Help → Check for Updates

---

## Credits

**Developed by:** Dr. M. Kamakshaiah / AMCHIK SOLUTIONS  
**Website:** [Codingfigs](http://codingfigs.com)  
**Contact:** dr.m.kamakshaiah@gmail.com

**Built with:**
- Python 3.11+
- Tkinter (GUI)
- Pandas (Data manipulation)
- NumPy (Numerical computing)
- SciPy (Scientific computing)
- Scikit-learn (Machine learning)
- Matplotlib/Seaborn (Visualization)
- Statsmodels (Statistical modeling)

---

## Frequently Asked Questions (FAQ)

**Q: Can I use SADSA for commercial purposes?**  
A: Yes, with a commercial license. Contact us for pricing.

**Q: Is my data secure?**  
A: All data processing is done locally on your computer. No data is transmitted to external servers.

**Q: Can I import data from databases?**  
A: Currently, SADSA supports file-based imports. Database connectivity is planned for future versions.

**Q: What if I get an error during analysis?**  
A: Check that your data meets the analysis assumptions (e.g., no missing values, appropriate data types). Contact support if issues persist.

**Q: Can I save my analysis workflow?**  
A: Currently, you must manually repeat analysis steps. Workflow automation is planned for a future release.

**Q: Is there a Mac or Linux version?**  
A: Currently Windows-only. Mac and Linux versions are under consideration.

---

**Thank you for choosing SADSA!**

---
# LATEST UPDATES
---

## 🗂️ File Menu: Advanced File Import
- **Menu:** `File → Advanced File Import...`
- Supports: CSV, Excel, JSON, Parquet, ODS, TSV, TXT, WEKA, SPSS, MATLAB, Tableau, SQLite, MySQL, PostgreSQL
- **How to use:**
  1. Click **File** menu
  2. Select **Advanced File Import...**
  3. Choose format tab (Standard, Advanced, Databases)
  4. Browse, preview, and import your data

---

## 🤖 Machine Learning: Ensemble Methods
- **Menu:** `Machine Learning → Ensemble Methods → [XGBoost, LightGBM, AdaBoost, Gradient Boosting]`
- Unified 2x2 grid interface for all ensemble algorithms
- **Steps:**
  1. Select target variable
  2. Select features
  3. Configure parameters (n_estimators, learning_rate, max_depth, test_size)
  4. Train and view results

---

## 🖱️ Context Menu Integration
- **Feature:** "Open with SADSA" in Windows right-click menu for `.csv`, `.xlsx`, `.xls`, `.txt`, `.tsv`, `.sadsa`, `.dat`, `.data`
- **Install:**
  - Run `install_context_menu.bat` (admin)
  - Or use `install_context_menu.py` or registry files
- **Uninstall:**
  - Use `uninstall_context_menu.bat`, `uninstall_context_menu.py`, or registry uninstaller

---

## 🐍 Python Console
- **Menu:** `Tools → Python Console`
- Launch an interactive Python terminal within SADSA.
- Access app objects: `app`, `data`, `pd`, `np`, `plt`, etc.
- Run any Python code, inspect data, and call SADSA methods directly (e.g., `app.perform_correlation_analysis()`).
- Features:
  - Command history (Up/Down arrows)
  - Themed interface (choose your favorite look)
  - Type `help()` for a quick guide, `methods()` for a list of available app methods
  - Supports all standard Python imports (e.g., `import seaborn as sns`)
- Great for power users, debugging, and custom analysis!

---

## 📚 Other Features
- **Help Menu:** License activation, documentation, support
- **NLP, Meta Analysis, Bibliometrics:** (optional, see deployment guide)
- **Advanced plotting, reporting, and export**

---

## 📖 References & Guides
- See the `backup/` folder for detailed markdown guides:
  - `ADVANCED_FILE_IMPORT_GUIDE.md`
  - `ENSEMBLE_QUICK_START.md`
  - `CONTEXT_MENU_GUIDE.md`
  - `FINAL_CLEANUP_SUMMARY.md`
  - ...and more!

---

## 🏁 Quick Start
1. Install Python dependencies: `pip install -r requirements.txt`
2. Run the app: `python pyfda.py`
3. Use the installer for Windows deployment: `SADSA_Installer_*.exe`

---

## 💡 Need Help?
- Open the **Help** menu in the app
- See the documentation in the `backup/` folder
- Or contact the developer via GitHub

---

> © 2025 AMCHIK SOLUTIONS. All rights reserved.
