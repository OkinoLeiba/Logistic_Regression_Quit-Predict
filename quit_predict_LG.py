# %% [markdown]
# # <span style="display: flex; justify-content: center; text-decoration: underline overline; text-decoration-style: solid; text-decoration-color: #732626; color: #000080;">Predication of Voluntary Quit Employment</span>

# %% [markdown]
# <a href="https://www.linkedin.com/in/okinoleiba" style="display: flex; justify-content: center;">Okino Kamali Leiba</a>
# 
# <span style="display: flex; justify-content: center;">![purple-divider](https://user-images.githubusercontent.com/7065401/52071927-c1cd7100-2562-11e9-908a-dde91ba14e59.png)</span>

# %%
import pandas as pd, numpy as np, seaborn as sns, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error, mean_absolute_error


# %% [markdown]
# ## <span style="padding-left: 650px; text-decoration: underline overline; text-decoration-style: double; text-decoration-color: #732626; color: #000080;">Exploratory Data Analysis</span>
# <span style="display: flex; justify-content: center;">![purple-divider](https://user-images.githubusercontent.com/7065401/52071927-c1cd7100-2562-11e9-908a-dde91ba14e59.png)</span>

# %%
filepath = "C:/Users/Owner/source/vsc_repo/machine_learn_cookbook/Logistic_Regression_Quit-Predict/hr_file.csv"
hr_data = pd.read_csv(filepath, delimiter=",", header=0, engine="python", encoding="utf-8", on_bad_lines="warn")
hr_data.rename(str.title, axis='columns', inplace=True)
hr_data['Departments '] = [s.title() for s in hr_data['Departments ']]
hr_data['Salary'] = [s.title() for s in hr_data['Salary']]
hr_data['Departments '] = hr_data['Departments '].replace(["Hr","and", "It","Mng", "R&d"], ["HR","&","IT","MNG","R&D"], regex=True, inplace=False)
hr_data.head(5)


# %%
hr_data.tail(5)

# %%
hr_data.info()

# %%
hr_data.index

# %%
hr_data.shape

# %%
hr_data.ndim

# %%
hr_data.columns

# %%
hr_data.dtypes

# %%
hr_data["Departments "].unique()

# %%
hr_data["Salary"].unique()

# %%
hr_data.describe()

# %%
hr_data.groupby(["Departments "]).sum().round(1)

# %%
hr_data['Salary'] = [s.title() for s in hr_data['Salary']]
hr_data.groupby(["Salary"]).sum().round(1)

# %%
# https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing
le = LabelEncoder()
hr_data["Departments_Encode"] = le.fit_transform(hr_data["Departments "])
hr_data["Salary_Encode"] = le.fit_transform(hr_data["Salary"])
hr_data.head(5)

# %%
hr_data.corr()

# %% [markdown]
# ## <span style="padding-left: 450px; text-decoration: underline overline; text-decoration-style: dotted; text-decoration-color: #732626; color: #000080;">Data Visualizations</span>
# <span style="display: flex; justify-content: center;">![purple-divider](https://user-images.githubusercontent.com/7065401/52071927-c1cd7100-2562-11e9-908a-dde91ba14e59.png)</span>

# %%
import warnings
warnings.filterwarnings("ignore")
mpl.rcParams['figure.facecolor'] = "darkmagenta"
mpl.rcParams['axes.facecolor'] = "gray"

sns.pairplot(hr_data[0:500], kind="scatter", markers="o", diag_kind="kde", palette="tab10");

# %%
mpl.rcParams['figure.facecolor'] = "crimson"
mpl.rcParams['ytick.labelleft'] = True;
mpl.rcParams['xtick.labelbottom'] = True;
plt.figure(figsize=(16,8), constrained_layout=True, dpi=100)
sns.heatmap(hr_data.corr(), annot=True)

# %%
plt.get_cmap("plasma")

# %%
mpl.rcParams['figure.facecolor'] = "peru"
hr_data['Departments '] = [s.title() for s in hr_data['Departments ']]
hr_data['Departments '] = hr_data['Departments '].replace(["Hr","and", "It","Mng", "R&d"], ["HR","&","IT","MNG","R&D"], regex=True, inplace=False)
hr_data['Departments '].value_counts(["Quit the Company"]).plot(kind='pie', figsize=(20,6), cmap="plasma")
plt.axis("off")
plt.title("Departments")

# %%
hr_data.columns

# %%
y = hr_data["Quit The Company"]
features = ['Satisfaction Level', 'Last Evaluation', 'Number Of Projects',
       'Monthly Hours', 'Total Time At The Company', 'Work Accidents',
       'Promoted In Last 5 Yrs', 'Management', 
       'Departments_Encode', 'Salary_Encode']
X = hr_data[features]

# %%
scale = StandardScaler()
X = scale.fit_transform(X)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# %% [markdown]
# ## <span style="padding-left: 750px; text-decoration: underline overline; text-decoration-style: dashed; text-decoration-color: #732626;  color: #000080;">Logistic Regression</span>
# <span style="display: flex; justify-content: center;">![purple-divider](https://user-images.githubusercontent.com/7065401/52071927-c1cd7100-2562-11e9-908a-dde91ba14e59.png)</span>

# %%
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import GridSearchCV
param_grid = {"penalty" : ["l1", "l2", "elasticnet"], "random_state" : [0,42], "solver" : ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"],
"max_iter" : [100, 150, 200, 250]}
grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="neg_mean_squared_log_error")
grid_search.fit(X_train, y_train)
print(grid_search.best_params_)

log = LogisticRegression(max_iter=100, penalty='l1', random_state=0, solver='liblinear')
log.fit(X_train, y_train)
y_predict = log.predict(X)
y_prob = log.predict_proba(X)[:,1]


# %%
hr_data["Predictions"] = y_predict
hr_data["Probabilities"] = y_prob


# %%
log_coef = log.coef_
log_intercept = log.intercept_
print("Logisitic Coefficent: ", log_coef)
print("Logisitic Intercept: ", log_intercept)

# %%
print("Mean Absolute Error: ", mean_absolute_error(y, y_predict))
print("Mean Squared Error: ", mean_squared_error(y, y_predict))

# %%
hr_data.head(5)

# %% [markdown]
# ## <span style="padding-left: 350px; text-decoration: underline overline; text-decoration-style: wavy; text-decoration-color: #732626; color: #000080;">Making Art with Science
# <span style="display: flex; justify-content: center;">![purple-divider](https://user-images.githubusercontent.com/7065401/52071927-c1cd7100-2562-11e9-908a-dde91ba14e59.png)</span>

# %%
# Making Art with Science
mpl.rcParams['figure.facecolor'] = "purple"
hr_data["Predictions"].plot.pie(figsize=(16,8), ylabel=" ");

# %%
mpl.rcParams['figure.figsize'] = (17,8)
mpl.rcParams['axes.facecolor'] = "darkolivegreen"
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['ytick.labelleft'] = False
mpl.rcParams['grid.linestyle'] = "-."
mpl.rcParams['grid.color'] = "gold"
mpl.rcParams['grid.linewidth'] = 0.91


andrews_data = hr_data[['Number Of Projects',
       'Monthly Hours', 'Total Time At The Company', 'Work Accidents',
       'Promoted In Last 5 Yrs', 'Management', 
       'Departments_Encode', 'Salary_Encode', 'Salary']]
pd.plotting.andrews_curves(andrews_data, class_column="Salary", colormap="twilight")

# %%
mpl.rcParams['figure.facecolor'] = "indigo"
mpl.rcParams['axes.facecolor'] = "darkseagreen"
mpl.rcParams['grid.color'] = "saddlebrown"
mpl.rcParams['grid.linewidth'] = 10


pd.plotting.autocorrelation_plot(X, color="steelblue")

# %%
from cycler import cycler
mpl.rcParams['figure.facecolor'] = "purple";
# mpl.rcParams['axes.spines.left'] = False;
mpl.rcParams['xtick.labelbottom'] = False
mpl.rcParams['axes.facecolor'] = "darkgoldenrod";
mpl.rcParams['axes.grid'] = False;



pd.plotting.scatter_matrix(hr_data, figsize=(16,8), ax=None, color="firebrick");


# %% [markdown]
# ## Machine Learning with Power BI

# %%
import pandas as pd, numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# filepath = "C:/Users/Owner/source/vsc_repo/machine_learn_cookbook/Logistic_Regression_Quit-Predict/hr_file.csv"
# dataset = pd.read_csv(filepath, delimiter=",", header=0, engine="python", encoding="utf-8", on_bad_lines="warn")
dataset.rename(str.title, axis='columns', inplace=True)
dataset['Departments '] = [s.title() for s in dataset['Departments ']]
dataset['Salary'] = [s.title() for s in dataset['Salary']]
dataset['Departments '] = dataset['Departments '].replace(["Hr","and", "It","Mng", "R&d"], ["HR","&","IT","MNG","R&D"], regex=True, inplace=False)

le = LabelEncoder()
dataset["Departments_Encode"] = le.fit_transform(dataset["Departments "])
dataset["Salary_Encode"] = le.fit_transform(dataset["Salary"])

y = dataset["Quit The Company"]
features = ['Satisfaction Level', 'Last Evaluation', 'Number Of Projects',
       'Monthly Hours', 'Total Time At The Company', 'Work Accidents',
       'Promoted In Last 5 Yrs', 'Management', 
       'Departments_Encode', 'Salary_Encode']
X = dataset[features]

scale = StandardScaler()
X = scale.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
                                                    
log = LogisticRegression(max_iter=100, penalty='l1', random_state=0, solver='liblinear')
log.fit(X_train, y_train)
y_predict = log.predict(X)
y_prob = log.predict_proba(X)[:,1]

dataset["Predictions"] = y_predict
dataset["Probabilities"] = y_prob



