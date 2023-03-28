# Step 1 prepare library

# Check Python version
import sys
print('python:{}'.format(sys.version))
# Check scipy
import scipy
print('scipy:{}'.format(scipy.__version__))
# Check numpy
import numpy
print('numpy:{}'.format(numpy.__version__))
# Check matplotlib
import matplotlib
print('matplotlib:{}'.format(matplotlib.__version__))
# Check pandas
import pandas
print('pandas:{}'.format(pandas.__version__))
# Check scikit-learn
import sklearn
print('sklean:{}'.format(sklearn.__version__))

# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Step 2 load dataset

url="https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-length','sepal-width','petal-length','petal-width','class']
dataset=read_csv(url,names=names)

# Step 3 summarise dataset

# Shape
print(dataset.shape)
# Head
print(dataset.head(20))
# Descriptions
print(dataset.describe())
# Class distribution
print(dataset.groupby('class').size())

# Step 4 visualise dataset

# Box and whisker plots
# dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
# pyplot.show()
# Histograms
# dataset.hist()
# pyplot.show()
# Scatter plot matrix
# scatter_matrix(dataset)
# pyplot.show()

# Step 5 evaluate algorithms

# Split out validation dataset
array = dataset.values
x = array[:,0:4]
y = array[:,4]
x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=0.20, random_state=1)

# Spot check algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

# Compare models
pyplot.boxplot(results, labels=names)
pyplot.title('Algorithms Comparison')
pyplot.show()

# Step 6 make predictions

# Make predictions on validation dataset
model = SVC(gamma='auto')
model.fit(x_train, y_train)
predictions = model.predict(x_validation)

# Evaluate predictions
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))