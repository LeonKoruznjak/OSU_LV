import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn . metrics import accuracy_score
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay
from sklearn . linear_model import LogisticRegression


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)



# a)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train, s=15, cmap=mcolors.ListedColormap(["red", "blue"]))
plt.scatter(X_test[:, 0], X_test[:, 1], marker="x", c=y_test, s=25, cmap=mcolors.ListedColormap(["red", "blue"]))
plt.show()

# b)

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c)


# d)
y_prediction=LogRegression_model.predict(X_test)
cm=confusion_matrix(y_test,y_prediction)
disp=ConfusionMatrixDisplay(cm)
disp.plot()
plt.title('Matrica zabune')
plt.show()
print(f'Točnost: {accuracy_score(y_test,y_prediction)}')
print(f'Preciznost: {precision_score(y_test,y_prediction)}')
print(f'Odziv: {recall_score(y_test,y_prediction)}')


