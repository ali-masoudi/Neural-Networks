# In the name of Allah

from sklearn.linear_model import LogisticRegression
from LogRegModule import read_data
import numpy as np

x, y = read_data()

model = LogisticRegression(fit_intercept=True)
model.fit(x.T, y)
print('Intercept (Theta 0: {}). Coefficients: {}'.format(model.intercept_, model.coef_))

# Predict admission probability (class ONE) for a test data
scores = np.array([[10, 12]])
class_probs = model.predict_proba(scores)
print("probability of being admitted for scores=%s: %2.1f" % (scores, class_probs[0][1]))

