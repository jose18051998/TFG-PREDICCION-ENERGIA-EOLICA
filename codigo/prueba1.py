from sklearn.neural_network import MLPClassifier
X = [[0.], [1.], [1.], [2.], [2.], [0.], [0.], [0.]]
y = [0, 1, 1, 1, 1, 0, 0, 0]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 4), random_state=1)

clf.fit(X, y)
MLPClassifier(alpha=1e-05, hidden_layer_sizes=(10,), random_state=1, solver='lbfgs')

print(clf.predict([[2.], [1.], [0.]]))

print([coef.shape for coef in clf.coefs_])
