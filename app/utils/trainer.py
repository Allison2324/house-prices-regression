from sklearn.tree import DecisionTreeRegressor


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return DecisionTreeRegressor(random_state=42).fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
