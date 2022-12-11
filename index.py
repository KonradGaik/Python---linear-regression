import pandas;
from pandas import DataFrame;
import matplotlib.pyplot as plt;
from sklearn.linear_model import LinearRegression;
data = pandas.read_csv('cost_revenue_clean.csv')
# print(data.describe())
x = DataFrame(data, columns=['production_budget_usd'])
y = DataFrame(data, columns=['worldwide_gross_usd'])

regression = LinearRegression()
regression.fit(x, y)
regression.coef_ #theta 1
regression.intercept_ #theta 0
regression.score(x,y)

plt.figure(figsize=(10, 6))
plt.scatter(x,y, alpha = 0.3)
plt.plot(x, regression.predict(x), color='green', linewidth=4)
plt.title('Film Cost vs Revenue')
plt.xlabel('Production budget')
plt.ylabel('Revenue')
plt.ylim(0,3000000000)
plt.xlim(0,450000000)
plt.show()

