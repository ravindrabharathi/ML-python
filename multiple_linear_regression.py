from sklearn import linear_model
x = [[0.18, 0.89], [1.0 , 0.26], [0.92 ,0.11], [0.07, 0.37], [0.85, 0.16], [0.99, 0.41],[0.87, 0.47]]
y = [109.85, 155.72, 137.66, 76.17,139.75,162.6,151.77]
lm = linear_model.LinearRegression()
lm.fit(x, y)
a = lm.intercept_
b = lm.coef_
print(a, b[0], b[1])
X_new=[[0.49 ,0.18],[0.57 ,0.83], [0.56 ,0.64] , [0.76 ,0.18]  ]
Y_new=lm.predict(X_new);
print(Y_new)
#for i in len(X_new):
#print(a+b[0]*X_new[i,0]+b[1]*)