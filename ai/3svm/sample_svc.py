#encoding=utf8
import numpy as np
import pylab as pl
from sklearn import svm
from sklearn.externals import joblib
# we create 40 separable points
np.random.seed(0)#保证每次运行时抓的值不变
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]#取20个点维度是2
Y = [0]*20 +[1]*20#前面20个点归为0，后面20个点归为1

#fit the model
clf = svm.SVC(kernel='linear',probability = True)
clf.fit(X, Y)
#
Z= np.r_[np.random.randn(1, 2) - [2, 2], np.random.randn(1, 2) + [2, 2]]
print(clf.predict(Z))
clf.predict_proba(Z)
# get support vectors
print (clf.support_vectors_)
# get indices of support vectors
print("========clf.support_=============")
print (clf.support_)
# get number of support vectors for each class
print("=======clf.n_support_==============")
print (clf.n_support_)

# get the separating hyperplane
w = clf.coef_[0]#取的w的值
a = -w[0]/w[1]#点斜式的斜率
xx = np.linspace(-5, 5)#从-5到5产生连续的值
yy = a*xx - (clf.intercept_[0])/w[1]#clf.intercept_[0]相当于是w3

# plot the parallels to the separating hyperplane that pass through the support vectors
b = clf.support_vectors_[0]
yy_down = a*xx + (b[1] - a*b[0])
b = clf.support_vectors_[-1]#-1指的是最后一个值
yy_up = a*xx + (b[1] - a*b[0])

print( "w: ")
print(w)
print ("a: ")
print(a)
# print "xx: ", xx
# print "yy: ", yy
print ("support_vectors_: ")
print("clf.support_vectors_")
print ("clf.coef_: ")
print(clf.coef_)
joblib.dump(clf, "train_model.pkl")
#加载模型
clf=joblib.load("train_model.pkl")
# switching to the generic n-dimensional parameterization of the hyperplan to the 2D-specific equation
# of a line y=a.x +b: the generic w_0x + w_1y +w_3=0 can be rewritten y = -(w_0/w_1) x + (w_3/w_1)


# plot the line, the points, and the nearest vectors to the plane
pl.plot(xx, yy, 'k-')
pl.plot(xx, yy_down, 'k--')
pl.plot(xx, yy_up, 'k--')

pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
          s=80, facecolors='none')
pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

pl.axis('tight')
pl.show()
