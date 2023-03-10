import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import r2_score, mean_squared_error

data = pd.read_csv('fuel_consumption.csv')
pd.set_option('display.max_columns',13)
pd.set_option('display.width', None)

#-----1.stavka-----

print(data.head(5)) #ispis prvih 5 vrsta

#-----2.stavka-----
print(data.info())
print(data.describe(include=[object]))

# provera da li ima nan vrednosti
total = data.isnull().sum().sort_values(ascending=False)
perc1 = data.isnull().sum() / data.isnull().count() * 100
perc2 = (round(perc1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, perc2], axis=1, keys=['Total', '%'])
print(missing_data.head(5))

# -----3. STAVKA------

#popunjavanje NaN sa najcescom vrednoscu
data.TRANSMISSION = data.TRANSMISSION.fillna(data.TRANSMISSION.mode()[0])
data.ENGINESIZE = data.ENGINESIZE.fillna(data.ENGINESIZE.mode()[0])
data.FUELTYPE = data.FUELTYPE.fillna(data.FUELTYPE.mode()[0])
#print(data.loc[data.TRANSMISSION.isnull()].head(5))

#----4.stavka----

# zavisnost kontinualnih vrednosti prikazan na kor. matrici
plt.figure()
num_data = data.select_dtypes(include=np.number)
del num_data['MODELYEAR']
del num_data['CYLINDERS']

mat = num_data.corr()
plt.figure(figsize=(10,10))
sb.heatmap(mat,
             annot=True,
             fmt='.2f',
             annot_kws={'fontsize': 10}
           )
plt.xticks(fontsize=7,rotation=45)
plt.yticks(fontsize=7, rotation=30)
plt.show()

# ----- 5.STAVKA -----

# prikaz zavisnosti izlaza od ulaza u dekartovom k. s.
for i in range(0, len(num_data.columns)-1):
    plt.figure()
    plt.scatter(num_data.iloc[:,i], num_data.iloc[:,-1])
    plt.xlabel(num_data.columns[i])
    plt.ylabel(num_data.columns[-1])
    #plt.show()

# ----- 6. STAVKA -----

# zavisnost izlaza od kategorickih ulaza

# godina proizvodnje je za sve automobile ista pa neće biti prikazana grafički

cat_data = data.select_dtypes(exclude='number')
output = data['CO2EMISSIONS']
cat_data.insert(0,'CO2EMISSIONS',output)
print(cat_data)

plt.figure(figsize=(20,10))
sb.catplot(data=cat_data, x='MAKE', y="CO2EMISSIONS", kind="swarm",height=5, aspect=4,s=10)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(20,10))
sb.catplot(data=cat_data, x='VEHICLECLASS', y="CO2EMISSIONS", kind="swarm",height=5, aspect=3,s=10)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(20,10))
sb.catplot(data=cat_data, x='TRANSMISSION', y="CO2EMISSIONS", kind="swarm",height=5, aspect=2,s=10)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(20,10))
sb.catplot(data=cat_data, x='FUELTYPE', y="CO2EMISSIONS", kind="swarm",height=5, aspect=1,s=10)
plt.show()

makers = cat_data['MAKE'].unique()

# plotovanje zavisnosti emisije od modela automobila prema proizvodjacima

# for i in range(0, len(makers)):
#     #print(makers[i])
#     #print(cat_data.loc[cat_data["MAKE"]==makers[i]], 'MODEL')
#     data_models = cat_data.loc[cat_data["MAKE"]==makers[i]]
#     print(data_models)
#     plt.figure(figsize=(20, 10))
#     sb.catplot(data=data_models, y='MODEL', x="CO2EMISSIONS", kind="swarm", height=5, aspect=1, s=10)
#     plt.title(makers[i])
#     plt.show()

# -----7. STAVKA -----

# kodiranje kategorickih vrednosti
# print(data['VEHICLECLASS'].unique())
# print(data['FUELTYPE'].unique())

# enc = OrdinalEncoder()
#data[['VEHICLECLASS','FUELTYPE','CYLINDERS']] = enc.fit_transform(data[['VEHICLECLASS','FUELTYPE','CYLINDERS']])

# data_vclass = pd.get_dummies(data['VEHICLECLASS'])
# data_ftype = pd.get_dummies(data['FUELTYPE'])
# data_cyl = pd.get_dummies(data['CYLINDERS'])

#  odabir atributa koji ucestvuju u modelu

Y = data['CO2EMISSIONS']
X = data[['CYLINDERS','ENGINESIZE','FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_CITY']]

#normalizovanje podataka

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X))

#podela na test i train skup

X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                     test_size=0.2,
                                                     shuffle=True,
                                                     random_state=1)
#print(X_test.shape)

#ugradjena fja za lin. regresiju
lr = linear_model.LinearRegression()
lr.fit(X_train, Y_train)
Y_pred = lr.predict(X_test)

lr_score = r2_score(Y_test, Y_pred)
lr_rmse = mean_squared_error(Y_test, Y_pred, squared = False)
print("LR: Coefficients: ", lr.coef_)
print("LR: Intercept: ", lr.intercept_)
print("LR: R2 Score : ", lr_score)
print("LR: RMSE : ", lr_rmse, '\n \n')

# moja fja za lr gradient desceny

from LRGradientDescent import LinearRegressionGradientDescent

LRgd_model = LinearRegressionGradientDescent()
LRgd_model.fit(X_train, Y_train)
learning_rates = np.array([[0.28],[0.11], [0.13], [0.22], [0.17]])
lrgd_coeff, mse_history = LRgd_model.perform_gradient_descent(learning_rates, 100)
predicted = LRgd_model.predict(X_test)
lrgd_rmse = mean_squared_error(np.array(Y_test), predicted, squared=False)
lrgd_r2_score = r2_score(np.array(Y_test), predicted)

#print('rmse: ', mean_squared_error(np.array(Y_test), predicted, squared=False))

#print('score: ', r2_score(np.array(Y_test), predicted), end='\n \n')

print("LRGD: Coefficients: ", *lrgd_coeff[1:5])
print("LRGD: Intercept: ", lrgd_coeff[0])
print("LRGD: R2 Score : ", lrgd_r2_score)
print("LRGD: RMSE : ", lrgd_rmse)


# grafik zavisnosti greske od broja iteracija
plot6= plt.figure(6)
plt.plot(np.arange(0, len(mse_history), 1), np.array(mse_history))
plt.xlabel('broj iteracija')
plt.ylabel('mse')
plt.savefig('test6.png', dpi=250)

plt.show()
