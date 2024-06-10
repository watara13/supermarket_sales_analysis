import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport
from statsmodels.tsa.arima.model import ARIMA

data = 'supermarket_sales.csv'
df = pd.read_csv(data)

profile = ProfileReport(df)
output_path = "supermarket_sales_profile_report.html"
profile.to_file(output_path)

df['Date'] = pd.to_datetime(df['Date'])
print(df.head(5))
print(df.columns)
print(df.describe().round(1))
print(df.info())
print(df[df.duplicated()])
df = df.drop_duplicates()

df_problmee = df[['Product line', 'Unit price', 'Quantity', 'Tax 5%', 'Total']]
oui = df_problmee[df_problmee['Product line'].isnull()]
ok = df['Product line'].dropna()
prix_manquant = oui.groupby('Unit price')['Product line'].unique()
valeur_remplacement = df.dropna()
liste_product = valeur_remplacement['Product line'].unique()
maj = valeur_remplacement.groupby('Product line')['Unit price'].mean().round()
moyennes_dict = maj.to_dict()

df['Unit price'] = (df['Total'] - df['Tax 5%']) / df['Quantity']
print(df['Unit price'].isnull().sum())
dictionnaire = df.dropna(subset=['Product line']).drop_duplicates(subset=['Unit price']).set_index('Unit price')['Product line'].to_dict()
liste = list(liste_product)
df['Product line'] = df.apply(
    lambda row: random.choice(liste_product) if pd.isna(row['Product line']) else row['Product line'],
    axis=1)

prix_ht = df['Unit price'] * df['Quantity']
tva = df['Tax 5%'].round(2)
df['new_total'] = prix_ht + tva
pays_total = df.groupby('City')['Total'].mean()
pays_total.plot(kind='bar', color='red')
plt.ylabel('Total')
plt.xlabel('City')
plt.title('Total par ville')
plt.show()

fig, ax = plt.subplots(figsize=(18, 8))
par_genre = df.groupby(['Gender', 'Product line'])['Total'].mean().sort_values()
colors = {'Male': 'blue', 'Female': 'pink'}
color_list = [colors[gender] for gender, _ in par_genre.index]
par_genre.plot(kind='bar', color=color_list)
plt.grid()
plt.show()

marge_par_produit = df.groupby('Product line')[['Total', 'gross margin percentage']].size()
print('Nous avons une marge fixe par produit')
print(marge_par_produit)
print(df[['Total', 'new_total']].tail(5))
df = df.dropna()

df['Gross Income'] = df['Total'] - df['cogs']
marge_par_produit = df.groupby('Product line')['Gross Income'].sum()
marge_par_produit.plot(kind='bar', color='green')
plt.ylabel('Marge bénéficiaire')
plt.xlabel('Ligne de produit')
plt.title('Marge bénéficiaire par ligne de produit')
plt.show()

df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

daily_sales = df.resample('D')['Total'].sum()

train = daily_sales[:'2019-02-28']
test = daily_sales['2019-03-01':]

model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(10, 6))
plt.plot(train, label='Entraînement')
plt.plot(test, label='Test')
plt.plot(forecast, label='Prévision', color='red')
plt.xlabel('Date')
plt.ylabel('Ventes')
plt.title('Prévision des ventes avec ARIMA')
plt.legend()
plt.show()

modeles = [
    {'name': 'LinearRegression', 'modele': LinearRegression()},
]
x = df[['Unit price', 'Quantity', 'Tax 5%']]
y = df['Total']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

for item in modeles:
    name = item['name']
    modele = item['modele']
    modele.fit(x_train, y_train)
    y_pred = modele.predict(x_test)
    MSE = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'Entraînement du modèle {name}')
    print(f'MSE : {MSE}')
    print(f'R2 : {r2}')
