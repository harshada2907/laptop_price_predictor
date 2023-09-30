import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

data = pd.read_csv("laptop_data_cleaned.csv")
print(data)

print(data.isnull().sum())

features = data.drop("Price", axis = "columns")
target = data["Price"]

nfeatures = pd.get_dummies(features)

print(features)
print(nfeatures)

k = int(len(data) ** 0.5)
if k % 2 == 0:
	k = k + 1

mms = MinMaxScaler()
nnfeatures = mms.fit_transform(nfeatures)

x_train, x_test, y_train, y_test = train_test_split(nfeatures, target)

model = KNeighborsRegressor(n_neighbors = k, metric = "euclidean")
model.fit(x_train, y_train)

s1 = model.score(x_train, y_train)
print("Training score = ", s1)

s2 = model.score(x_test, y_test)
print("Testing score = ", s2)

company = int(input("Company  Enter 1 for Acer  2 for Apple  3 for Asus  4 for Chuwi  5 for Dell  6 for Fujitsu  7 for Google  8 for HP  9 for Huawei  10 for Lenovo  11 for LG  12 for Mediacom  13 for Microsoft  14 for MSI  15 for Razer  16 for Samsung  17 for Toshiba  18 for Vero  19 for Xiaomi : "))

if company == 1:
	d = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 2:
	d = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 3:
	d = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 4:
	d = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 5:
	d = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 6:
	d = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 7:
	d = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 8:
	d = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 9:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 10:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 11:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
elif company == 12:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
elif company == 13:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
elif company == 14:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
elif company == 15:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
elif company == 16:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
elif company == 17:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
elif company == 18:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
else:
	d = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

tn = int(input(" Type Name : Enter 1 for 2_in_1_covertible  2 for Gaming  3 for Netbook  4 for Notebook  5 for Ultrabook  6 for Workstation : "))

if tn == 1:
	d = d + [1, 0, 0, 0, 0, 0]

elif tn == 2:
	d = d + [0, 1, 0, 0, 0, 0]

elif tn == 3:
	d = d + [0, 0, 1, 0, 0, 0]

elif tn == 4:
	d = d + [0, 0, 0, 1, 0, 0]

elif tn == 5:
	d = d + [0, 0, 0, 0, 1, 0]

else:
	d = d + [0, 0, 0, 0, 0, 1]

ram = int(input("Enter the ram : "))

d = d + [ram]

weight = float(input("Enter the weight : "))

d = d + [weight]

ts = int(input("Enter 1 if laptop is touchscreen else 0 : "))

d = d + [ts]

ips = int(input("Enter 1 if in-plane switching else 0 : "))

d = d + [ips]

ppi = float(input("Enter the pixel per inch : "))

d = d + [ppi] 

cb = int(input("CPU-branch : Enter  1 for AMD Processor  2 for Intel core i3  3 for Intel core i5  4 for Intel core i7  5 for Other Intel processors : "))

if cb == 1:
	d = d + [1, 0, 0, 0, 0]
elif cb == 2:
	d = d + [0, 1, 0, 0, 0]
elif cb == 3:
	d = d + [0, 0, 1, 0, 0]
elif cb == 4:
	d = d + [0, 0, 0, 1, 0]
else:
	d = d + [0, 0, 0, 0, 1]


hdd = int(input("Enter you hdd capacity : "))

d = d + [hdd]

ssd = int(input("Enter your ssd capacity : "))

d = d + [ssd]

gb = int(input("GPU-branch : Enter  1 for AMD  2 for Intel  3 for Nvidia : "))

if gb == 1:
	d = d + [1, 0, 0]
elif d == 2:
	d = d + [0, 1, 0]
else:
	d = d + [0, 0, 1]


os = int(input("OS : Enter  1 for Mac  2 for Others  3 for Windows : "))

if os == 1:
	d = d + [1, 0, 0]
elif os == 2:
	d = d + [0, 1, 0]
else:
	d = d + [0, 0, 1]

ans1 = mms.fit_transform([d])

ans = model.predict(ans1)
print(ans)
