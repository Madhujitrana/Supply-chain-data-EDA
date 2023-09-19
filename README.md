# Supply-chain-data-EDA
To perform the EDA in python i have taken this dataset from kaggle and here i have try to do find out the insight of some business problems

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer as knn


#### The main focus to EDA of netflix dataset to get the insight of it.

### Step invoive loading the dataset,Cleaning ,univeriety,biveriety and multivariety analysis.


#### Loading the dataset

df=pd.read_csv("C:/Users/Madhujit/Desktop/DataCoSupplyChainDataset.csv",encoding="ISO-8859-1")




## Checking the columns 

df.columns

### TOtal shape of the dataset

df.shape

### checking the info of the dataset
*0
df.info()
 
### checking the null value in the dataset

0
df.isnull().sum()

### checking the duplicate value in the dataset

df.duplicated().sum()


### checking the outlier in the dataset

df.columns=df.columns.str.replace(" ","_")

df.columns

numeric_data=df.select_dtypes(exclude="object")

col=numeric_data.columns

def function (data,co):
    q1=data[co].quantile(0.25)
    q3=data[co].quantile(0.75)
    iqr=q3-q1
    upper_bound=q3+1.5*iqr
    lower_bound=q1-1.5*iqr
    x=data[(data[co]>=lower_bound)|(data[co]<=upper_bound)]
    return x

for i in col:
    new=function(df,i)

new.isnull().sum()


new.columns=new.columns.str.replace(" ","_")

new.isnull().sum()

### Dropping the unnecessary columns

new.drop(["Customer_Lname","Customer_Zipcode","Order_Zipcode","Product_Description"],axis=1,inplace=True)

###Checking null values


new.isnull().sum()

### Checking duplicates

new.duplicated().sum()


###Top 10 most selling product Name based on orders volumes

fig,ax=plt.subplots(figsize=(20,8))
new["Product_Name"].value_counts()[:10].plot(kind="bar",color=["red","green","purple"])
ax.bar_label(ax.containers[0],fontsize=15)
plt.xlabel("products_name",fontsize=20)
plt.ylabel("Total_orders",fontsize=20)
plt.xticks(fontsize=20)
plt.title("Top 10 most selling products",fontsize=20)
plt.show()

###Top 10 most selling products based on qty

fig,ax=plt.subplots(figsize=(25,12))
custom_colors = ["Purple", "Green", "Blue", "Orange", "Red"]
new.groupby("Product_Name").agg({"Order_Item_Quantity":"sum"}).sort_values("Order_Item_Quantity",ascending=False)[:10].plot(kind="bar",color=custom_colors,ax=ax)

ax.bar_label(ax.containers[0],fontsize=30)
plt.xlabel("products_name",fontsize=30)
plt.ylabel("Total_orders",fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=20)
plt.title("Top 10 most selling products based on volume",fontsize=30)
plt.show()

####Top 10 most selling product Name based on profit


fig,ax=plt.subplots(figsize=(25,13))
custom_colors = ["Purple", "Green", "Blue", "Orange", "Red"]
new.groupby("Product_Name").agg({"Order_Profit_Per_Order":"sum"}).sort_values("Order_Profit_Per_Order",ascending=False)[:10].plot(kind="bar",color="Pink",ax=ax)
ax.bar_label(ax.containers[0],fontsize=30)
plt.xlabel("products_name",fontsize=30)
plt.ylabel("Total_orders",fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=20)
plt.title("Top 10 most selling products based on Profit",fontsize=35)
plt.show()


###Top country where the order volume is more 

orders1=new["Order_Id"]
country=new[["Order_Id","Customer_Country"]]
country.duplicated().sum()

data=country.drop_duplicates(keep="first").reset_index()


fig,ax=plt.subplots(figsize=(15,13))
data["Customer_Country"].value_counts().plot(kind="bar",color="Pink",ax=ax)
ax.bar_label(ax.containers[0],fontsize=30)
plt.xlabel("Country_name",fontsize=30)
plt.ylabel("Total_orders",fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=20)
plt.title("Top countries based on order volume",fontsize=35)
plt.show()


###Top 10 city from which orders are mostly generated
new.columns
country=new[["Order_Id","Customer_City"]]

country.duplicated().sum()

data=country.drop_duplicates(keep="first").reset_index()


fig,ax=plt.subplots(figsize=(15,8))
data["Customer_City"].value_counts()[:10].plot(kind="bar",color=["Pink","red","blue"],ax=ax)
ax.bar_label(ax.containers[0],fontsize=20)
plt.xlabel("Country_name",fontsize=30)
plt.ylabel("Total_orders",fontsize=30)
plt.xticks(fontsize=30)
plt.yticks(fontsize=20)
plt.title("Top 10 city based on order volume",fontsize=35)
plt.show()

###Top 10 non profit product id 

new.columns

not_profit=new[new["Benefit_per_order"]<0]

non_profit_final=not_profit[["Benefit_per_order","Product_Name"]]

fig,ax=plt.subplots(figsize=(18,8))
non_profit_final.groupby("Product_Name").agg({"Benefit_per_order":"sum"}).sort_values("Benefit_per_order",ascending=True)[:10].plot(kind="bar",color=["red","red","blue"],ax=ax,width=0.3)
ax.bar_label(ax.containers[0],fontsize=20)
plt.xlabel("Product_name",fontsize=20)
plt.ylabel("Total_non_profit",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Top 10 non profit product id ",fontsize=20)
plt.show()


####Top Best selling category

best_category=new[["Order_Item_Quantity","Category_Name"]]
best_category["Category_Name"].nunique()




fig,ax=plt.subplots(figsize=(18,8))
best_category.groupby("Category_Name").agg({"Order_Item_Quantity":"sum"}).sort_values("Order_Item_Quantity",ascending=False)[:10].plot(kind="bar",color='C5',ax=ax,width=0.3)
ax.bar_label(ax.containers[0],fontsize=20)
plt.xlabel("Category_Name",fontsize=20)
plt.ylabel("Order_Item_Quantity",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Top 10 Best selling category ",fontsize=20)
plt.show()


###Business loss due to late delivery

new.columns

loss_busniess=new[new["Late_delivery_risk"]==1]
loss_busniess["Late_delivery_risk"]

loss_business=loss_busniess[loss_busniess["Benefit_per_order"]<0]

loss_business

Total_loss=sum(loss_business["Benefit_per_order"])

df2=loss_business["Order_Status"].value_counts().reset_index()

fig,ax=plt.subplots(figsize=(25,5))
sns.barplot(data=df2,x="index",y="Order_Status",ax=ax,width=0.3)
ax.bar_label(ax.containers[0],fontsize=20)
plt.xlabel("Order_status",fontsize=20)
plt.ylabel("Order_Item_Quantity",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Factor effecting ",fontsize=20)
plt.show()


###Profitable customer segment

p_c_s=new[new["Benefit_per_order"]>0]


fig,ax=plt.subplots(figsize=(18,8))
p_c_s.groupby("Customer_Segment").agg({"Benefit_per_order":"sum"}).sort_values("Benefit_per_order",ascending=False)[:10].plot(kind="bar",color='C5',ax=ax)
ax.bar_label(ax.containers[0],fontsize=20)
plt.xlabel("Customer_Segment",fontsize=20)
plt.ylabel("Benefit_per_order",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Top Profitable customer segment ",fontsize=20)
plt.show()


total=p_c_s.groupby("Customer_Segment").agg({"Benefit_per_order":"sum"})


total["Percentage"]=total["Benefit_per_order"]/sum(total["Benefit_per_order"])


total=total.reset_index()

new=total[["Customer_Segment","Percentage"]]


explode = [0, 0.1, 0.1]
plt.pie(new["Percentage"],labels=new["Customer_Segment"],autopct='%.0f%%',explode=explode)
plt.title("profitable customer segment",fontsize=10)

#### High profit department

high_profit_dep=new[["Department_Name","Benefit_per_order"]]

high_profit_dep=high_profit_dep[high_profit_dep["Benefit_per_order"]>0]

high_profit_dep["percentage"]=high_profit_dep["Benefit_per_order"]/sum(high_profit_dep["Benefit_per_order"])*100

fig,ax=plt.subplots(figsize=(18,8))
high_profit_dep.groupby("Department_Name").agg({"percentage":"sum"}).sort_values("percentage",ascending=False)[:10].plot(kind="bar",color='C2',ax=ax)
ax.bar_label(ax.containers[0],fontsize=20)
plt.xlabel("Department_Name",fontsize=20)
plt.ylabel("percentage",fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title("Top Profitable customer segment ",fontsize=20)
plt.show()


####Which month the order volume is to high

new.columns

new["month"]=pd.to_datetime(new["order_date_(DateOrders)"])
new["month"]=new["month"].dt.strftime("%b")

order_month=new[["Order_Id","month"]]

order_month.duplicated().sum()

order_month=order_month.drop_duplicates(keep="first")

order_count=order_month.groupby("month").agg({"Order_Id":"count"}).reset_index().sort_values("Order_Id",ascending=False)



order_count["percentage"]=order_count["Order_Id"]/sum(order_count["Order_Id"])*100
explode=[1.0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
plt.pie(order_count["percentage"],labels=order_count["month"],explode=explode,autopct="%2.1f%%")
plt.xticks(fontsize=10)
plt.title("Order volume based on month",fontsize=10)

