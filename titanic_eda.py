import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

# -------------------------
# Basic data understanding
# -------------------------
print("First 5 rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values in dataset:")
print(df.isnull().sum())

# -------------------------
# Data Cleaning
# -------------------------
# Fill missing Age values with mean
df["Age"].fillna(df["Age"].mean(), inplace=True)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# -------------------------
# Exploratory Data Analysis
# -------------------------

# Survival count
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.show()

# Survival by gender
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.show()

# Survival by passenger class
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.show()

# Age distribution
plt.hist(df["Age"], bins=8, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Fare vs survival
sns.boxplot(x="Survived", y="Fare", data=df)
plt.title("Fare vs Survival")
plt.show()
