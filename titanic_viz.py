import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set style for all plots
sns.set(style="whitegrid")

# Load dataset
df = sns.load_dataset("titanic")

# =========================
# 1. Bar Plot: Survival by Class
# =========================
plt.figure(figsize=(8, 5))
sns.countplot(x="class", hue="survived", data=df)
plt.title("Survival Count by Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.tight_layout()
plt.show()

# =========================
# 2. Pie Chart: Gender Distribution
# =========================
gender_counts = df['sex'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Gender Distribution on Titanic")
plt.axis('equal')
plt.tight_layout()
plt.show()

# =========================
# 3. Heatmap: Feature Correlation
# =========================
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()
plt.show()

# =========================
# 4. Box Plot: Age Distribution by Survival
# =========================
plt.figure(figsize=(8, 5))
sns.boxplot(x="survived", y="age", data=df)
plt.title("Age Distribution by Survival Status")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

# =========================
# 5. Insight Printout (Optional for storytelling)
# =========================
print("\n--- Insights Summary ---")
print("1. First-class passengers had higher survival rates.")
print("2. Women survived at a much higher rate than men.")
print("3. Age shows a trend: children had better chances of survival.")
print("4. Strong correlation between fare and class, weak between age and survival.")
