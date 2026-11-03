import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

df = pd.read_csv("Titanic-Dataset.csv")

print("Dataset shakli:", df.shape)
print(df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].describe())

X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].copy()
y = df['Survived']

X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
TN, FP, FN, TP = cm.ravel()

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
accuracy = (TP + TN) / (TP + TN + FP + FN)

print(f"\nTP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall: {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-score: {f1:.4f} ({f1*100:.2f}%)")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
fig.patch.set_facecolor('#c5e0b4')
ax.set_facecolor('#c5e0b4')
ax.axis('off')

def draw_cell(x, y, label, value, subtext, color):
    box = FancyBboxPatch((x, y), 0.36, 0.34, boxstyle="round,pad=0.02,rounding_size=0.025", color=color, ec="none")
    ax.add_patch(box)
    ax.text(x + 0.18, y + 0.19, f"{label}  {value}", color='white', fontsize=16, fontweight='bold', ha='center', va='center', family='serif')
    ax.text(x + 0.18, y + 0.08, subtext, color='white', fontsize=10, ha='center', va='center', family='serif')

green = '#548235'
orange = '#e35205'
header_color = '#007078'

draw_cell(0.21, 0.48, "TP", TP, "To'g'ri musbat", green)
draw_cell(0.61, 0.48, "FN", FN, "Noto'g'ri manfiy", orange)
draw_cell(0.21, 0.06, "FP", FP, "Noto'g'ri musbat", orange)
draw_cell(0.61, 0.06, "TN", TN, "To'g'ri manfiy", green)

ax.text(0.39, 0.91, "BASHORAT: Olgan", color=header_color, fontsize=10, fontweight='bold', ha='center')
ax.text(0.79, 0.91, "BASHORAT: Olmagan", color=header_color, fontsize=10, fontweight='bold', ha='center')

ax.text(0.10, 0.65, "HAQIQAT: Olgan", color=header_color, fontsize=10, fontweight='bold', va='center', ha='center')
ax.text(0.10, 0.23, "HAQIQAT: Olmagan", color=header_color, fontsize=10, fontweight='bold', va='center', ha='center')

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("confusion_matrix_titanic.png", facecolor=fig.get_facecolor(), edgecolor='none')
plt.show()
