"""
Script untuk generate visualisasi hasil klasifikasi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import metrics
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create visualizations directory
if not os.path.exists('./visualizations'):
    os.makedirs('./visualizations')

print("Loading dataset...")
# Load data
df = pd.read_csv("./data/data-pemilih-kpu.csv", encoding='utf-8-sig')
df = df.dropna(how='all')

# Map labels
jk_map = {"Laki-Laki": 1, "Perempuan": 0}
df["jenis_kelamin"] = df["jenis_kelamin"].map(jk_map)

# Feature and target
X = df[["nama"]].values
y = df[["jenis_kelamin"]].values

# Split data
text_train, text_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

print("Training models...")

# Train Logistic Regression
clf_lg = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(2, 6))),
    ('clf', LogisticRegression())
])
clf_lg.fit(text_train.ravel(), y_train.ravel())
pred_lg = clf_lg.predict(text_test.ravel())
acc_lg = metrics.accuracy_score(y_test, pred_lg)

# Train Naive Bayes
clf_nb = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(2, 6))),
    ('clf', MultinomialNB())
])
clf_nb.fit(text_train.ravel(), y_train.ravel())
pred_nb = clf_nb.predict(text_test.ravel())
acc_nb = metrics.accuracy_score(y_test, pred_nb)

# Train Random Forest
clf_rf = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(2, 6))),
    ('clf', RandomForestClassifier(n_estimators=90, n_jobs=-1, random_state=42))
])
clf_rf.fit(text_train.ravel(), y_train.ravel())
pred_rf = clf_rf.predict(text_test.ravel())
acc_rf = metrics.accuracy_score(y_test, pred_rf)

print(f"Logistic Regression Accuracy: {acc_lg:.4f}")
print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")

# ===== VISUALIZATION 1: Data Distribution =====
print("\nGenerating data distribution chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution in original dataset
gender_counts = df['jenis_kelamin'].value_counts()
labels = ['Perempuan', 'Laki-Laki']
colors = ['#FF6B6B', '#4ECDC4']

axes[0].pie(gender_counts.values, labels=labels, autopct='%1.2f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 12})
axes[0].set_title('Distribusi Jenis Kelamin dalam Dataset\n(Total: {} data)'.format(len(df)), 
                   fontsize=14, fontweight='bold')

# Bar chart
axes[1].bar(labels, gender_counts.values, color=colors, alpha=0.7, edgecolor='black')
axes[1].set_ylabel('Jumlah Data', fontsize=12)
axes[1].set_title('Jumlah Data per Jenis Kelamin', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(gender_counts.values):
    axes[1].text(i, v + 50, str(v), ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('./visualizations/data_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/data_distribution.png")

# ===== VISUALIZATION 2: Confusion Matrix - Logistic Regression =====
print("Generating confusion matrix - Logistic Regression...")
cm_lg = metrics.confusion_matrix(y_test, pred_lg, labels=[1, 0])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_lg, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Jumlah'},
            xticklabels=['Perempuan', 'Laki-Laki'],
            yticklabels=['Perempuan', 'Laki-Laki'],
            annot_kws={'fontsize': 14})
plt.title('Confusion Matrix - Logistic Regression\nAkurasi: {:.2f}%'.format(acc_lg * 100),
          fontsize=14, fontweight='bold')
plt.ylabel('Actual (True Label)', fontsize=12)
plt.xlabel('Predicted (Model Prediction)', fontsize=12)
plt.tight_layout()
plt.savefig('./visualizations/confusion_matrix_lr.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/confusion_matrix_lr.png")

# ===== VISUALIZATION 3: Confusion Matrix - Naive Bayes =====
print("Generating confusion matrix - Naive Bayes...")
cm_nb = metrics.confusion_matrix(y_test, pred_nb, labels=[1, 0])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': 'Jumlah'},
            xticklabels=['Perempuan', 'Laki-Laki'],
            yticklabels=['Perempuan', 'Laki-Laki'],
            annot_kws={'fontsize': 14})
plt.title('Confusion Matrix - Naive Bayes\nAkurasi: {:.2f}%'.format(acc_nb * 100),
          fontsize=14, fontweight='bold')
plt.ylabel('Actual (True Label)', fontsize=12)
plt.xlabel('Predicted (Model Prediction)', fontsize=12)
plt.tight_layout()
plt.savefig('./visualizations/confusion_matrix_nb.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/confusion_matrix_nb.png")

# ===== VISUALIZATION 4: Confusion Matrix - Random Forest =====
print("Generating confusion matrix - Random Forest...")
cm_rf = metrics.confusion_matrix(y_test, pred_rf, labels=[1, 0])

plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Oranges', cbar_kws={'label': 'Jumlah'},
            xticklabels=['Perempuan', 'Laki-Laki'],
            yticklabels=['Perempuan', 'Laki-Laki'],
            annot_kws={'fontsize': 14})
plt.title('Confusion Matrix - Random Forest\nAkurasi: {:.2f}%'.format(acc_rf * 100),
          fontsize=14, fontweight='bold')
plt.ylabel('Actual (True Label)', fontsize=12)
plt.xlabel('Predicted (Model Prediction)', fontsize=12)
plt.tight_layout()
plt.savefig('./visualizations/confusion_matrix_rf.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/confusion_matrix_rf.png")

# ===== VISUALIZATION 5: Accuracy Comparison =====
print("Generating accuracy comparison chart...")
models = ['Logistic\nRegression', 'Naive\nBayes', 'Random\nForest']
accuracies = [acc_lg * 100, acc_nb * 100, acc_rf * 100]
colors_acc = ['#3498db', '#2ecc71', '#e74c3c']

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Bar chart
bars = axes[0].bar(models, accuracies, color=colors_acc, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel('Akurasi (%)', fontsize=12)
axes[0].set_title('Perbandingan Akurasi Model', fontsize=14, fontweight='bold')
axes[0].set_ylim([90, 95])
axes[0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

# Line chart
axes[1].plot(models, accuracies, marker='o', linewidth=2, markersize=10, color='#9b59b6')
axes[1].set_ylabel('Akurasi (%)', fontsize=12)
axes[1].set_title('Tren Akurasi Model', fontsize=14, fontweight='bold')
axes[1].set_ylim([90, 95])
axes[1].grid(True, alpha=0.3)

# Add value labels on points
for i, (model, acc) in enumerate(zip(models, accuracies)):
    axes[1].text(i, acc + 0.1, f'{acc:.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('./visualizations/accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/accuracy_comparison.png")

# ===== VISUALIZATION 6: Performance Metrics Comparison =====
print("Generating performance metrics comparison...")

# Get precision, recall, f1-score for each model
report_lg = metrics.classification_report(y_test, pred_lg, labels=[1, 0], output_dict=True)
report_nb = metrics.classification_report(y_test, pred_nb, labels=[1, 0], output_dict=True)
report_rf = metrics.classification_report(y_test, pred_rf, labels=[1, 0], output_dict=True)

metrics_data = {
    'Logistic Regression': [
        report_lg['weighted avg']['precision'],
        report_lg['weighted avg']['recall'],
        report_lg['weighted avg']['f1-score']
    ],
    'Naive Bayes': [
        report_nb['weighted avg']['precision'],
        report_nb['weighted avg']['recall'],
        report_nb['weighted avg']['f1-score']
    ],
    'Random Forest': [
        report_rf['weighted avg']['precision'],
        report_rf['weighted avg']['recall'],
        report_rf['weighted avg']['f1-score']
    ]
}

metric_names = ['Precision', 'Recall', 'F1-Score']
x = np.arange(len(metric_names))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))

bars1 = ax.bar(x - width, metrics_data['Logistic Regression'], width, label='Logistic Regression', color='#3498db', alpha=0.8)
bars2 = ax.bar(x, metrics_data['Naive Bayes'], width, label='Naive Bayes', color='#2ecc71', alpha=0.8)
bars3 = ax.bar(x + width, metrics_data['Random Forest'], width, label='Random Forest', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Perbandingan Metrik Performa Model', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim([0.90, 0.96])
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('./visualizations/performance_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/performance_metrics.png")

# ===== VISUALIZATION 7: Train vs Test Accuracy =====
print("Generating train vs test accuracy comparison...")

# Calculate training accuracies
train_pred_lg = clf_lg.predict(text_train.ravel())
train_acc_lg = metrics.accuracy_score(y_train, train_pred_lg)

train_pred_nb = clf_nb.predict(text_train.ravel())
train_acc_nb = metrics.accuracy_score(y_train, train_pred_nb)

train_pred_rf = clf_rf.predict(text_train.ravel())
train_acc_rf = metrics.accuracy_score(y_train, train_pred_rf)

models_short = ['LR', 'NB', 'RF']
train_scores = [train_acc_lg * 100, train_acc_nb * 100, train_acc_rf * 100]
test_scores = [acc_lg * 100, acc_nb * 100, acc_rf * 100]

x = np.arange(len(models_short))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, train_scores, width, label='Training Accuracy', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, test_scores, width, label='Testing Accuracy', color='#e74c3c', alpha=0.8)

ax.set_ylabel('Akurasi (%)', fontsize=12)
ax.set_title('Training vs Testing Accuracy\n(LR: Logistic Regression, NB: Naive Bayes, RF: Random Forest)', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_short)
ax.legend(fontsize=10)
ax.set_ylim([90, 101])
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('./visualizations/train_vs_test_accuracy.png', dpi=300, bbox_inches='tight')
print("✓ Saved: visualizations/train_vs_test_accuracy.png")

print("\n" + "="*60)
print("✓ All visualizations generated successfully!")
print("="*60)
print("\nVisualization files saved in ./visualizations/:")
print("  1. data_distribution.png")
print("  2. confusion_matrix_lr.png")
print("  3. confusion_matrix_nb.png")
print("  4. confusion_matrix_rf.png")
print("  5. accuracy_comparison.png")
print("  6. performance_metrics.png")
print("  7. train_vs_test_accuracy.png")
print("\nYou can use these images in your LAPORAN.md")

