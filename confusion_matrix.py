import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

X_test, y_test = joblib.load("test_set.pkl")
le = joblib.load("music_classifier_svm_label_encoder.pkl")
clf = joblib.load("music_classifier_svm.pkl")

y_pred = clf.predict(X_test)

print('Accuracy: ', accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=le.classes_,
            yticklabels=le.classes_,
            cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', dpi=300)
print('Saved sns heatmap confusion_matrix.png')
