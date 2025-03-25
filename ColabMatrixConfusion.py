import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix
from tensorflow.keras import datasets, layers, models
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from itertools import cycle

log_dir = 'logs/'
os.system(f"tensorboard --logdir={log_dir}")

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

train_images, test_images = train_images / 255.0, test_images / 255.0

classes=[0,1,2,3,4,5,6,7,8,9]

model =  models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x=train_images,
          y=train_labels,
          epochs=5,
          validation_data=(test_images, test_labels))


y_test = test_labels
y_pred = model.predict(test_images)
y_pred = np.argmax(y_pred, axis=1)

# --- CONFUSION MATRIX

mcm = multilabel_confusion_matrix(y_test, y_pred)

for i, cm in enumerate(mcm):
  TN, FP, FN, TP = cm.ravel()
  sensi = TP / (TP + FN)
  espec = TN / (FP + TN)
  acc = (TP + TN) / (TP+TN+FP+FN)
  prec = TP / (TP + FP)
  fscore = 2 * (prec * sensi) / (prec + sensi)
  print(f"Classe {i}: Sensibilidade={sensi}, Especificidade={espec}, Accuracy={acc}, Precisão={prec}, F-score={fscore}")


con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()
con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)

con_mat_df = pd.DataFrame(con_mat_norm,
                          index= classes,
                          columns = classes)


figure = plt.Figure(figsize=(8, 8)) 
sns.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues)v
plt.tight_layout()
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- ROC CURVE PLOT

n_classes = len(classes)
y_test_bin = label_binarize(y_test, classes=range(n_classes))

y_pred_proba = model.predict(test_images)

colors = cycle(["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "cyan", "black"])

plt.figure(figsize=(8, 8))

for i, color in zip(range(n_classes), colors):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'Classe {i} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], "k--", lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Falso Positivo (FPR)")
plt.ylabel("Verdadeiro Positivo (TPR)")
plt.title("Curva ROC Multiclasse")
plt.legend(loc="lower right")
plt.show()