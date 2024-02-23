import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
import numpy as np

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from google.colab import drive
drive.mount('/content/drive')

base_dir = '/content/drive/My Drive/Datasets/Dataset Flores Pronto'
train_dir = os.path.join(base_dir, 'Treino')
validation_dir = os.path.join(base_dir, 'Validacao')
test_dir = os.path.join(base_dir, 'Teste')

# Carregar a arquitetura da VGG16 pré-treinada
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar camadas convolucionais
for layer in base_model.layers:
    layer.trainable = False


# Congelar camadas convolucionais
for layer in base_model.layers:
    layer.trainable = False

# Adicionar camadas personalizadas
x = Flatten()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = Dense(5, activation='softmax')(x)  # N é o número de classes

# Criar o modelo final
model = Model(inputs=base_model.input, outputs=x)

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1.0/255)

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(224, 224),
                                                    batch_size=10,
                                                    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                             target_size=(224, 224),
                                                             batch_size=32,
                                                             class_mode='categorical')

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(224, 224),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle = False)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


epochs = 20

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    verbose=1
)

test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))
print(f'Test accuracy: {test_accuracy}')

# Prever as classes para o conjunto de teste
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Calcular métricas de classificação
precision = precision_score(y_true, y_pred_classes, average='weighted')
f1 = f1_score(y_true, y_pred_classes, average='weighted')
confusion = confusion_matrix(y_true, y_pred_classes)

print(f'Precision: {precision}')
print(f'F1-score: {f1}')
print('Confusion Matrix:')
print(confusion)