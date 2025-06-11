import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid')
import shutil
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)
print('All modules have been imported')

# Renkli metin yazdırma fonksiyonu
def print_in_color(txt_msg, fore_tupple=(0,255,255), back_tupple=(100,100,100)):
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(bb) + 'm'
    print(msg.format(mat), flush=True)
    print('\33[0m', flush=True)
    return

msg = 'Test of default colors'
print_in_color(msg)

# Veri çerçeveleri oluştur
def make_dataframes(base_dir):
    splits = ['train', 'validation', 'test']
    dataframes = {}
    total_counts = {}
    class_distributions = {}
    
    classlist = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    
    for split in splits:
        filepaths = []
        labels = []
        class_dist = {}
        total = 0
        
        for klass in classlist:
            sklass = klass[:25]
            split_dir = os.path.join(base_dir, klass, split)
            if not os.path.exists(split_dir):
                print(f"Error: Directory {split_dir} does not exist.")
                exit(1)
            
            flist = sorted(os.listdir(split_dir))
            count = len(flist)
            class_dist[sklass] = count
            total += count
            
            desc = f'{split:6s}-{sklass:25s}'
            for f in tqdm(flist, ncols=130, desc=desc, unit='files', colour='blue'):
                fpath = os.path.join(split_dir, f)
                filepaths.append(fpath)
                labels.append(sklass)
        
        total_counts[split] = total
        class_distributions[split] = class_dist
        
        Fseries = pd.Series(filepaths, name='filepaths')
        Lseries = pd.Series(labels, name='labels')
        df = pd.concat([Fseries, Lseries], axis=1)
        dataframes[split] = df
    
    train_df = dataframes['train']
    valid_df = dataframes['validation']
    test_df = dataframes['test']
    
    classes = sorted(train_df['labels'].unique())
    class_count = len(classes)
    
    sample_df = train_df.sample(n=min(50, len(train_df)), replace=False)
    ht = wt = count = 0
    for i in range(len(sample_df)):
        fpath = sample_df['filepaths'].iloc[i]
        try:
            img = cv2.imread(fpath)
            h, w = img.shape[:2]
            wt += w
            ht += h
            count += 1
        except:
            print(f"Warning: Could not read image {fpath}")
            continue
    have = int(ht / count) if count > 0 else 0
    wave = int(wt / count) if count > 0 else 0
    aspect_ratio = have / wave if wave > 0 else 0
    
    print('\nDataset Summary:')
    for split in splits:
        print(f'{split.capitalize()} dataset: {total_counts[split]} images')
        print(f'Class distribution in {split} dataset:')
        for cls, count in class_distributions[split].items():
            print(f'  {cls:25s}: {count} images')
    print('\nProcessed Dataset Details:')
    print('Number of classes:', class_count)
    print('Train_df length:', len(train_df))
    print('Valid_df length:', len(valid_df))
    print('Test_df length:', len(test_df))
    print('Average image height:', have)
    print('Average image width:', wave)
    print('Aspect ratio (h/w):', aspect_ratio)
    
    return train_df, test_df, valid_df, classes, class_count

base_dir = './dataset_balanced_20004'
train_df, test_df, valid_df, classes, class_count = make_dataframes(base_dir)

# Etiketleri kontrol et
def check_labels(df, num_samples=5):
    classes = df['labels'].unique()
    plt.figure(figsize=(15, 5))
    for i, cls in enumerate(classes[:num_samples]):
        sample = df[df['labels'] == cls].sample(1)
        img_path = sample['filepaths'].iloc[0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(img)
        plt.title(cls)
        plt.axis('off')
    plt.savefig('label_check.png')
    plt.show()

check_labels(train_df)

# Generatörleri oluştur
def make_gens(batch_size, train_df, test_df, valid_df, img_size):
    ycol = 'labels'
    trgen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        preprocessing_function=lambda x: x/255.0
    )
    t_and_v_gen = ImageDataGenerator(preprocessing_function=lambda x: x/255.0)
    print_in_color('Creating train generator...')
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                         class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    print_in_color('Creating valid generator...')
    valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                               class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    length = len(test_df)
    test_batch_size = sorted([int(length/n) for n in range(1, length+1) if length % n == 0 and length/n <= 80], reverse=True)[0]
    test_steps = int(length / test_batch_size)
    print_in_color('Creating test generator...')
    test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col=ycol, target_size=img_size,
                                              class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size)
    classes = list(train_gen.class_indices.keys())
    class_count = len(classes)
    print('Test batch size:', test_batch_size, 'Test steps:', test_steps, 'Number of classes:', class_count)
    return train_gen, test_gen, valid_gen, test_batch_size, test_steps, classes

batch_size = 32
train_gen, test_gen, valid_gen, test_batch_size, test_steps, classes = make_gens(batch_size, train_df, test_df, valid_df, img_size=(224, 224))

# Örnek görüntüleri göster
def show_image_samples(gen):
    t_dict = gen.class_indices
    classes = list(t_dict.keys())
    images, labels = next(gen)
    plt.figure(figsize=(25, 25))
    length = len(labels)
    r = min(length, 25)
    for i in range(r):
        plt.subplot(5, 5, i + 1)
        image = images[i]
        plt.imshow(image)
        index = np.argmax(labels[i])
        class_name = classes[index]
        plt.title(class_name, color='blue', fontsize=18)
        plt.axis('off')
    plt.savefig('sample_images.png')
    plt.show()

show_image_samples(train_gen)

# Modeli oluştur
def make_model(img_size, class_count):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size[0], img_size[1], 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(class_count, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=outputs)
    return model

model = make_model(img_size=(224, 224), class_count=class_count)

# Modeli derle
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Özel geri çağırma sınıfı
class LR_ASK(keras.callbacks.Callback):
    def __init__(self, model_instance, epochs, factor=0.4, dwell=True, ask_epoch=2):
        super(LR_ASK, self).__init__()
        self.model_instance = model_instance
        self.epochs = epochs
        self.factor = factor
        self.dwell = dwell
        self.ask_epoch = ask_epoch
        self.lowest_vloss = np.inf
        self.best_weights = None
        self.best_epoch = 0
        self.count = 0

    def on_train_begin(self, logs=None):
        print_in_color(f'Training will proceed until epoch {self.ask_epoch} then you will be asked to continue or halt.')
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        try:
            lr = float(self.model_instance.optimizer.learning_rate.numpy())
        except AttributeError:
            lr = float(tf.keras.backend.get_value(self.model_instance.optimizer.learning_rate))
        v_loss = logs.get('val_loss')
        if v_loss < self.lowest_vloss:
            self.lowest_vloss = v_loss
            self.best_weights = self.model_instance.get_weights()
            self.best_epoch = epoch + 1
            print_in_color(f'Validation loss improved to {v_loss:.4f}, saving weights at epoch {self.best_epoch}')
        elif self.dwell:
            self.model_instance.set_weights(self.best_weights)
            new_lr = lr * self.factor
            try:
                self.model_instance.optimizer.learning_rate.assign(new_lr)
            except AttributeError:
                tf.keras.backend.set_value(self.model_instance.optimizer.learning_rate, new_lr)
            print_in_color(f'Validation loss did not improve, resetting weights and reducing lr to {new_lr:.6f}')
            self.count += 1
            if self.count >= 3:
                print_in_color('No improvement after 3 epochs, stopping training.')
                self.model_instance.stop_training = True

        if epoch + 1 == self.ask_epoch:
            print_in_color(f'Epoch {epoch + 1} completed. Continue training? (y/n) or enter new epochs:')
            ans = input()
            if ans.lower() == 'n':
                self.model_instance.stop_training = True
                print_in_color('Training halted by user.')
            elif ans.isdigit():
                self.ask_epoch = int(ans)
                print_in_color(f'Training will continue until epoch {self.ask_epoch}')

    def on_train_end(self, logs=None):
        self.model_instance.set_weights(self.best_weights)
        duration = time.time() - self.start_time
        print_in_color(f'Training completed in {duration / 60:.2f} minutes. Model weights set to best at epoch {self.best_epoch}')

# Geri çağırmaları tanımla
lr_ask = LR_ASK(model, epochs=50, ask_epoch=3)
checkpoint = keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min')
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Modeli eğit
history = model.fit(
    train_gen,
    epochs=5,
    validation_data=valid_gen,
    callbacks=[lr_ask, checkpoint, early_stopping],
    verbose=1
)

# Eğitim grafiklerini çiz
def plot_training(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_plot.png')
    plt.show()

plot_training(history)

# Test setinde tahmin yap, confusion matrix ve classification report oluştur
def evaluate_model(test_gen, test_steps, classes):
    y_pred = model.predict(test_gen, steps=test_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('confusion_matrix.png')
    plt.show()
    print('\nClassification Report:')
    print(classification_report(y_true, y_pred_classes, target_names=classes))

evaluate_model(test_gen, test_steps, classes)

# Hatalı sınıflandırılan dosyaları listele ve örnekleri göster
def get_misclassified(test_gen, test_steps, model, classes):
    test_gen.reset()
    y_pred = model.predict(test_gen, steps=test_steps)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_gen.classes
    filepaths = test_gen.filepaths
    misclassified = []
    for i in range(len(y_true)):
        if y_pred_classes[i] != y_true[i]:
            misclassified.append({
                'filepath': filepaths[i],
                'true': classes[y_true[i]],
                'predicted': classes[y_pred_classes[i]]
            })
    print(f'Total misclassified images: {len(misclassified)}')
    if misclassified:
        print('\nExample Misclassified Images:')
        plt.figure(figsize=(15, 5))
        for i, item in enumerate(misclassified[:5]):
            img = cv2.imread(item['filepath'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.title(f"True: {item['true']}\nPred: {item['predicted']}", fontsize=10)
            plt.axis('off')
        plt.savefig('misclassified_images.png')
        plt.show()
    return misclassified

misclassified = get_misclassified(test_gen, test_steps, model, classes)

# Modeli kaydet
model.save('final_dermatology_model.keras')
print_in_color('Model saved as final_dermatology_model.keras')
