#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"
# os.environ["QT_QPA_PLATFORM"] = "xcb"
# os.environ["LD_LIBRARY_PATH"] = f"/home/celik/miniconda3/envs/AI3/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"
import time
import matplotlib
matplotlib.use('Agg') # GUI olmayan ortamlar için
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
sns.set_style('darkgrid') # Seaborn stilini global olarak ayarla
import shutil
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import warnings
warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# TensorFlow Addons varlığını kontrol et (bilgi amaçlı)
try:
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy
    TFA_AVAILABLE = True
    print("TensorFlow Addons found (SigmoidFocalCrossEntropy check successful). Custom Categorical Focal Loss will be used.")
except ImportError:
    TFA_AVAILABLE = False
    print("TensorFlow Addons not found. Custom Categorical Focal Loss will be used.")

# Özel Kategorik Focal Loss implementasyonu (Global kapsamda tanımlı)
def categorical_focal_loss(gamma=2.0, alpha=0.25):
    """
    Softmax aktivasyonu ile çok sınıflı odak kaybı.
    """
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_loss_val = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(focal_loss_val)
    return focal_loss

print('All modules have been imported')

def print_in_color(txt_msg, fore_tupple=(0,255,255), back_tupple=(100,100,100)):
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(bb) + 'm'
    print(msg.format(mat), flush=True)
    print('\33[0m', flush=True)
    return

msg = 'Test of default colors for console output.'
print_in_color(msg)

def make_dataframes(base_dir, use_undersampling=False, train_target_count=1960, valid_target_count=420):
    splits = ['train', 'validation', 'test']
    dataframes = {}
    if not os.path.isdir(base_dir):
        print_in_color(f"Error: Base directory {base_dir} does not exist.", fore_tupple=(255,0,0)); exit(1)
    classlist = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    if not classlist:
        print_in_color(f"Error: No class subdirectories found in {base_dir}.", fore_tupple=(255,0,0)); exit(1)

    for split in splits:
        filepaths, labels = [], []
        for klass in classlist:
            split_dir = os.path.join(base_dir, klass, split)
            if not os.path.exists(split_dir):
                print_in_color(f"Warning: Dir {split_dir} not found for class {klass}. Skipping.", fore_tupple=(255,165,0)); continue
            flist = sorted(os.listdir(split_dir))
            desc = f'{split:10s}-{klass:25s}'
            for f in tqdm(flist, ncols=130, desc=desc, unit='files', colour='green'):
                filepaths.append(os.path.join(split_dir, f)); labels.append(klass)
        df = pd.DataFrame({'filepaths': filepaths, 'labels': labels}) if filepaths else pd.DataFrame(columns=['filepaths', 'labels'])
        if df.empty: print_in_color(f"Warning: No files for {split} set.", fore_tupple=(255,165,0))
        dataframes[split] = df

    train_df_orig = dataframes.get('train', pd.DataFrame(columns=['filepaths', 'labels']))
    valid_df_orig = dataframes.get('validation', pd.DataFrame(columns=['filepaths', 'labels']))
    test_df = dataframes.get('test', pd.DataFrame(columns=['filepaths', 'labels']))

    if use_undersampling:
        print_in_color("Applying undersampling...", fore_tupple=(255,255,0))
        train_dfs_sampled = [df_class.sample(n=min(train_target_count, len(df_class)), random_state=42) for lbl, df_class in train_df_orig.groupby('labels')]
        train_df = pd.concat(train_dfs_sampled).sample(frac=1, random_state=42).reset_index(drop=True) if train_dfs_sampled else pd.DataFrame(columns=['filepaths', 'labels'])
        valid_dfs_sampled = [df_class.sample(n=min(valid_target_count, len(df_class)), random_state=42) for lbl, df_class in valid_df_orig.groupby('labels')]
        valid_df = pd.concat(valid_dfs_sampled).sample(frac=1, random_state=42).reset_index(drop=True) if valid_dfs_sampled else pd.DataFrame(columns=['filepaths', 'labels'])
    else:
        print_in_color("Using full dataset (no undersampling).", fore_tupple=(0,255,0))
        train_df = train_df_orig.sample(frac=1, random_state=42).reset_index(drop=True)
        valid_df = valid_df_orig.sample(frac=1, random_state=42).reset_index(drop=True)

    classes_list = sorted(train_df['labels'].unique()) if not train_df.empty else classlist
    class_count_val = len(classes_list)
    sample_df = train_df.sample(n=min(50, len(train_df)), replace=False) if not train_df.empty else pd.DataFrame()
    ht, wt, count_img = 0, 0, 0
    if not sample_df.empty:
        for _, row in sample_df.iterrows():
            try:
                img = cv2.imread(row['filepaths']);
                if img is None: continue
                h, w = img.shape[:2]; ht += h; wt += w; count_img += 1
            except Exception: continue
    have, wave = (int(ht/count_img) if count_img > 0 else 0), (int(wt/count_img) if count_img > 0 else 0)

    print('\nDataset Summary:')
    for name, df_s in [('Train', train_df), ('Validation', valid_df), ('Test', test_df)]:
        print(f'{name} dataset: {len(df_s)} images')
        if not df_s.empty: [print(f'  {cls:10s}: {len(df_s[df_s["labels"] == cls])} images') for cls in classes_list]
    print(f'\nDetails: Classes={class_count_val}, Avg H={have}, Avg W={wave}')
    return train_df, test_df, valid_df, classes_list, class_count_val

# --- Ayarlar ---
BASE_DIR = './dataset_balanced_20004'
USE_UNDERSAMPLING = False
TRAIN_TARGET_COUNT_IF_UNDERSAMPLING = 1960
VALID_TARGET_COUNT_IF_UNDERSAMPLING = 420
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 20 # Önceki logda 35'e kadar gitmişti, bu değer ayarlanabilir.
FINE_TUNE_AT_PERCENT = 0.4
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINETUNE = 1e-5
DROPOUT_RATE = 0.45
L2_REGULARIZATION = 0.001
LOSS_FUNCTION_TYPE = 'focal_loss'
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

# --- Veri Yükleme ---
train_df, test_df, valid_df, classes, class_count = make_dataframes(
    BASE_DIR, USE_UNDERSAMPLING, TRAIN_TARGET_COUNT_IF_UNDERSAMPLING, VALID_TARGET_COUNT_IF_UNDERSAMPLING
)
if train_df.empty or valid_df.empty or class_count == 0:
    print_in_color("Critical error: DataFrames empty or no classes. Exiting.", fore_tupple=(255,0,0)); exit(1)

def check_labels(df, filename='label_check.png', num_samples_per_class=1):
    # (Bu fonksiyon aynı kalabilir, bir önceki versiyondaki gibi)
    if df.empty: print_in_color("DataFrame empty, cannot check labels.", fore_tupple=(255,165,0)); return
    unique_classes = df['labels'].unique()
    num_classes_to_show = len(unique_classes)
    if num_classes_to_show == 0: return
    plt.figure(figsize=(min(5 * num_samples_per_class, 20), 5 * ((num_classes_to_show -1) // num_samples_per_class + 1) ))
    plot_idx = 1
    for cls in unique_classes:
        class_samples = df[df['labels'] == cls].sample(n=min(num_samples_per_class, len(df[df['labels'] == cls])), random_state=42)
        for _, row in class_samples.iterrows():
            try:
                img = cv2.imread(row['filepaths'])
                if img is None: continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(((num_classes_to_show * num_samples_per_class -1) // 5 + 1), min(5, num_classes_to_show * num_samples_per_class), plot_idx)
                plt.imshow(img); plt.title(cls); plt.axis('off'); plot_idx += 1
            except Exception as e: print_in_color(f"Error reading/plotting {row['filepaths']}: {e}", fore_tupple=(255,165,0))
    plt.tight_layout(); plt.savefig(filename); print_in_color(f"Label check image saved to {filename}")

check_labels(train_df, filename='label_check_v3.png')

def make_gens(batch_size, train_df, test_df, valid_df, img_size_tuple):
    ycol = 'labels'
    trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=25, width_shift_range=0.15, height_shift_range=0.15,
                               zoom_range=0.15, shear_range=0.15, brightness_range=[0.8, 1.2], fill_mode='nearest',
                               preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    t_and_v_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input)
    
    print_in_color('Creating train generator...')
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col=ycol, target_size=img_size_tuple,
                                         class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    print_in_color('Creating valid generator...')
    valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col=ycol, target_size=img_size_tuple,
                                               class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    test_gen, test_steps_eff = None, 0
    if not test_df.empty:
        test_batch_size_eff = min(batch_size, len(test_df)) # Test batch size, örnek sayısını geçmemeli
        test_steps_eff = int(np.ceil(len(test_df) / test_batch_size_eff))
        print_in_color('Creating test generator...')
        test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col=ycol, target_size=img_size_tuple,
                                                class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size_eff)
        print(f'Test batch size: {test_batch_size_eff}, Test steps: {test_steps_eff}')
    else: print_in_color("Test DataFrame empty, test generator not created.", fore_tupple=(255,165,0))
    classes_from_gen = list(train_gen.class_indices.keys())
    print(f'Train classes from generator: {classes_from_gen}')
    return train_gen, test_gen, valid_gen, test_steps_eff, classes_from_gen # test_batch_size_eff kaldırıldı, test_gen.batch_size kullanılabilir

train_gen, test_gen, valid_gen, test_steps, classes_from_generator = make_gens(
    BATCH_SIZE, train_df, test_df, valid_df, IMG_SIZE
)
classes = classes_from_generator
class_count = len(classes)

def show_image_samples(gen, filename='sample_augmented_images.png'):
    # (Bu fonksiyon aynı kalabilir, bir önceki versiyondaki gibi)
    if gen is None: return
    class_names = list(gen.class_indices.keys())
    images, labels = next(gen) # Bir batch al
    images_to_show = (images + 1.0) * 127.5 # MobileNetV2 ön işlemesinin tersi
    images_to_show = np.clip(images_to_show, 0, 255).astype('uint8')
    plt.figure(figsize=(20, 20))
    for i in range(min(len(images_to_show), 25)): # En fazla 25 resim göster
        plt.subplot(5, 5, i + 1)
        plt.imshow(images_to_show[i]); plt.axis('off')
        plt.title(class_names[np.argmax(labels[i])], color='blue', fontsize=12)
    plt.tight_layout(); plt.savefig(filename)
    print_in_color(f"Sample augmented images saved to {filename}")

show_image_samples(train_gen, filename='sample_augmented_images_v3.png')

def make_model(img_size_tuple, num_classes, dr_rate, l2_val, initial_lr, loss_type, focal_gamma, focal_alpha):
    base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size_tuple[0], img_size_tuple[1], 3))
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_val))(x)
    x = Dropout(dr_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model_instance = Model(inputs=base.input, outputs=outputs)
    
    loss_name_for_print = loss_type
    if loss_type == 'focal_loss':
        selected_loss = categorical_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
        loss_name_for_print = f"Custom Categorical Focal Loss (g={focal_gamma},a={focal_alpha})"
    else:
        selected_loss = 'categorical_crossentropy'
    model_instance.compile(optimizer=Adam(learning_rate=initial_lr), loss=selected_loss,
                           metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])
    print_in_color(f"Model created. LR={initial_lr}, Loss: {loss_name_for_print}")
    return model_instance, base

model, base_model_obj = make_model(IMG_SIZE, class_count, DROPOUT_RATE, L2_REGULARIZATION, LEARNING_RATE_INITIAL,
                                   LOSS_FUNCTION_TYPE, FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA)

callbacks_initial_phase = [
    ModelCheckpoint('initial_best_weights_v3.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1e-7, verbose=1)
]

class_weights_map = None
if LOSS_FUNCTION_TYPE == 'categorical_crossentropy' and not USE_UNDERSAMPLING:
    # ... (Class weights hesaplama mantığı aynı kalabilir)
    print_in_color("Class weights logic for CCE would be here if CCE was selected.", fore_tupple=(255,165,0))
else:
    print_in_color("Class weights not used (Focal Loss active or undersampling used).", fore_tupple=(0,255,0))

print_in_color(f"--- Initial Training (Top Layers) for {INITIAL_EPOCHS} epochs ---", fore_tupple=(0,255,0))
history_initial_phase = model.fit(
    train_gen, epochs=INITIAL_EPOCHS, validation_data=valid_gen,
    callbacks=callbacks_initial_phase,
    class_weight=class_weights_map if LOSS_FUNCTION_TYPE == 'categorical_crossentropy' else None, verbose=1
)

if os.path.exists('initial_best_weights_v3.h5'):
    print_in_color("Loading best initial weights for fine-tuning...", fore_tupple=(255,255,0))
    model.load_weights('initial_best_weights_v3.h5')

print_in_color(f"--- Fine-Tuning for {FINE_TUNE_EPOCHS} epochs ---", fore_tupple=(0,255,0))
base_model_obj.trainable = True
fine_tune_from_layer = int(len(base_model_obj.layers) * (1 - FINE_TUNE_AT_PERCENT))
for layer in base_model_obj.layers[:fine_tune_from_layer]: layer.trainable = False
print_in_color(f"Unfreezing layers from index {fine_tune_from_layer}.")

loss_ft = categorical_focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA) if LOSS_FUNCTION_TYPE == 'focal_loss' else 'categorical_crossentropy'
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE_FINETUNE), loss=loss_ft,
              metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')])

callbacks_finetune_phase = [
    ModelCheckpoint('finetune_best_weights_v3.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-8, verbose=1)
]

start_epoch_ft = history_initial_phase.epoch[-1] + 1 if history_initial_phase and history_initial_phase.epoch else INITIAL_EPOCHS
total_fine_tune_epochs = start_epoch_ft + FINE_TUNE_EPOCHS

history_finetune_phase = model.fit(
    train_gen, epochs=total_fine_tune_epochs, initial_epoch=start_epoch_ft,
    validation_data=valid_gen, callbacks=callbacks_finetune_phase,
    class_weight=class_weights_map if LOSS_FUNCTION_TYPE == 'categorical_crossentropy' else None, verbose=1
)

def combine_histories(hist1, hist2):
    # (Bu fonksiyon aynı kalabilir, bir önceki versiyondaki gibi)
    comb_hist = {}
    hist1_dict = hist1.history if hist1 and hasattr(hist1, 'history') else {}
    hist2_dict = hist2.history if hist2 and hasattr(hist2, 'history') else {}
    if not hist1_dict and not hist2_dict: return {}
    if not hist1_dict: return hist2_dict
    if not hist2_dict: return hist1_dict
    for key_ in hist1_dict.keys():
        comb_hist[key_] = hist1_dict[key_] + hist2_dict.get(key_, [])
    for key_ in hist2_dict.keys():
        if key_ not in comb_hist: #Should not happen if metrics are same
             comb_hist[key_] = ([0] * len(hist1_dict.get(next(iter(hist1_dict)),[]))) + hist2_dict[key_]
    return comb_hist

final_history = combine_histories(history_initial_phase, history_finetune_phase)

def plot_training_history(hist_dict, filename='training_history_final.png'):
    if not hist_dict or not any(key in hist_dict for key in ['accuracy', 'loss']):
        print_in_color("History empty or missing keys, skipping plotting.", fore_tupple=(255,0,0)); return

    plt.style.use('seaborn-darkgrid') # Matplotlib'in kendi darkgrid stilini kullan
    plt.figure(figsize=(20, 8))
    metrics_info = [('Accuracy', 'accuracy', 'val_accuracy'),
                    ('Loss', 'loss', 'val_loss'),
                    ('Precision & Recall', ['precision', 'recall'], ['val_precision', 'val_recall'])]
    
    num_epochs = len(hist_dict.get(metrics_info[0][1], [])) # İlk train metriğinin uzunluğunu al
    epochs_range = range(num_epochs)

    for i, (title, train_keys, val_keys) in enumerate(metrics_info):
        plt.subplot(1, len(metrics_info), i + 1)
        if isinstance(train_keys, list): # Precision & Recall
            for tk, vk, ls in zip(train_keys, val_keys, ['-', '--']): # Train ve Val için aynı metrikler
                if tk in hist_dict: plt.plot(epochs_range, hist_dict[tk], label=f'Train {tk.capitalize()}', linestyle=ls)
                if vk in hist_dict: plt.plot(epochs_range, hist_dict[vk], label=f'Val {vk.split("_")[-1].capitalize()}', linestyle=ls) # val_precision -> Precision
        else: # Accuracy, Loss
            if train_keys in hist_dict: plt.plot(epochs_range, hist_dict[train_keys], label=f'Train {train_keys.replace("_"," ").capitalize()}')
            if val_keys in hist_dict: plt.plot(epochs_range, hist_dict[val_keys], label=f'Val {val_keys.replace("val_","").replace("_"," ").capitalize()}')
        plt.title(title); plt.xlabel('Epoch'); plt.ylabel(title.split(' ')[0]); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(filename)
    print_in_color(f"Training history plot saved to {filename}")

plot_training_history(final_history, filename='training_history_final_v3.png')

def evaluate_final_model(model_to_eval, test_gen_data, steps_count, class_names_list, cm_filename='cm.png'):
    if test_gen_data is None or steps_count == 0:
        print_in_color("Test generator NA. Skipping final evaluation.", fore_tupple=(255,165,0)); return None, None, None
    print_in_color("\n--- Evaluating Final Model on Test Set ---", fore_tupple=(0,255,0))
    test_gen_data.reset()
    y_pred_probs = model_to_eval.predict(test_gen_data, steps=steps_count, verbose=1)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen_data.classes[:len(y_pred_classes)] # Önemli: Sadece tahmin edilen kadarını al

    cm = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names_list, yticklabels=class_names_list)
    plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout(); plt.savefig(cm_filename)
    print_in_color(f"Confusion matrix saved to {cm_filename}")

    report_dict = classification_report(y_true, y_pred_classes, target_names=class_names_list, zero_division=0, output_dict=True)
    print('\nFinal Classification Report:\n', classification_report(y_true, y_pred_classes, target_names=class_names_list, zero_division=0))
    
    if 'nv' in class_names_list and 'mel' in class_names_list:
        nv_idx, mel_idx = class_names_list.index('nv'), class_names_list.index('mel')
        print_in_color("\n--- nv vs mel Confusion Details ---", fore_tupple=(255,255,0))
        # ... (nv vs mel analiz kısmı aynı kalabilir, report_dict kullanılarak)
        total_nv, total_mel = np.sum(cm[nv_idx, :]), np.sum(cm[mel_idx, :])
        if total_nv > 0: print(f"True 'nv': {total_nv}, Correct: {cm[nv_idx,nv_idx]} ({(cm[nv_idx,nv_idx]/total_nv)*100:.2f}%), To 'mel': {cm[nv_idx,mel_idx]} ({(cm[nv_idx,mel_idx]/total_nv)*100:.2f}%)")
        if total_mel > 0: print(f"True 'mel': {total_mel}, Correct: {cm[mel_idx,mel_idx]} ({(cm[mel_idx,mel_idx]/total_mel)*100:.2f}%), To 'nv': {cm[mel_idx,nv_idx]} ({(cm[mel_idx,nv_idx]/total_mel)*100:.2f}%)")
        if 'nv' in report_dict and 'mel' in report_dict:
             print(f"Metrics 'nv': P={report_dict['nv']['precision']:.2f}, R={report_dict['nv']['recall']:.2f}, F1={report_dict['nv']['f1-score']:.2f}")
             print(f"Metrics 'mel': P={report_dict['mel']['precision']:.2f}, R={report_dict['mel']['recall']:.2f}, F1={report_dict['mel']['f1-score']:.2f}")
    return y_true, y_pred_classes, y_pred_probs


best_ft_weights = 'finetune_best_weights_v3.h5'
best_init_weights = 'initial_best_weights_v3.h5'
loaded_best = False
if os.path.exists(best_ft_weights):
    print_in_color(f"Loading best fine-tuned weights: {best_ft_weights}", fore_tupple=(255,255,0))
    try: model.load_weights(best_ft_weights); loaded_best = True
    except Exception as e: print_in_color(f"Error loading fine-tune weights: {e}", fore_tupple=(255,0,0))
if not loaded_best and os.path.exists(best_init_weights):
    print_in_color(f"Loading best initial weights: {best_init_weights}", fore_tupple=(255,255,0))
    try: model.load_weights(best_init_weights); loaded_best = True
    except Exception as e: print_in_color(f"Error loading initial weights: {e}", fore_tupple=(255,0,0))
if not loaded_best: print_in_color("No best weights loaded. Evaluating with current model.", fore_tupple=(255,165,0))

y_true_final, y_pred_final, y_prob_final = evaluate_final_model(model, test_gen, test_steps, classes, cm_filename='confusion_matrix_final_v3.png')

def show_misclassified_examples(model_mc, test_gen_mc, steps_mc, class_names_mc, filename_plot='misclassified_v3.png'):
    # (Bu fonksiyon aynı kalabilir, bir önceki versiyondaki gibi)
    if test_gen_mc is None or steps_mc == 0: print_in_color("Test gen NA for misclassified.", fore_tupple=(255,165,0)); return
    test_gen_mc.reset()
    y_pred_p_mc = model_mc.predict(test_gen_mc, steps=steps_mc, verbose=1)
    y_pred_c_mc = np.argmax(y_pred_p_mc, axis=1)
    y_true_mc = test_gen_mc.classes[:len(y_pred_c_mc)]
    filepaths_mc = test_gen_mc.filepaths[:len(y_pred_c_mc)]
    misclassified = [{'fp': fp, 'true': class_names_mc[yt], 'pred': class_names_mc[yp]} 
                     for i, (fp, yt, yp) in enumerate(zip(filepaths_mc, y_true_mc, y_pred_c_mc)) if yt != yp]
    print(f'\nTotal misclassified in test: {len(misclassified)}')
    if misclassified:
        plt.figure(figsize=(20, max(10, (len(misclassified[:10])-1)//5 * 4 + 4) )) # Dinamik yükseklik
        for i, item in enumerate(misclassified[:min(10, len(misclassified))]):
            try:
                img = cv2.imread(item['fp'])
                if img is None: print(f"Warn: Img not read {item['fp']}"); continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.subplot(max(1, (len(misclassified[:10])-1)//5 + 1), min(5, len(misclassified[:10])), i + 1) # Dinamik satır/sütun
                plt.imshow(img); plt.axis('off'); plt.title(f"T:{item['true']}\nP:{item['pred']}", fontsize=10)
            except Exception as e: print_in_color(f"Error plotting misclassified {item['fp']}: {e}", fore_tupple=(255,165,0))
        plt.tight_layout(); plt.savefig(filename_plot)
        print_in_color(f"Misclassified examples saved to {filename_plot}")

if test_gen: # Sadece test_gen varsa çalıştır
    show_misclassified_examples(model, test_gen, test_steps, classes, filename_plot='misclassified_examples_final_v3.png')

print_in_color("\n--- Saving Final Model ---", fore_tupple=(0,255,0))
final_weights_path = 'final_dermatology_model_BEST_WEIGHTS_v3.h5'
final_model_path = 'final_dermatology_model_FULL_v3.keras'
try:
    model.save_weights(final_weights_path)
    print_in_color(f"Best weights saved: {final_weights_path}")
    model.save(final_model_path)
    print_in_color(f"Full model saved: {final_model_path}")
except Exception as e: print_in_color(f"Error saving model: {e}", fore_tupple=(255,0,0))

print_in_color("--- Script Execution Finished ---", fore_tupple=(0,255,0))
