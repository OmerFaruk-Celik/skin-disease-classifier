#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from tqdm import tqdm
import warnings
from matplotlib.backends.backend_pdf import PdfPages
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, Callback

from sklearn.metrics import f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, precision_recall_curve

try:
    import scikitplot as skplt
    SCIKITPLOT_AVAILABLE = True
except ImportError:
    SCIKITPLOT_AVAILABLE = False
    print("scikit-plot not found. Lift and cumulative gains curves will be skipped.")

warnings.simplefilter("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

try:
    from tensorflow_addons.losses import SigmoidFocalCrossEntropy
    TFA_AVAILABLE = True
    print("TensorFlow Addons found (SigmoidFocalCrossEntropy check successful).")
except ImportError:
    TFA_AVAILABLE = False
    print("TensorFlow Addons not found. Custom Categorical Focal Loss will be used.")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) found: {[gpu.name for gpu in gpus]}")
    print("TensorFlow will use GPU.")
else:
    print("No GPU found! TensorFlow will use CPU.")

def categorical_focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_loss_val = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(focal_loss_val)
    return focal_loss

print('All modules have been imported')
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"TensorFlow version: {tf.__version__}")

def print_in_color(txt_msg, fore_tupple=(0,255,255), back_tupple=(100,100,100)):
    rf, gf, bf = fore_tupple
    rb, gb, bb = back_tupple
    msg = '{0}' + txt_msg
    mat = '\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' + str(gb) + ';' + str(bb) + 'm'
    print(msg.format(mat), flush=True)
    print('\33[0m', flush=True)

msg = 'Test of default colors for console output.'
print_in_color(msg)

# --- CALLBACK: Per-Class metrics and logging ---
class PerClassMetrics(Callback):
    def __init__(self, train_generator, validation_generator, classes, eval_frequency=1, sample_size=1000):
        super(PerClassMetrics, self).__init__()
        self.train_gen = train_generator
        self.valid_gen = validation_generator
        self.classes = classes
        self.num_classes = len(classes)
        self.class_indices = train_generator.class_indices
        self.eval_frequency = eval_frequency
        self.sample_size = sample_size

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.eval_frequency != 0:
            return
        logs = logs or {}
        self.train_gen.reset()
        self.valid_gen.reset()
        
        train_y_true, train_y_pred = [], []
        valid_y_true, valid_y_pred = [], []
        
        train_steps = min(len(self.train_gen), max(1, self.sample_size // self.train_gen.batch_size))
        for _ in range(train_steps):
            X, y = next(self.train_gen)
            y_pred = self.model.predict(X, verbose=0)
            train_y_true.extend(np.argmax(y, axis=1))
            train_y_pred.extend(np.argmax(y_pred, axis=1))
        
        valid_steps = min(len(self.valid_gen), max(1, self.sample_size // self.valid_gen.batch_size))
        for _ in range(valid_steps):
            X, y = next(self.valid_gen)
            y_pred = self.model.predict(X, verbose=0)
            valid_y_true.extend(np.argmax(y, axis=1))
            valid_y_pred.extend(np.argmax(y_pred, axis=1))
        
        train_y_true = np.array(train_y_true)
        train_y_pred = np.array(train_y_pred)
        valid_y_true = np.array(valid_y_true)
        valid_y_pred = np.array(valid_y_pred)
        
        train_class_acc = []
        valid_class_acc = []
        train_class_f1 = []
        valid_class_f1 = []
        
        for i, cls in enumerate(self.classes):
            train_mask = train_y_true == i
            if train_mask.sum() > 0:
                train_class_acc.append(np.mean(train_y_pred[train_mask] == i))
            else:
                train_class_acc.append(0.0)
            
            valid_mask = valid_y_true == i
            if valid_mask.sum() > 0:
                valid_class_acc.append(np.mean(valid_y_pred[valid_mask] == i))
            else:
                valid_class_acc.append(0.0)
            
            train_f1 = f1_score(train_y_true, train_y_pred, labels=[i], average='micro', zero_division=0)
            valid_f1 = f1_score(valid_y_true, valid_y_pred, labels=[i], average='micro', zero_division=0)
            train_class_f1.append(train_f1)
            valid_class_f1.append(valid_f1)
        
        for i, cls in enumerate(self.classes):
            logs[f'train_{cls}_accuracy'] = train_class_acc[i]
            logs[f'valid_{cls}_accuracy'] = valid_class_acc[i]
            logs[f'train_{cls}_f1'] = train_class_f1[i]
            logs[f'valid_{cls}_f1'] = valid_class_f1[i]

# --- CALLBACK: Per-epoch plotting and extra analyses ---
class PerEpochPlotter(tf.keras.callbacks.Callback):
    def __init__(self, classes, valid_gen, out_dir="epoch_plots", pr_roc_frequency=5):
        super().__init__()
        self.classes = classes
        self.valid_gen = valid_gen
        self.out_dir = out_dir
        self.pr_roc_frequency = pr_roc_frequency
        os.makedirs(out_dir, exist_ok=True)
        self.history = {}

    def on_train_begin(self, logs=None):
        self.history = {}

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Learning rate kaydı
        if hasattr(self.model.optimizer, 'lr'):
            try:
                lr_val = float(tf.keras.backend.get_value(self.model.optimizer.lr))
            except Exception:
                lr_val = self.model.optimizer.lr if isinstance(self.model.optimizer.lr, float) else 0.0
            logs['lr'] = lr_val

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        self.plot_epoch(epoch)

    def plot_epoch(self, epoch):
        pdf_path = f"{self.out_dir}/epoch_{epoch+1:03d}_plots.pdf"
        with PdfPages(pdf_path) as pdf:
            history = self.history
            epochs_range = range(1, len(history.get('accuracy', [])) + 1)
            # Aggregate
            fig, axes = plt.subplots(2, 3, figsize=(20,12))
            # Accuracy
            if 'accuracy' in history and 'val_accuracy' in history:
                axes[0,0].plot(epochs_range, history['accuracy'], label='Train Accuracy')
                axes[0,0].plot(epochs_range, history['val_accuracy'], label='Val Accuracy')
                axes[0,0].set_title('Aggregate Accuracy')
                axes[0,0].legend()
            # Loss
            if 'loss' in history and 'val_loss' in history:
                axes[0,1].plot(epochs_range, history['loss'], label='Train Loss')
                axes[0,1].plot(epochs_range, history['val_loss'], label='Val Loss')
                axes[0,1].set_title('Aggregate Loss')
                axes[0,1].legend()
            # Precision & Recall
            for metric, val_metric, label in [('precision', 'val_precision', 'Precision'), ('recall', 'val_recall', 'Recall')]:
                if metric in history and val_metric in history:
                    axes[0,2].plot(epochs_range, history[metric], label=f'Train {label}')
                    axes[0,2].plot(epochs_range, history[val_metric], label=f'Val {label}')
            axes[0,2].set_title('Aggregate Precision & Recall')
            axes[0,2].legend()
            # Learning rate
            if 'lr' in history:
                axes[1,0].plot(epochs_range, history['lr'], label='Learning Rate', color='g')
                axes[1,0].set_title('Learning Rate')
                axes[1,0].legend()
            # Top-3 accuracy
            if 'top3_acc' in history and 'val_top3_acc' in history:
                axes[1,1].plot(epochs_range, history['top3_acc'], label='Train Top-3 Acc')
                axes[1,1].plot(epochs_range, history['val_top3_acc'], label='Val Top-3 Acc')
                axes[1,1].set_title('Top-3 Accuracy')
                axes[1,1].legend()
            # Overfitting/Underfitting Analizi
            if 'accuracy' in history and 'val_accuracy' in history:
                acc_gap = np.array(history['accuracy']) - np.array(history['val_accuracy'])
                loss_gap = np.array(history['val_loss']) - np.array(history['loss'])
                axes[1,2].plot(epochs_range, acc_gap, label='Acc Gap (Train-Val)')
                axes[1,2].plot(epochs_range, loss_gap, label='Loss Gap (Val-Train)')
                axes[1,2].axhline(0, color='grey', linestyle='--')
                axes[1,2].set_title('Overfitting/Underfitting Analysis')
                axes[1,2].legend()
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
            # Per-class
            for cls in self.classes:
                fig, axes = plt.subplots(1, 3, figsize=(18,6))
                if f'train_{cls}_accuracy' in history and f'valid_{cls}_accuracy' in history:
                    axes[0].plot(epochs_range, history[f'train_{cls}_accuracy'], label='Train Accuracy')
                    axes[0].plot(epochs_range, history[f'valid_{cls}_accuracy'], label='Val Accuracy')
                    axes[0].set_title(f'{cls} Accuracy')
                    axes[0].legend()
                if f'train_{cls}_f1' in history and f'valid_{cls}_f1' in history:
                    axes[1].plot(epochs_range, history[f'train_{cls}_f1'], label='Train F1')
                    axes[1].plot(epochs_range, history[f'valid_{cls}_f1'], label='Val F1')
                    axes[1].set_title(f'{cls} F1 Score')
                    axes[1].legend()
                axes[2].text(0.5, 0.5, 'Per-Class Loss Not Computed', ha='center', va='center')
                axes[2].set_title(f'{cls} Loss')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
            # Her 5 epoch'ta bir ROC, PR, Confusion Matrix, Class Distribution, Precision/Recall/F1 Heatmap
            if (epoch + 1) % self.pr_roc_frequency == 0:
                print(f"Epoch {epoch+1}: Computing validation ROC, PR, confusion matrix, class distribution, metrics heatmap.")
                y_true = []
                y_pred_prob = []
                self.valid_gen.reset()
                for i in range(len(self.valid_gen)):
                    X, y = next(self.valid_gen)
                    y_pred = self.model.predict(X, verbose=0)
                    y_true.append(y)
                    y_pred_prob.append(y_pred)
                y_true = np.concatenate(y_true)
                y_pred_prob = np.concatenate(y_pred_prob)
                y_pred_cls = np.argmax(y_pred_prob, axis=1)
                y_true_cls = np.argmax(y_true, axis=1)
                # Confusion matrix
                fig, ax = plt.subplots(figsize=(8,6))
                cm = confusion_matrix(y_true_cls, y_pred_cls)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.classes, yticklabels=self.classes, ax=ax)
                ax.set_title(f'Confusion Matrix (Epoch {epoch+1})')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('True')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # ROC curves
                fig, ax = plt.subplots(figsize=(8,6))
                for i, cls in enumerate(self.classes):
                    fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
                    try:
                        roc_auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
                    except Exception:
                        roc_auc = 0.0
                    ax.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.2f})')
                ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
                ax.set_title(f'ROC Curves (Epoch {epoch+1})')
                ax.set_xlabel('False Positive Rate')
                ax.set_ylabel('True Positive Rate')
                ax.legend()
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # Precision-Recall curves
                fig, ax = plt.subplots(figsize=(8,6))
                for i, cls in enumerate(self.classes):
                    prec, rec, _ = precision_recall_curve(y_true[:, i], y_pred_prob[:, i])
                    ax.plot(rec, prec, label=cls)
                ax.set_title(f'Precision-Recall Curves (Epoch {epoch+1})')
                ax.set_xlabel('Recall')
                ax.set_ylabel('Precision')
                ax.legend()
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # Class distribution
                fig, ax = plt.subplots(figsize=(8,4))
                sns.histplot(y_true_cls, bins=len(self.classes), discrete=True, ax=ax)
                ax.set_xticks(range(len(self.classes)))
                ax.set_xticklabels(self.classes)
                ax.set_title(f'Validation Class Distribution (Epoch {epoch+1})')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
                # Precision/Recall/F1 heatmap
                report = classification_report(y_true_cls, y_pred_cls, target_names=self.classes, output_dict=True, zero_division=0)
                metric_mat = np.zeros((len(self.classes), 3))
                for i, cls in enumerate(self.classes):
                    metric_mat[i, 0] = report[cls]['precision']
                    metric_mat[i, 1] = report[cls]['recall']
                    metric_mat[i, 2] = report[cls]['f1-score']
                fig, ax = plt.subplots(figsize=(8,6))
                sns.heatmap(metric_mat, annot=True, fmt=".2f", cmap="YlGnBu", 
                            xticklabels=['Precision', 'Recall', 'F1'], yticklabels=self.classes, ax=ax)
                ax.set_title(f'Per-Class PRF1 Heatmap (Epoch {epoch+1})')
                plt.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        print(f"Epoch {epoch+1} plots saved to {pdf_path}")

# --- DATASET, MODEL, GENERATOR KISMI ---

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
    return train_df, test_df, valid_df, classes_list, class_count_val
    
def make_gens(batch_size, train_df, test_df, valid_df, img_size_tuple):
    ycol = 'labels'
    trgen = ImageDataGenerator(horizontal_flip=True, rotation_range=25, width_shift_range=0.15, height_shift_range=0.15,
                               zoom_range=0.15, shear_range=0.15, brightness_range=[0.8, 1.2], fill_mode='nearest',
                               preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input)
    t_and_v_gen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input)
    
    print_in_color('Creating train generator...')
    train_gen = trgen.flow_from_dataframe(train_df, x_col='filepaths', y_col=ycol, target_size=img_size_tuple,
                                         class_mode='categorical', color_mode='rgb', shuffle=True, batch_size=batch_size)
    print_in_color('Creating valid generator...')
    valid_gen = t_and_v_gen.flow_from_dataframe(valid_df, x_col='filepaths', y_col=ycol, target_size=img_size_tuple,
                                               class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=batch_size)
    test_gen, test_steps_eff = None, 0
    if not test_df.empty:
        test_batch_size_eff = min(batch_size, len(test_df))
        test_steps_eff = int(np.ceil(len(test_df) / test_batch_size_eff))
        print_in_color('Creating test generator...')
        test_gen = t_and_v_gen.flow_from_dataframe(test_df, x_col='filepaths', y_col=ycol, target_size=img_size_tuple,
                                                class_mode='categorical', color_mode='rgb', shuffle=False, batch_size=test_batch_size_eff)
        print(f'Test batch size: {test_batch_size_eff}, Test steps: {test_steps_eff}')
    else:
        print_in_color("Test DataFrame empty, test generator not created.", fore_tupple=(255,165,0))
    
    # Get classes from generator
    classes_from_gen = list(train_gen.class_indices.keys())
    print(f'Train classes from generator: {classes_from_gen}')
    
    # Save class indices to labels.txt
    labels_file_path = 'labels.txt'
    try:
        with open(labels_file_path, 'w', encoding='utf-8') as f:
            for class_name, index in sorted(train_gen.class_indices.items(), key=lambda x: x[1]):
                f.write(f"{index} {class_name}\n")
        print_in_color(f"Saved class indices to {labels_file_path}", fore_tupple=(0,255,0))
    except Exception as e:
        print_in_color(f"Error saving labels.txt: {e}", fore_tupple=(255,0,0))
        raise
    
    return train_gen, test_gen, valid_gen, test_steps_eff, classes_from_gen

def make_model(img_size_tuple, num_classes, dr_rate, l2_val, initial_lr, loss_type, focal_gamma, focal_alpha):
    base = MobileNetV3Large(
        input_shape=(img_size_tuple[0], img_size_tuple[1], 3),
        include_top=False,
        weights='imagenet'
    )
    print_in_color("MobileNetV3Large loaded successfully.")
    base.trainable = False
    x = GlobalAveragePooling2D()(base.output)
    x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(l2_val))(x)
    x = Dropout(dr_rate)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    model_instance = Model(inputs=base.input, outputs=outputs)
    loss_name_for_print = loss_type
    metrics_list = [
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
    if loss_type == 'focal_loss':
        selected_loss = categorical_focal_loss(gamma=focal_gamma, alpha=focal_alpha)
        loss_name_for_print = f"Custom Categorical Focal Loss (g={focal_gamma},a={focal_alpha})"
    else:
        selected_loss = 'categorical_crossentropy'
    model_instance.compile(optimizer=Adam(learning_rate=initial_lr), loss=selected_loss, metrics=metrics_list)
    print_in_color(f"Model created. LR={initial_lr}, Loss: {loss_name_for_print}")
    return model_instance, base

# === HAZIRLIK ===
BASE_DIR = './data2'
USE_UNDERSAMPLING = False
TRAIN_TARGET_COUNT_IF_UNDERSAMPLING = 2000
VALID_TARGET_COUNT_IF_UNDERSAMPLING = 500
IMG_SIZE = (224, 224)
BATCH_SIZE = 64
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 1
FINE_TUNE_AT_PERCENT = 0.4
LEARNING_RATE_INITIAL = 1e-4
LEARNING_RATE_FINETUNE = 1e-5
DROPOUT_RATE = 0.45
L2_REGULARIZATION = 0.001
LOSS_FUNCTION_TYPE = 'focal_loss'
FOCAL_LOSS_GAMMA = 2.0
FOCAL_LOSS_ALPHA = 0.25

train_df, test_df, valid_df, classes, class_count = make_dataframes(
    BASE_DIR, USE_UNDERSAMPLING, TRAIN_TARGET_COUNT_IF_UNDERSAMPLING, VALID_TARGET_COUNT_IF_UNDERSAMPLING
)
if train_df.empty or valid_df.empty or class_count == 0:
    print_in_color("Critical error: DataFrames empty or no classes. Exiting.", fore_tupple=(255,0,0)); exit(1)

train_gen, test_gen, valid_gen, test_steps, classes_from_generator = make_gens(
    BATCH_SIZE, train_df, test_df, valid_df, IMG_SIZE
)
classes = classes_from_generator
class_count = len(classes)

model, base_model_obj = make_model(IMG_SIZE, class_count, DROPOUT_RATE, L2_REGULARIZATION, LEARNING_RATE_INITIAL,
                                   LOSS_FUNCTION_TYPE, FOCAL_LOSS_GAMMA, FOCAL_LOSS_ALPHA)

callbacks_initial_phase = [
    ModelCheckpoint('initial_best_weights_mnv3.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=6, min_lr=1e-7, verbose=1),
    PerClassMetrics(train_gen, valid_gen, classes, eval_frequency=1, sample_size=1000),
    PerEpochPlotter(classes, valid_gen, out_dir="epoch_plots_initial", pr_roc_frequency=5)
]

print_in_color(f"--- Initial Training (Top Layers) for {INITIAL_EPOCHS} epochs ---", fore_tupple=(0,255,0))
try:
    history_initial_phase = model.fit(
        train_gen,
        epochs=INITIAL_EPOCHS,
        validation_data=valid_gen,
        callbacks=callbacks_initial_phase,
        verbose=1
    )
except Exception as e:
    print_in_color(f"Initial training failed: {e}", fore_tupple=(255,0,0))
    # Model ve verileri kurtar!
    model.save_weights("emergency_initial_weights.h5")
    model.save("emergency_initial_model.keras")
    raise

if os.path.exists('initial_best_weights_mnv3.weights.h5'):
    try:
        model.load_weights('initial_best_weights_mnv3.weights.h5')
        print_in_color("Loaded best initial weights for fine-tuning.", fore_tupple=(255,255,0))
    except Exception as e:
        print_in_color(f"Could not load best initial weights: {e}", fore_tupple=(255,0,0))

base_model_obj.trainable = True
fine_tune_from_layer = int(len(base_model_obj.layers) * (1 - FINE_TUNE_AT_PERCENT))
for layer in base_model_obj.layers[:fine_tune_from_layer]: layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_FINETUNE),
    loss=categorical_focal_loss(gamma=FOCAL_LOSS_GAMMA, alpha=FOCAL_LOSS_ALPHA),
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
    ]
)

callbacks_finetune_phase = [
    ModelCheckpoint('finetune_best_weights_mnv3.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True, mode='min', verbose=1),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=1e-8, verbose=1),
    PerClassMetrics(train_gen, valid_gen, classes, eval_frequency=1, sample_size=1000),
    PerEpochPlotter(classes, valid_gen, out_dir="epoch_plots_finetune", pr_roc_frequency=5)
]

start_epoch_ft = history_initial_phase.epoch[-1] + 1 if history_initial_phase and hasattr(history_initial_phase, 'epoch') else INITIAL_EPOCHS
total_fine_tune_epochs = start_epoch_ft + FINE_TUNE_EPOCHS

print_in_color(f"--- Fine-Tuning for {FINE_TUNE_EPOCHS} epochs ---", fore_tupple=(0,255,0))
try:
    history_finetune_phase = model.fit(
        train_gen, 
        epochs=total_fine_tune_epochs, 
        initial_epoch=start_epoch_ft,
        validation_data=valid_gen, 
        callbacks=callbacks_finetune_phase,
        verbose=1
    )
except Exception as e:
    print_in_color(f"Fine-tuning failed: {e}", fore_tupple=(255,0,0))
    model.save_weights("emergency_finetune_weights.h5")
    model.save("emergency_finetune_model.keras")
    raise

# === EĞİTİM SONRASI GÜVENLİ KAYIT ===
try:
    model.save_weights('final_mnv3_weights.h5')
    model.save('final_mnv3_model.keras')
    print_in_color("Final model and weights saved securely.", fore_tupple=(0,255,0))
except Exception as e:
    print_in_color(f"Error saving final model: {e}", fore_tupple=(255,0,0))

print_in_color("--- Script Execution Completed ---", fore_tupple=(0,255,0))
