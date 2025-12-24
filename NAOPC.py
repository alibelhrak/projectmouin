import os
import numpy as np
import cv2
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from tensorflow.keras import layers, optimizers, callbacks, Model
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras import mixed_precision


mixed_precision.set_global_policy('mixed_float16')
#all the details to help training the model 
IMG_SIZE = 224
NUM_CLASSES = 2
MAX_PER_CLASS = 10700
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 2e-4
KFOLDS = 5
DATA_PATH = "/mnt/projects/sutravek_project/Ali_belhrak/COVID-19_Radiography_Dataset"
RESULTS_DIR = "./results_effnetb7"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =============================================================
# 📁 Load Dataset
# =============================================================

file_paths, labels = [], []
#sorting image one by one
for cls_name in sorted(os.listdir(DATA_PATH)):
    cls_dir = os.path.join(DATA_PATH, cls_name)
    imgs = os.listdir(cls_dir)
    #when it sort the images it shuffle it , this is helping for refreshing the model 
    random.shuffle(imgs)
    for fname in imgs[:MAX_PER_CLASS]:
        #for each name or let's say a class in each image 
        file_paths.append(os.path.join(cls_dir, fname))
        labels.append(cls_name)

#encoding the classes 
le = LabelEncoder()
y_all = le.fit_transform(labels)
#making the images into arrays
paths = np.array(file_paths)
#making the classes into arrays
y_all = np.array(y_all)
#resize the images
images = np.array([cv2.resize(cv2.imread(p), (IMG_SIZE, IMG_SIZE)) for p in paths], dtype='float32') / 255.0


# =============================================================
# 🧪 Hold-Out Test Set
# =============================================================
#splitting the data
X_temp, X_test, y_temp, y_test = train_test_split(images, y_all, test_size=0.10, stratify=y_all, random_state=42)
X, y = X_temp, y_temp


#Loading The Model 
from tensorflow.keras.models import load_model

MODEL_PATH = "/mnt/projects/sutravek_project/Ali_belhrak/Next_Step/final_model.keras"
final_model = load_model(MODEL_PATH)

import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras

# Display
from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt


model_builder = final_model
img_size = (224, 224)
preprocess_input = keras.applications.xception.preprocess_input
decode_predictions = keras.applications.xception.decode_predictions

last_conv_layer_name = "top_conv"

img_path = "/mnt/projects/sutravek_project/COVID-19_Radiography_Dataset/COVID/COVID-4.png"

display(Image(img_path))

#grad cam 
def get_img_array(img_path, size):
    img = keras.utils.load_img(img_path, target_size=size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array
def make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape1:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape3:
                conv_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]

            first_grad = tape3.gradient(class_channel, conv_output)
        second_grad = tape2.gradient(first_grad, conv_output)
    third_grad = tape1.gradient(second_grad, conv_output)

    global_sum = tf.reduce_sum(conv_output, axis=(1, 2), keepdims=True)
    alpha_num = second_grad
    alpha_denom = 2 * second_grad + third_grad * global_sum
    alpha_denom = tf.where(alpha_denom != 0.0, alpha_denom, tf.ones_like(alpha_denom))
    alphas = alpha_num / alpha_denom
    alphas = tf.nn.relu(alphas)

    # Compute weights and ensure correct shape
    weights = tf.reduce_sum(alphas * tf.nn.relu(first_grad), axis=(1, 2))  # shape (1, C)
    weights = tf.reshape(weights, (1, 1, 1, -1))  # reshape to broadcast with conv_output

    # Multiply and reduce along channels
    heatmap = tf.reduce_sum(weights * conv_output, axis=-1)

    heatmap = tf.nn.relu(heatmap)
    heatmap = heatmap / tf.reduce_max(heatmap)
    return heatmap[0].numpy()

img_size = (224, 224)  # match your model input shape
img_array = preprocess_input(get_img_array(img_path, size=img_size))

model = final_model
model.layers[-1].activation = None

preds = model.predict(img_array)

# Replace decode_predictions with your class names
class_names = ["Normal", "COVID"]  # or whatever your two classes are
pred_class = np.argmax(preds)
print(f"Predicted class: {class_names[pred_class]} (confidence: {preds[0][pred_class]:.2f})")

# Grad-CAM
heatmap = make_gradcam_plus_plus_heatmap(img_array, model, last_conv_layer_name)
plt.matshow(heatmap)
plt.show()
plt.savefig(f"{RESULTS_DIR}/heatmap{__import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")



from datetime import datetime
from IPython.display import display, Markdown, Image
import matplotlib as mpl
import matplotlib.image as mpimg
def save_and_display_gradcam(img_path, heatmap, alpha=0.4, RESULTS_DIR="results_effnetb7"):
    # Make sure RESULTS_DIR exists
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap
    heatmap = np.uint8(255 * heatmap)

    # Apply jet colormap
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Convert to image and resize
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose heatmap
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Create a unique filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cam_path = os.path.join(RESULTS_DIR, f"GradCam_{timestamp}.png")

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Get model prediction
    preds = final_model.predict(np.expand_dims(img, axis=0))
    predicted_index = np.argmax(preds[0])
    predicted_label = le.inverse_transform([predicted_index])[0]

    # Display prediction and image
    
    
    
    # Load both images
    img1 = mpimg.imread(img_path)   # Original image
    img2 = mpimg.imread(cam_path)   # Grad-CAM image
    
    # Display side by side using subplots
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.axis('off')
    plt.title("Original")
    
    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title(f"Grad-CAM — {predicted_label}")
    
    plt.tight_layout()  # organize layout before saving
    save_path = f"{RESULTS_DIR}/GradCam_Output_{__import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"
    plt.savefig(save_path, bbox_inches='tight')  # <- ensure nothing is cut off
    print(f"Grad-CAM figure saved to: {save_path}")
    plt.show()


# Example usage
save_and_display_gradcam(img_path, heatmap)


# =============================================================
# NAOPC
# =============================================================
def compute_naopc(model, img, heatmap, patch_size=8, steps=None, perturbation_type="mean"):
    img = img.copy().astype(np.float32)
    
    if heatmap.ndim == 3:
        heatmap = heatmap[..., 0]
    
    H, W, C = img.shape
    
    n_patches_h = H // patch_size
    n_patches_w = W // patch_size
    total_patches = n_patches_h * n_patches_w
    
    if steps is None:
        steps = total_patches
    else:
        steps = min(steps, total_patches)
    
    patch_importance = []
    for i in range(n_patches_h):
        for j in range(n_patches_w):
            y_start, y_end = i * patch_size, (i + 1) * patch_size
            x_start, x_end = j * patch_size, (j + 1) * patch_size
            importance = np.mean(heatmap[y_start:y_end, x_start:x_end])
            patch_importance.append((importance, i, j))
    
    patch_importance.sort(key=lambda x: -x[0])
    
    img_input = np.expand_dims(img, axis=0)
    orig_pred = model.predict(img_input, verbose=0)
    orig_class = np.argmax(orig_pred[0])
    orig_confidence = float(orig_pred[0][orig_class])
    
    if perturbation_type == "mean":
        baseline_value = np.mean(img)
    elif perturbation_type == "zero":
        baseline_value = 0.0
    elif perturbation_type == "blur":
        blurred_img = cv2.GaussianBlur(img, (51, 51), 0)
    elif perturbation_type == "random":
        pass
    
    modified = img.copy()
    normalized_drops = []
    
    patches_per_step = max(1, total_patches // steps)
    
    for step in range(steps):
        start_idx = step * patches_per_step
        end_idx = min((step + 1) * patches_per_step, total_patches)
        
        for idx in range(start_idx, end_idx):
            if idx >= len(patch_importance):
                break
            _, i, j = patch_importance[idx]
            y_start, y_end = i * patch_size, (i + 1) * patch_size
            x_start, x_end = j * patch_size, (j + 1) * patch_size
            
            if perturbation_type == "blur":
                modified[y_start:y_end, x_start:x_end] = blurred_img[y_start:y_end, x_start:x_end]
            elif perturbation_type == "random":
                modified[y_start:y_end, x_start:x_end] = np.random.rand(patch_size, patch_size, C).astype(np.float32)
            else:
                modified[y_start:y_end, x_start:x_end] = baseline_value
        
        modified_input = np.expand_dims(modified, axis=0)
        new_pred = model.predict(modified_input, verbose=0)
        new_confidence = float(new_pred[0][orig_class])
        
        drop = orig_confidence - new_confidence
        normalized_drop = drop / (orig_confidence + 1e-8)
        normalized_drops.append(normalized_drop)
    
    naopc_score = np.mean(normalized_drops)
    
    return naopc_score, normalized_drops, orig_confidence


def compute_naopc_batch(model, images, heatmaps, patch_size=8, steps=200, perturbation_type="mean"):
    all_naopc = []
    
    for i, (img, hmap) in enumerate(zip(images, heatmaps)):
        naopc, _, _ = compute_naopc(model, img, hmap, patch_size, steps, perturbation_type)
        all_naopc.append(naopc)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(images)} images...")
    
    mean_naopc = np.mean(all_naopc)
    return mean_naopc, all_naopc


def plot_naopc_curve(normalized_drops, orig_confidence, save_path=None):
    steps = len(normalized_drops)
    x_vals = np.linspace(0, 100, steps)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, normalized_drops, 'g-o', linewidth=2, markersize=3)
    plt.fill_between(x_vals, 0, normalized_drops, alpha=0.3, color='green')
    
    naopc = np.mean(normalized_drops)
    plt.axhline(y=naopc, color='r', linestyle='--', label=f'NAOPC = {naopc:.4f}')
    
    plt.title(f"NAOPC Curve (Original Confidence: {orig_confidence:.4f})", fontsize=14, fontweight='bold')
    plt.xlabel("Percentage of Important Patches Removed (%)", fontsize=12)
    plt.ylabel("Normalized Confidence Drop", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved to: {save_path}")
    
    plt.show()



orig_img = cv2.imread(img_path)
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
orig_img = cv2.resize(orig_img, (224, 224))
orig_img = orig_img.astype(np.float32) / 255.0

img_array = preprocess_input(get_img_array(img_path, size=(224, 224)))
heatmap = make_gradcam_plus_plus_heatmap(img_array, final_model, last_conv_layer_name)

heatmap_resized = cv2.resize(heatmap.astype(np.float64), (224, 224))
heatmap_resized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)

print("\n" + "="*60)
print("NAOPC (Normalized Area Over Perturbation Curve) Analysis")
print("="*60)

perturbation_types = ["mean", "zero", "blur"]
naopc_results = {}

for p_type in perturbation_types:
    print(f"\nComputing NAOPC with '{p_type}' perturbation...")
    naopc_score, norm_drops, orig_conf = compute_naopc(
        final_model, 
        orig_img, 
        heatmap_resized, 
        patch_size=8,
        steps=200,
        perturbation_type=p_type
    )
    naopc_results[p_type] = {
        "naopc": naopc_score,
        "norm_drops": norm_drops,
        "orig_conf": orig_conf
    }
    print(f"  NAOPC ({p_type}): {naopc_score:.4f}")

plt.figure(figsize=(12, 6))
colors = ['blue', 'red', 'green']
for (p_type, result), color in zip(naopc_results.items(), colors):
    x_vals = np.linspace(0, 100, len(result["norm_drops"]))
    plt.plot(x_vals, result["norm_drops"], '-o', color=color, linewidth=2, 
             markersize=3, label=f'{p_type.capitalize()} (NAOPC={result["naopc"]:.4f})')

plt.title("NAOPC Comparison: Different Perturbation Methods", fontsize=14, fontweight='bold')
plt.xlabel("Percentage of Important Patches Removed (%)", fontsize=12)
plt.ylabel("Normalized Confidence Drop", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_path = f"{RESULTS_DIR}/naopc_comparison_{timestamp}.png"
plt.savefig(save_path, dpi=300)
print(f"\nNAOPC comparison plot saved to: {save_path}")
plt.show()

print("\n" + "="*60)
print("NAOPC Results Summary")
print("="*60)
print(f"\nOriginal Confidence: {naopc_results['mean']['orig_conf']:.4f}")
print("\n{:<12} | {:<12}".format("Perturbation", "NAOPC"))
print("-" * 28)
for p_type, result in naopc_results.items():
    print(f"{p_type.capitalize():<12} | {result['naopc']:<12.4f}")

print("\nInterpretation:")
print("  NAOPC > 0.5  → Excellent (>50% of confidence lost)")
print("  NAOPC 0.3-0.5 → Good")
print("  NAOPC < 0.3  → Weak explanation")

timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
plot_naopc_curve(
    naopc_results["mean"]["norm_drops"],
    naopc_results["mean"]["orig_conf"],
    save_path=f"{RESULTS_DIR}/naopc_curve_{timestamp}.png"
)
# =============================================================
# 📊 Final Prediction
# =============================================================

test_preds = final_model.predict(X_test)
y_test_pred = np.argmax(test_preds, axis=1)
cm = confusion_matrix(y_test, y_test_pred)
ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(cmap=plt.cm.Blues)
plt.title("Final Test Confusion Matrix")
plt.savefig(f"{RESULTS_DIR}/final_test_confusion_matrix{__import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
plt.show()


# =============================================================
# 🧩 Prepare Categorical Labels (One-hot encoding)
# =============================================================
NUM_CLASSES = len(np.unique(y_all))  # or manually define if you already know it

y_temp_cat = tf.keras.utils.to_categorical(y_temp, NUM_CLASSES)
y_test_cat = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# =============================================================
# 📊 Final Metrics Summary (Train vs Test)
# =============================================================

train_preds = final_model.predict(X_temp)
train_labels = np.argmax(y_temp_cat, axis=1)
train_pred_labels = np.argmax(train_preds, axis=1)
test_pred_labels = np.argmax(test_preds, axis=1)

final_metrics = {
    "Dataset": ["Training", "Testing"],
    "Accuracy": [
        accuracy_score(train_labels, train_pred_labels),
        accuracy_score(y_test, test_pred_labels)
    ],
    "Precision": [
        precision_score(train_labels, train_pred_labels, average="macro"),
        precision_score(y_test, test_pred_labels, average="macro")
    ],
    "Recall": [
        recall_score(train_labels, train_pred_labels, average="macro"),
        recall_score(y_test, test_pred_labels, average="macro")
    ],
    "F1-score": [
        f1_score(train_labels, train_pred_labels, average="macro"),
        f1_score(y_test, test_pred_labels, average="macro")
    ]
}

df_final = pd.DataFrame(final_metrics)
print("\n=== Final Training vs Testing Metrics ===")
print(df_final.to_string(index=False))

# Accuracy & Loss Plots
train_loss = final_model.evaluate(X_temp, y_temp_cat, verbose=0)[0]
test_loss = final_model.evaluate(X_test, tf.keras.utils.to_categorical(y_test, NUM_CLASSES), verbose=0)[0]

plt.figure()
plt.bar(["Train", "Test"], [train_loss, test_loss])
plt.title("Loss Comparison")
plt.ylabel("Loss")
plt.savefig(f"{RESULTS_DIR}/train_vs_test_loss.png")
plt.close()

plt.figure()
plt.bar(["Train", "Test"], df_final["Accuracy"])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy")
plt.savefig(f"{RESULTS_DIR}/train_vs_test_accuracy{__import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")
plt.close()
