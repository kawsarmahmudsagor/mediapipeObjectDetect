import os
import shutil
import json
import math
from collections import defaultdict

import tensorflow as tf
assert tf.__version__.startswith('2')
from mediapipe_model_maker import object_detector

import matplotlib.pyplot as plt
from matplotlib import patches, patheffects

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# ----------------------------
# Visualization helpers
# ----------------------------
def draw_outline(obj):
    obj.set_path_effects([patheffects.Stroke(linewidth=4, foreground='black'), patheffects.Normal()])

def draw_box(ax, bb):
    patch = ax.add_patch(patches.Rectangle((bb[0], bb[1]), bb[2], bb[3], fill=False, edgecolor='red', lw=2))
    draw_outline(patch)

def draw_text(ax, bb, txt, disp):
    text = ax.text(bb[0], (bb[1]-disp), txt, verticalalignment='top', color='white', fontsize=10, weight='bold')
    draw_outline(text)

def draw_bbox(ax, annotations_list, id_to_label, image_shape):
    for annotation in annotations_list:
        image_id = annotation["category_id"]
        bbox = annotation["bbox"]
        draw_box(ax, bbox)
        draw_text(ax, bbox, id_to_label[image_id], image_shape[0] * 0.05)

def visualize(dataset_folder, max_examples=None):
    with open(os.path.join(dataset_folder, "labels.json"), "r") as f:
        labels_json = json.load(f)
    images = labels_json["images"]
    image_id_to_label = {item["id"]: item["name"] for item in labels_json["categories"]}
    image_annots = defaultdict(list)
    for annotation_obj in labels_json["annotations"]:
        image_id = annotation_obj["image_id"]
        image_annots[image_id].append(annotation_obj)

    if max_examples is None:
        max_examples = len(image_annots.items())
    n_rows = math.ceil(max_examples / 3)
    fig, axs = plt.subplots(n_rows, 3, figsize=(24, n_rows*8))
    for ind, (image_id, annotations_list) in enumerate(list(image_annots.items())[:max_examples]):
        ax = axs[ind//3, ind%3]
        img = plt.imread(os.path.join(dataset_folder, "images", images[image_id]["file_name"]))
        ax.imshow(img)
        draw_bbox(ax, annotations_list, image_id_to_label, img.shape)
    plt.show()

# ----------------------------
# Paths
# ----------------------------
train_dataset_path = "/home/xr23/myenv/MediapipeObjectDetect/Face-Detection-Final-V1-7/train"
validation_dataset_path = "/home/xr23/myenv/MediapipeObjectDetect/Face-Detection-Final-V1-7/valid"
export_base_dir = "detection_exported_model"


# Visualize some examples
visualize(train_dataset_path, 9)

# ----------------------------
# Load datasets
# ----------------------------
train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)

# ----------------------------
# Warmup LR schedule
# ----------------------------
class WarmUpLearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, base_lr, warmup_steps, total_steps, after_warmup_decay="cosine"):
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.after_warmup_decay = after_warmup_decay

        if after_warmup_decay == "cosine":
            self.decay_schedule = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=base_lr,
                decay_steps=max(1, total_steps - warmup_steps)
            )
        elif after_warmup_decay == "constant":
            self.decay_schedule = lambda step: base_lr
        else:
            raise ValueError("Unsupported decay type")

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_lr = self.base_lr * (step / tf.cast(self.warmup_steps, tf.float32))
        return tf.cond(
            step < self.warmup_steps,
            lambda: warmup_lr,
            lambda: self.decay_schedule(step - self.warmup_steps)
        )

# ----------------------------
# Hyperparameter Sets
# ----------------------------
hyperparams_list = [
    
    {"batch_size": 32, "epochs": 300, "learning_rate": 0.03, "warmup_steps": 500},
   
]

spec = object_detector.SupportedModels.MOBILENET_MULTI_AVG_I384

# Log results
results_log = []

# ----------------------------
# Train multiple models
# ----------------------------
for i, hparam_set in enumerate(hyperparams_list, start=1):
    print(f"\n🔹 Training model version {i} with hyperparameters: {hparam_set}")

    # Base HParams (with float LR only)
    hparams = object_detector.HParams(
        export_dir="detection_exported_model",
        batch_size=hparam_set["batch_size"],
        epochs=hparam_set["epochs"],
        learning_rate=hparam_set["learning_rate"]  # float, not schedule
    )

    options = object_detector.ObjectDetectorOptions(
        supported_model=spec,
        hparams=hparams
    )

    # Create model
    model = object_detector.ObjectDetector.create(
        train_data=train_data,
        validation_data=validation_data,
        options=options
    )

    # Override optimizer with warmup schedule
    steps_per_epoch = math.ceil(train_data.size / hparam_set["batch_size"])
    total_steps = steps_per_epoch * hparam_set["epochs"]

    lr_schedule = WarmUpLearningRate(
        base_lr=hparam_set["learning_rate"],
        warmup_steps=hparam_set["warmup_steps"],
        total_steps=total_steps,
        after_warmup_decay="cosine"
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model._model.compile(
        optimizer=optimizer,
        loss=model._model.loss,
        metrics=model._model.metrics
    )

    # Evaluate
    loss, coco_metrics = model.evaluate(validation_data, batch_size=4)
    print(f"Validation loss: {loss}")
    print(f"Validation COCO metrics: {coco_metrics}")
    
    # Export TFLite model
    model.export_model()
    tflite_model_path = os.path.join(hparams.export_dir, "model.tflite")
    destination = os.path.join(export_base_dir, f"MediapipeUpdatedDataset{i}.tflite")
    shutil.copy(tflite_model_path, destination)
    print(f"✅ Model version {i} exported as {destination}")
    
    # Log results with hyperparameters explicitly
    results_log.append({
        "version": i,
        "hyperparameters": {
            "batch_size": int(hparam_set["batch_size"]),
            "epochs": int(hparam_set["epochs"]),
            "learning_rate": float(hparam_set["learning_rate"])
        },
        "validation_loss": [float(x) for x in loss],
        "coco_metrics": {k: float(v) for k, v in coco_metrics.items()},  # convert dict values
        "tflite_file": destination
    })
    with open(f"training_results_log{i}.json", "w") as f:
        json.dump(results_log, f, indent=4)
    print(f"\nAll training results logged in training_results_log{i}.json")


# Save log to JSON

