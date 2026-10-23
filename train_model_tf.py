"""Transfer-learning training with font-grouped holdout and train-only augmentation."""
from pathlib import Path
import json, csv, time, random
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
from labels import LETTERS

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
MODEL_DIR = ROOT / "model"
EVAL_DIR = ROOT / "evaluation"
IMG_SIZE = (160,160)
BATCH = 16
HEAD_EPOCHS = 12
FINE_EPOCHS = 8
SEED = 501
MODEL_DIR.mkdir(exist_ok=True); EVAL_DIR.mkdir(exist_ok=True)
random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

def collect():
    paths=[]; labels=[]; groups=[]
    for cid, (_, name) in enumerate(LETTERS):
        folder = DATASET / f"{cid:02d}_{name}"
        if not folder.exists():
            raise FileNotFoundError(f"Missing {folder}. Run python generate_dataset.py first.")
        imgs = sorted([p for p in folder.iterdir() if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp"}])
        if not imgs: raise ValueError(f"No images found in {folder}")
        for p in imgs:
            paths.append(str(p)); labels.append(cid)
            prefix = p.stem.split("_",1)[0]
            groups.append(prefix if prefix.startswith("font") else "unidentified")
    return np.array(paths), np.array(labels, np.int32), np.array(groups)

def split(paths, y, groups):
    fonts = sorted(set(groups) - {"unidentified"})
    if len(fonts) >= 3 and np.all(groups != "unidentified"):
        test_group = fonts[-1]
        val_group = fonts[-2]
        test = np.where(groups == test_group)[0]
        val = np.where(groups == val_group)[0]
        train = np.where((groups != test_group) & (groups != val_group))[0]
        desc = f"font-grouped holdout: validation={val_group}, test={test_group}"
    else:
        from sklearn.model_selection import train_test_split
        idx = np.arange(len(y))
        train, temp = train_test_split(idx, test_size=.30, random_state=SEED, stratify=y)
        val, test = train_test_split(temp, test_size=.50, random_state=SEED, stratify=y[temp])
        desc = "stratified random split; font leakage may remain"
    return train, val, test, desc

def decode(path, label):
    img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    img.set_shape([None,None,3])
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    return tf.cast(img, tf.float32), tf.cast(label, tf.int32)

def dataset(paths, labels, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training: ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

def build_model():
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.04, fill_mode="constant", fill_value=255.0),
        tf.keras.layers.RandomTranslation(0.04, 0.04, fill_mode="constant", fill_value=255.0),
        tf.keras.layers.RandomZoom((-0.06,0.06), (-0.06,0.06), fill_mode="constant", fill_value=255.0),
        tf.keras.layers.RandomContrast(0.12),
        tf.keras.layers.RandomBrightness(0.08, value_range=(0,255)),
    ], name="training_augmentation")
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE,3))
    base.trainable = False
    inp = tf.keras.Input(shape=(*IMG_SIZE,3), name="image")
    x = augmentation(inp)  # Keras augmentation layers are training-only when model.fit is used.
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(.30)(x)
    out = tf.keras.layers.Dense(len(LETTERS), activation="softmax", name="class_probabilities")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model, base

def main():
    paths, y, groups = collect()
    tr, va, te, desc = split(paths,y,groups)
    print(f"Images={len(paths)} train={len(tr)} validation={len(va)} test={len(te)}")
    print(desc)
    train_ds = dataset(paths[tr], y[tr], True)
    val_ds = dataset(paths[va], y[va])
    test_ds = dataset(paths[te], y[te])
    model, base = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4, mode="max", restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=.3, patience=2, min_lr=1e-6)
    ]
    start = time.time()
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, callbacks=callbacks, verbose=1)
    base.trainable = True
    for layer in base.layers[:-30]: layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization): layer.trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=FINE_EPOCHS, callbacks=callbacks, verbose=1)
    probs = model.predict(test_ds, verbose=0)
    pred = probs.argmax(axis=1)
    accuracy = float(accuracy_score(y[te],pred))
    macro = float(f1_score(y[te],pred,average="macro",zero_division=0))
    mapping = [{"class_id":i,"letter":ch,"name":name} for i,(ch,name) in enumerate(LETTERS)]
    model.save(MODEL_DIR / "persian_model.keras")
    (MODEL_DIR / "label_mapping.json").write_text(json.dumps(mapping,ensure_ascii=False,indent=2),encoding="utf-8")
    report = classification_report(y[te],pred,labels=list(range(len(LETTERS))),target_names=[n for _,n in LETTERS],output_dict=True,zero_division=0)
    (EVAL_DIR / "classification_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    cm = confusion_matrix(y[te],pred,labels=list(range(len(LETTERS))))
    with open(EVAL_DIR / "confusion_matrix.csv","w",newline="",encoding="utf-8-sig") as f:
        w=csv.writer(f); w.writerow(["true/predicted"]+[n for _,n in LETTERS])
        for i,(_,name) in enumerate(LETTERS): w.writerow([name]+cm[i].tolist())
    summary = {"selected_model":"MobileNetV2","test_accuracy":accuracy,"test_macro_f1":macro,
               "split_description":desc,"train_images":len(tr),"validation_images":len(va),
               "test_images":len(te),"class_count":len(LETTERS),"seconds":round(time.time()-start,1),
               "note":"Synthetic font-only performance does not establish handwritten recognition accuracy."}
    (EVAL_DIR / "evaluation_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    plt.figure(figsize=(8,5))
    plt.plot(h1.history.get("accuracy",[])+h2.history.get("accuracy",[]),label="train accuracy")
    plt.plot(h1.history.get("val_accuracy",[])+h2.history.get("val_accuracy",[]),label="validation accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(EVAL_DIR/"training_history.png",dpi=150); plt.close()
    print(f"Test accuracy={accuracy:.2%}; macro-F1={macro:.2%}")
    print("Saved model/persian_model.keras, label_mapping.json, and evaluation reports.")

if __name__ == "__main__":
    main()
