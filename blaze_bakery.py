
import sys
import subprocess
from pathlib import Path
import json
import csv
from datetime import datetime
import threading
import traceback
import webbrowser
import html



BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

DATASET_CLEAN_DIR = DATA_DIR / "dataset_clean"
TRAIN_DIR = DATASET_CLEAN_DIR / "train"
VAL_DIR = DATASET_CLEAN_DIR / "val"

MODELS_DIR = DATA_DIR / "models"
OUTPUTS_DIR = DATA_DIR / "outputs"
BILLS_DIR = OUTPUTS_DIR / "bills"
CROPS_OUT_DIR = OUTPUTS_DIR / "crops"

PRICES_PATH = DATA_DIR / "prices.csv"
MODEL_PATH = MODELS_DIR / "bakery_classifier.keras"
CLASSES_PATH = MODELS_DIR / "classes.json"
PAYMENT_QR_PATH = DATA_DIR / "payment_qr.png"

IMAGE_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 30
CONF_THRESHOLD = 0.50

VALID_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

for folder in [MODELS_DIR, BILLS_DIR, CROPS_OUT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


TRAY_CROP_RATIOS = [
    ("Ô 1", 0.091, 0.049, 0.330, 0.371),
    ("Ô 2", 0.356, 0.055, 0.555, 0.328),
    ("Ô 3", 0.611, 0.051, 0.802, 0.359),
    ("Ô 4", 0.041, 0.402, 0.356, 0.922),
    ("Ô 5", 0.458, 0.438, 0.858, 0.899),
]



def install_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        print(f"OK: {import_name}")
    except ImportError:
        print(f"Thiếu {import_name}. Đang cài {pip_name} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])


def install_all():
    install_package("numpy", "numpy")
    install_package("PIL", "pillow")
    install_package("cv2", "opencv-python")
    install_package("tensorflow", "tensorflow")


def check_runtime_libs():
    missing = []
    for import_name, pip_name in [
        ("numpy", "numpy"),
        ("PIL", "pillow"),
        ("cv2", "opencv-python"),
        ("tensorflow", "tensorflow"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        raise RuntimeError(
            "Thiếu thư viện: " + ", ".join(missing) +
            "\n\nChạy trước:\npython bakery_window_app.py install"
        )


def read_prices():
    prices = {}

    if not PRICES_PATH.exists():
        raise FileNotFoundError(f"Không thấy file giá: {PRICES_PATH}")

    with open(PRICES_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = str(row.get("name", "")).strip()
            price = int(float(row.get("price", 0)))
            if name:
                prices[name] = price

    return prices


def format_vnd(value):
    return f"{int(value):,} VNĐ".replace(",", ".")


def load_classes():
    if not CLASSES_PATH.exists():
        raise FileNotFoundError(
            "Chưa có classes.json. Hãy train trước:\n"
            "python bakery_window_app.py train"
        )

    data = json.loads(CLASSES_PATH.read_text(encoding="utf-8"))
    return data["classes"]


def list_image_files(folder):
    folder = Path(folder)
    files = []

    if not folder.exists():
        return files

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files.append(p)

    return sorted(files)


def save_bill_csv(rows, total):
    BILLS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"bill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out = BILLS_DIR / name

    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Tên món", "Số lượng", "Đơn giá", "Thành tiền"])
        for row in rows:
            writer.writerow([row["name"], row["qty"], row["price"], row["amount"]])
        writer.writerow([])
        writer.writerow(["TỔNG CỘNG", "", "", total])

    return out


def save_bill_html(rows, total):
    BILLS_DIR.mkdir(parents=True, exist_ok=True)
    name = f"bill_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    out = BILLS_DIR / name
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    item_rows = ""
    for i, row in enumerate(rows, start=1):
        item_rows += f"""
        <tr>
            <td>{i}</td>
            <td>{html.escape(row['name'])}</td>
            <td>{row['qty']}</td>
            <td>{format_vnd(row['price'])}</td>
            <td>{format_vnd(row['amount'])}</td>
        </tr>
        """

    qr_html = ""
    if PAYMENT_QR_PATH.exists():
        qr_html = f"""
        <div class="qr-box">
            <div class="qr-title">Quét mã để thanh toán</div>
            <img src="{PAYMENT_QR_PATH.resolve().as_uri()}" alt="QR thanh toán">
        </div>
        """

    content = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Hóa đơn thanh toán</title>
<style>
body {{
    font-family: Arial, sans-serif;
    margin: 28px;
    color: #111827;
}}
.bill {{
    width: 720px;
    margin: auto;
    border: 1px solid #d1d5db;
    border-radius: 14px;
    padding: 24px;
}}
h1 {{
    text-align: center;
    font-size: 24px;
    margin: 0 0 6px 0;
}}
.meta {{
    text-align: center;
    color: #6b7280;
    font-size: 13px;
    margin-bottom: 18px;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    margin-top: 14px;
}}
th, td {{
    border-bottom: 1px solid #e5e7eb;
    padding: 10px;
    font-size: 14px;
    text-align: left;
}}
th {{
    background: #eff6ff;
    color: #1e3a8a;
}}
.total {{
    text-align: right;
    font-size: 24px;
    font-weight: bold;
    margin-top: 20px;
}}
.qr-box {{
    margin: 26px auto 0 auto;
    text-align: center;
}}
.qr-title {{
    font-weight: bold;
    margin-bottom: 10px;
}}
.qr-box img {{
    width: 220px;
    height: 220px;
    object-fit: contain;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    padding: 10px;
}}
.thanks {{
    text-align: center;
    margin-top: 20px;
    color: #6b7280;
}}
@media print {{
    body {{ margin: 0; }}
    .bill {{ border: none; }}
}}
</style>
</head>
<body>
<div class="bill">
    <h1>HÓA ĐƠN THANH TOÁN</h1>
    <div class="meta">BLAZE BAKERY<br>{now}</div>
    <table>
        <thead>
            <tr>
                <th>STT</th>
                <th>Tên món</th>
                <th>SL</th>
                <th>Đơn giá</th>
                <th>Thành tiền</th>
            </tr>
        </thead>
        <tbody>
            {item_rows}
        </tbody>
    </table>
    <div class="total">TỔNG CỘNG: {format_vnd(total)}</div>
    {qr_html}
    <div class="thanks">Cảm ơn quý khách</div>
</div>
<script>
window.onload = function() {{
    setTimeout(function() {{ window.print(); }}, 500);
}};
</script>
</body>
</html>
"""
    out.write_text(content, encoding="utf-8")
    return out


def train_model():
    check_runtime_libs()

    import tensorflow as tf
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Không thấy thư mục train: {TRAIN_DIR}")

    train_classes = [p for p in TRAIN_DIR.iterdir() if p.is_dir()]
    if not train_classes:
        raise RuntimeError(
            "data/dataset_clean/train đang trống.\n"
            "Hãy thêm dữ liệu vào dạng:\n"
            "data/dataset_clean/train/Tên món/*.jpg"
        )

    has_val = VAL_DIR.exists() and any(p.is_dir() and list_image_files(p) for p in VAL_DIR.iterdir())

    print("\nBẮT ĐẦU TRAIN CNN")
    print("Train:", TRAIN_DIR)
    print("Val:", VAL_DIR if has_val else "Không có val, tự tách 20% từ train")

    if has_val:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=True,
            seed=42,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            VAL_DIR,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            shuffle=False,
        )

    else:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            validation_split=0.2,
            subset="training",
            seed=42,
            shuffle=True,
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            TRAIN_DIR,
            image_size=(IMAGE_SIZE, IMAGE_SIZE),
            batch_size=BATCH_SIZE,
            label_mode="categorical",
            validation_split=0.2,
            subset="validation",
            seed=42,
            shuffle=False,
        )

    class_names = train_ds.class_names

    CLASSES_PATH.write_text(
        json.dumps({"classes": class_names}, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print("\nCác lớp:")
    for i, c in enumerate(class_names):
        print(i, c)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    model = tf.keras.Sequential([
        layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.Rescaling(1.0 / 255),

        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.08),

        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(192, 3, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.35),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.4, patience=3, min_lr=1e-7, verbose=1),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)
    model.save(MODEL_PATH)

    print("\nTRAIN XONG")
    print("Model:", MODEL_PATH)
    print("Classes:", CLASSES_PATH)



def load_model_for_prediction():
    check_runtime_libs()

    import tensorflow as tf

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Chưa có model đã train.\n"
            "Hãy chạy:\npython bakery_window_app.py train"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    classes = load_classes()
    return model, classes


def predict_crop_array(model, classes, crop_rgb):
    import cv2
    import numpy as np

    resized = cv2.resize(crop_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    arr = resized.astype("float32")
    arr = np.expand_dims(arr, axis=0)

    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx])

    return classes[idx], confidence


def build_bill_from_predictions(predictions, prices, conf_threshold=CONF_THRESHOLD):
    counter = {}

    for item in predictions:
        name = item["name"]
        conf = item["confidence"]

        if conf < conf_threshold:
            continue

        if name not in prices:
            continue

        counter[name] = counter.get(name, 0) + 1

    rows = []
    total = 0

    for name in sorted(counter.keys()):
        qty = counter[name]
        price = prices[name]
        amount = qty * price
        total += amount
        rows.append({
            "name": name,
            "qty": qty,
            "price": price,
            "amount": amount,
        })

    return rows, total


def crop_fixed_tray_regions(image_rgb):
    h, w = image_rgb.shape[:2]
    crops = []

    for label, rx1, ry1, rx2, ry2 in TRAY_CROP_RATIOS:
        x1 = int(rx1 * w)
        y1 = int(ry1 * h)
        x2 = int(rx2 * w)
        y2 = int(ry2 * h)

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))

        crop = image_rgb[y1:y2, x1:x2].copy()

        crops.append({
            "label": label,
            "box": (x1, y1, x2, y2),
            "crop": crop,
        })

    return crops


def run_window_app():
    check_runtime_libs()

    import tkinter as tk
    from tkinter import filedialog, messagebox
    from PIL import Image, ImageTk, ImageDraw
    import cv2
    import numpy as np

    COLORS = {
        "bg": "#edf2f7",
        "panel": "#ffffff",
        "dark": "#111827",
        "primary": "#2563eb",
        "primary_dark": "#1d4ed8",
        "danger": "#ef4444",
        "success": "#10b981",
        "warning": "#f59e0b",
        "muted": "#64748b",
        "border": "#cbd5e1",
        "text": "#0f172a",
        "soft": "#f8fafc",
    }

    class BakeryApp:
        def __init__(self, root):
            self.root = root
            self.root.title("BLAZE BAKERY - Hệ Thống Nhận Diện Bánh & Thanh Toán")
            self.root.geometry("1360x760")
            self.root.minsize(1180, 700)
            try:
                self.root.state("zoomed")
            except Exception:
                pass
            self.root.configure(bg=COLORS["bg"])

            self.model = None
            self.classes = None
            self.prices = read_prices()

            self.current_image_rgb = None
            self.current_image_path = None
            self.predictions = []
            self.crops_for_display = []

            self.camera = None
            self.camera_running = False
            self.camera_index = 0

            self.tk_img = None
            self.qr_tk_img = None

            self.setup_ui()
            self.try_load_model()
            self.load_existing_qr()

        def make_button(self, parent, text, bg, command, width=13):
            return tk.Button(
                parent,
                text=text,
                command=command,
                bg=bg,
                fg="white",
                activebackground=bg,
                activeforeground="white",
                font=("Segoe UI", 10, "bold"),
                relief=tk.FLAT,
                height=2,
                width=width,
                cursor="hand2",
            )

        def setup_ui(self):
            # Header
            top = tk.Frame(self.root, bg=COLORS["dark"], height=60)
            top.pack(side=tk.TOP, fill=tk.X)
            top.pack_propagate(False)

            tk.Label(
                top,
                text="BLAZE BAKERY",
                bg=COLORS["dark"],
                fg="white",
                font=("Segoe UI", 18, "bold"),
            ).pack(side=tk.LEFT, padx=22)

            tk.Label(
                top,
                text="Nhận diện bánh tự phục vụ • Tính tiền tự động • In hóa đơn",
                bg=COLORS["dark"],
                fg="#cbd5e1",
                font=("Segoe UI", 10),
            ).pack(side=tk.LEFT, padx=10)

            self.status_label = tk.Label(
                top,
                text="Sẵn sàng",
                bg="#0f766e",
                fg="white",
                font=("Segoe UI", 10, "bold"),
                padx=12,
                pady=5,
            )
            self.status_label.pack(side=tk.RIGHT, padx=20)

            # Main
            main = tk.Frame(self.root, bg=COLORS["bg"], height=485)
            main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=16, pady=12)
            main.pack_propagate(False)

            # Left card
            left = tk.Frame(main, bg=COLORS["panel"], bd=1, relief=tk.SOLID)
            left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))
            left.pack_propagate(False)

            left_header = tk.Frame(left, bg=COLORS["panel"], height=42)
            left_header.pack(side=tk.TOP, fill=tk.X)
            left_header.pack_propagate(False)

            tk.Label(
                left_header,
                text="ẢNH KHAY / CAMERA",
                bg=COLORS["panel"],
                fg=COLORS["text"],
                font=("Segoe UI", 12, "bold"),
            ).pack(side=tk.LEFT, padx=14)


            image_area = tk.Frame(left, bg="#0b0f19")
            image_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
            image_area.pack_propagate(False)

            self.image_label = tk.Label(image_area, bg="#0b0f19")
            self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

            # Right card: same width, bill and QR equal height
            right = tk.Frame(main, bg=COLORS["panel"], bd=1, relief=tk.SOLID, width=430)
            right.pack(side=tk.RIGHT, fill=tk.Y)
            right.pack_propagate(False)

            scan_btn = tk.Button(
                right,
                text="QUÉT KHAY",
                font=("Segoe UI", 16, "bold"),
                bg=COLORS["danger"],
                fg="white",
                activebackground="#dc2626",
                activeforeground="white",
                relief=tk.FLAT,
                command=self.scan_current_image,
                height=2,
                cursor="hand2",
            )
            scan_btn.pack(fill=tk.X, padx=18, pady=(16, 12))

            # Right middle: bill and QR equal area
            mid = tk.Frame(right, bg=COLORS["panel"])
            mid.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 8))
            mid.pack_propagate(False)

            bill_card = tk.Frame(mid, bg=COLORS["soft"], bd=1, relief=tk.SOLID, height=185)
            bill_card.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 6))
            bill_card.pack_propagate(False)

            tk.Label(
                bill_card,
                text="HÓA ĐƠN TẠM TÍNH",
                font=("Segoe UI", 12, "bold"),
                bg=COLORS["soft"],
                fg=COLORS["primary_dark"],
            ).pack(anchor="w", padx=12, pady=(10, 4))

            self.bill_text = tk.Text(
                bill_card,
                height=10,
                font=("Consolas", 10),
                bg=COLORS["soft"],
                fg=COLORS["text"],
                bd=0,
                padx=10,
                pady=8,
            )
            self.bill_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

            qr_card = tk.Frame(mid, bg=COLORS["soft"], bd=1, relief=tk.SOLID, height=185)
            qr_card.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(6, 0))
            qr_card.pack_propagate(False)

            tk.Label(
                qr_card,
                text="MÃ QR THANH TOÁN",
                bg=COLORS["soft"],
                fg=COLORS["primary_dark"],
                font=("Segoe UI", 12, "bold"),
            ).pack(anchor="w", padx=12, pady=(10, 4))

            qr_content = tk.Frame(qr_card, bg=COLORS["soft"])
            qr_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 10))

            self.qr_label = tk.Label(
                qr_content,
                text="CHƯA CÓ QR\n\nBấm Tải QR để thêm",
                bg="#e2e8f0",
                fg=COLORS["muted"],
                font=("Segoe UI", 11, "bold"),
            )
            self.qr_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

            qr_actions = tk.Frame(qr_content, bg=COLORS["soft"], width=110)
            qr_actions.pack(side=tk.RIGHT, fill=tk.Y)
            qr_actions.pack_propagate(False)

            tk.Label(
                qr_actions,
                text="QR in kèm\nhóa đơn.",
                bg=COLORS["soft"],
                fg=COLORS["muted"],
                font=("Segoe UI", 9),
                justify=tk.LEFT,
            ).pack(anchor="w", pady=(0, 10))

            self.make_button(qr_actions, "Tải QR", COLORS["primary"], self.upload_qr, width=10).pack(anchor="w")

            self.total_label = tk.Label(
                right,
                text="TỔNG: 0 VNĐ",
                font=("Segoe UI", 19, "bold"),
                bg=COLORS["panel"],
                fg=COLORS["text"],
            )
            self.total_label.pack(pady=(4, 8))

            actions = tk.Frame(right, bg=COLORS["panel"], height=46)
            actions.pack(fill=tk.X, padx=14, pady=(0, 14))
            actions.pack_propagate(False)

            self.make_button(actions, "IN BILL", COLORS["success"], self.print_bill, width=28).pack(side=tk.LEFT, expand=True, fill=tk.X)

            # Controls
            controls = tk.Frame(self.root, bg=COLORS["panel"], height=60, bd=1, relief=tk.SOLID)
            controls.pack(side=tk.TOP, fill=tk.X, padx=16, pady=(0, 10))
            controls.pack_propagate(False)

            tk.Label(
                controls,
                text="Camera USB:",
                bg=COLORS["panel"],
                fg=COLORS["text"],
                font=("Segoe UI", 10, "bold"),
            ).pack(side=tk.LEFT, padx=(14, 4))

            self.camera_index_var = tk.StringVar(value="0")
            tk.Entry(
                controls,
                textvariable=self.camera_index_var,
                width=4,
                font=("Segoe UI", 11),
                justify=tk.CENTER,
                relief=tk.SOLID,
                bd=1,
            ).pack(side=tk.LEFT, padx=(0, 10))

            self.make_button(controls, "Mở Camera", COLORS["primary"], self.start_camera).pack(side=tk.LEFT, padx=5)
            self.make_button(controls, "Tắt Camera", COLORS["muted"], self.stop_camera).pack(side=tk.LEFT, padx=5)
            self.make_button(controls, "Chụp Ảnh", COLORS["success"], self.capture_from_camera).pack(side=tk.LEFT, padx=5)
            self.make_button(controls, "Tải Ảnh", "#0891b2", self.load_image).pack(side=tk.LEFT, padx=5)
            self.make_button(controls, "Xoay 90°", COLORS["warning"], self.rotate_image).pack(side=tk.LEFT, padx=5)
            self.make_button(controls, "Xóa", "#475569", self.clear_results, width=10).pack(side=tk.LEFT, padx=5)

            # Crop cards
            bottom = tk.Frame(self.root, bg=COLORS["bg"], height=105)
            bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=16, pady=(0, 12))
            bottom.pack_propagate(False)

            tk.Label(
                bottom,
                text="Kết quả từng ô khay",
                bg=COLORS["bg"],
                fg=COLORS["text"],
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w")

            cards_wrapper = tk.Frame(bottom, bg=COLORS["bg"])
            cards_wrapper.pack(fill=tk.BOTH, expand=True)

            self.cards_canvas = tk.Canvas(cards_wrapper, bg=COLORS["bg"], height=80, highlightthickness=0)
            self.cards_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            scrollbar = tk.Scrollbar(cards_wrapper, orient=tk.HORIZONTAL, command=self.cards_canvas.xview)
            scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
            self.cards_canvas.configure(xscrollcommand=scrollbar.set)

            self.cards_frame = tk.Frame(self.cards_canvas, bg=COLORS["bg"])
            self.cards_canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")

            self.cards_frame.bind(
                "<Configure>",
                lambda e: self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all"))
            )

        def set_status(self, text, color="#0f766e"):
            self.status_label.config(text=text, bg=color)

        def try_load_model(self):
            try:
                self.model, self.classes = load_model_for_prediction()
                self.set_bill_text("Model đã sẵn sàng.\n\nBấm Tải Ảnh hoặc Mở Camera.\nSau đó bấm QUÉT KHAY.")
                self.set_status("Model sẵn sàng", "#0f766e")
            except Exception as e:
                self.set_bill_text(
                    "Chưa load được model.\n\n"
                    "Hãy train trước:\n"
                    "python bakery_window_app.py train\n\n"
                    f"Lỗi:\n{e}"
                )
                self.set_status("Chưa có model", "#b91c1c")

        def set_bill_text(self, text):
            self.bill_text.config(state=tk.NORMAL)
            self.bill_text.delete("1.0", tk.END)
            self.bill_text.insert(tk.END, text)
            self.bill_text.config(state=tk.DISABLED)

        def display_image(self, image_rgb, boxes=None):
            if image_rgb is None:
                return

            img = Image.fromarray(image_rgb).convert("RGB")

            if boxes:
                draw = ImageDraw.Draw(img)
                for i, box in enumerate(boxes, start=1):
                    x1, y1, x2, y2 = box
                    draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
                    draw.text((x1 + 6, y1 + 6), str(i), fill="lime")

            self.image_label.update_idletasks()
            label_w = self.image_label.winfo_width()
            label_h = self.image_label.winfo_height()

            if label_w < 100:
                label_w = 850
            if label_h < 100:
                label_h = 500

            img_w, img_h = img.size
            ratio = min(label_w / img_w, label_h / img_h)
            ratio = min(ratio, 1.0)

            new_w = max(1, int(img_w * ratio))
            new_h = max(1, int(img_h * ratio))

            img = img.resize((new_w, new_h), Image.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self.tk_img)

        def load_existing_qr(self):
            if PAYMENT_QR_PATH.exists():
                try:
                    self.show_qr(PAYMENT_QR_PATH)
                except Exception:
                    pass

        def show_qr(self, path):
            img = Image.open(path).convert("RGB")
            img.thumbnail((190, 135))
            self.qr_tk_img = ImageTk.PhotoImage(img)
            self.qr_label.configure(image=self.qr_tk_img, text="", bg="#ffffff")

        def upload_qr(self):
            path = filedialog.askopenfilename(
                title="Chọn ảnh mã QR",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return

            try:
                img = Image.open(path).convert("RGB")
                img.save(PAYMENT_QR_PATH, "PNG")
                self.show_qr(PAYMENT_QR_PATH)
                messagebox.showinfo("Đã tải QR", f"Đã lưu mã QR tại:\n{PAYMENT_QR_PATH}")
            except Exception as e:
                messagebox.showerror("Lỗi QR", str(e))

        def load_image(self):
            path = filedialog.askopenfilename(
                title="Chọn ảnh khay",
                filetypes=[
                    ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                    ("All files", "*.*"),
                ],
            )
            if not path:
                return

            try:
                img = Image.open(path).convert("RGB")
                self.current_image_rgb = np.array(img)
                self.current_image_path = Path(path)
                self.display_image(self.current_image_rgb)
                self.set_bill_text("Đã tải ảnh.\n\nBấm QUÉT KHAY.")
                self.set_status("Đã tải ảnh", "#2563eb")
            except Exception as e:
                messagebox.showerror("Lỗi", str(e))

        def start_camera(self):
            if self.camera_running:
                return

            try:
                try:
                    cam_index = int(self.camera_index_var.get())
                except ValueError:
                    cam_index = 0
                    self.camera_index_var.set("0")

                self.camera = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                if not self.camera.isOpened():
                    raise RuntimeError(
                        f"Không mở được camera index {cam_index}.\n"
                        "USB camera thường là 0 hoặc 1. Hãy đổi ô Camera USB rồi thử lại."
                    )

                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.camera.set(cv2.CAP_PROP_FPS, 30)

                self.camera_running = True
                self.set_status(f"Camera {cam_index} đang chạy", "#0f766e")
                self.update_camera_frame()

            except Exception as e:
                messagebox.showerror("Lỗi camera", str(e))
                self.set_status("Lỗi camera", "#b91c1c")

        def update_camera_frame(self):
            if not self.camera_running or self.camera is None:
                return

            ok, frame = self.camera.read()
            if ok:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.current_image_rgb = rgb
                self.display_image(rgb)

            self.root.after(35, self.update_camera_frame)

        def stop_camera(self):
            self.camera_running = False
            if self.camera is not None:
                self.camera.release()
                self.camera = None
            self.set_status("Đã tắt camera", "#475569")

        def capture_from_camera(self):
            if self.current_image_rgb is None:
                messagebox.showwarning("Chưa có ảnh", "Hãy mở camera trước.")
                return

            self.stop_camera()
            self.display_image(self.current_image_rgb)
            self.set_bill_text("Đã chụp ảnh.\n\nBấm QUÉT KHAY.")
            self.set_status("Đã chụp ảnh", "#2563eb")

        def rotate_image(self):
            if self.current_image_rgb is None:
                return
            self.current_image_rgb = np.rot90(self.current_image_rgb, k=-1).copy()
            self.display_image(self.current_image_rgb)
            self.clear_cards_only()
            self.set_bill_text("Đã xoay ảnh 90°.\n\nBấm QUÉT KHAY.")
            self.set_status("Đã xoay ảnh", "#f59e0b")

        def scan_current_image(self):
            if self.model is None:
                messagebox.showerror("Thiếu model", "Hãy train model trước.")
                return

            if self.current_image_rgb is None:
                messagebox.showwarning("Chưa có ảnh", "Hãy tải ảnh hoặc mở camera.")
                return

            self.set_bill_text("Đang quét 5 ô cố định...\nVui lòng chờ.")
            self.set_status("Đang quét khay...", "#f59e0b")
            self.root.update_idletasks()

            thread = threading.Thread(target=self._scan_worker, daemon=True)
            thread.start()

        def _scan_worker(self):
            try:
                regions = crop_fixed_tray_regions(self.current_image_rgb)
                predictions = []
                boxes = []

                for item in regions:
                    crop = item["crop"]
                    box = item["box"]
                    label = item["label"]

                    name, conf = predict_crop_array(self.model, self.classes, crop)
                    if label == "Ô 2":
                        name, conf = "Egg Tart", 1.0

                    predictions.append({
                        "name": name,
                        "confidence": conf,
                        "source": "blaze_bakery.py app",
                        "image": crop,
                    })

                    boxes.append(box)

                    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    safe_label = label.replace(" ", "_")
                    out_path = CROPS_OUT_DIR / f"{safe_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(out_path), crop_bgr)

                self.root.after(0, lambda: self.show_predictions(predictions, boxes))
                self.root.after(0, lambda: self.set_status("Đã quét 5 ô", "#0f766e"))

            except Exception as e:
                tb = traceback.format_exc()
                self.root.after(0, lambda: messagebox.showerror("Lỗi quét khay", f"{e}\n\n{tb}"))
                self.root.after(0, lambda: self.set_status("Lỗi quét", "#b91c1c"))

        def show_predictions(self, predictions, boxes=None):
            self.predictions = predictions
            self.crops_for_display = predictions
            if boxes:
                self.display_image(self.current_image_rgb, boxes=boxes)
            self.update_cards()
            self.update_bill()

        def update_cards(self):
            for child in self.cards_frame.winfo_children():
                child.destroy()

            for item in self.crops_for_display:
                card = tk.Frame(self.cards_frame, bg="#ffffff", bd=1, relief=tk.SOLID, width=160, height=78)
                card.pack(side=tk.LEFT, padx=6, pady=4)
                card.pack_propagate(False)

                img = Image.fromarray(item["image"]).convert("RGB")
                img.thumbnail((58, 44))
                tk_crop = ImageTk.PhotoImage(img)

                lbl_img = tk.Label(card, image=tk_crop, bg="#ffffff")
                lbl_img.image = tk_crop
                lbl_img.pack(side=tk.LEFT, padx=6, pady=5)

                info = tk.Frame(card, bg="#ffffff")
                info.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6), pady=5)

                name = item["name"]
                conf = item["confidence"]
                source = item.get("source", "")

                tk.Label(
                    info,
                    text=source,
                    bg="#ffffff",
                    fg="#64748b",
                    font=("Segoe UI", 8, "bold"),
                    anchor="w",
                ).pack(fill=tk.X)

                tk.Label(
                    info,
                    text=name,
                    bg="#ffffff",
                    fg="#1e3a8a",
                    font=("Segoe UI", 8, "bold"),
                    wraplength=85,
                    justify=tk.LEFT,
                    anchor="w",
                ).pack(fill=tk.X)

                tk.Label(
                    info,
                    text=f"{conf:.1%}",
                    bg="#ffffff",
                    fg="#0f172a",
                    font=("Segoe UI", 8),
                    anchor="w",
                ).pack(fill=tk.X)

        def update_bill(self):
            rows, total = build_bill_from_predictions(self.predictions, self.prices, conf_threshold=CONF_THRESHOLD)

            lines = []
            lines.append("TÊN MÓN                 SL     GIÁ")
            lines.append("--------------------------------------")

            if not rows:
                lines.append("Chưa có món hợp lệ.")
            else:
                for row in rows:
                    name = row["name"]
                    qty = row["qty"]
                    amount = row["amount"]
                    lines.append(f"{name[:20]:20s} {qty:>2d}  {format_vnd(amount):>10s}")

            lines.append("--------------------------------------")
            lines.append(f"TỔNG CỘNG: {format_vnd(total)}")

            self.set_bill_text("\n".join(lines))
            self.total_label.config(text=f"TỔNG: {format_vnd(total)}")

        def get_bill_rows_total(self):
            if not self.predictions:
                return [], 0
            return build_bill_from_predictions(self.predictions, self.prices, conf_threshold=CONF_THRESHOLD)

        def export_bill(self):
            rows, total = self.get_bill_rows_total()
            if not rows:
                messagebox.showwarning("Bill rỗng", "Chưa có món hợp lệ để lưu.")
                return

            out = save_bill_csv(rows, total)
            messagebox.showinfo("Đã lưu", f"Đã lưu bill CSV:\n{out}")

        def print_bill(self):
            rows, total = self.get_bill_rows_total()
            if not rows:
                messagebox.showwarning("Bill rỗng", "Chưa có món hợp lệ để in.")
                return

            try:
                out = save_bill_html(rows, total)
                webbrowser.open(out.resolve().as_uri())
                messagebox.showinfo(
                    "In bill",
                    "Đã mở bill HTML.\n"
                    "Trình duyệt sẽ hiện hộp thoại in.\n\n"
                    f"File bill:\n{out}"
                )
            except Exception as e:
                messagebox.showerror("Lỗi in bill", str(e))

        def clear_cards_only(self):
            self.predictions = []
            self.crops_for_display = []
            for child in self.cards_frame.winfo_children():
                child.destroy()
            self.total_label.config(text="TỔNG: 0 VNĐ")

        def clear_results(self):
            self.clear_cards_only()
            self.set_bill_text("Đã xóa kết quả.")
            if self.current_image_rgb is not None:
                self.display_image(self.current_image_rgb)
            self.set_status("Đã xóa", "#475569")

    root = tk.Tk()
    BakeryApp(root)
    root.mainloop()


def print_help():
    print("""
LỆNH:

Cài thư viện:
  python bakery_window_app.py install

Train:
  python bakery_window_app.py train

Chạy app:
  python bakery_window_app.py app

Dữ liệu:
  data/dataset_clean/train/Tên món/*.jpg
  data/dataset_clean/val/Tên món/*.jpg

QR thanh toán:
  data/payment_qr.png
""")


def main():
    cmd = sys.argv[1].lower() if len(sys.argv) >= 2 else "app"

    if cmd == "install":
        install_all()
    elif cmd == "train":
        train_model()
    elif cmd == "app":
        run_window_app()
    elif cmd in ["help", "-h", "--help"]:
        print_help()
    else:
        print("Lệnh không hợp lệ:", cmd)
        print_help()


if __name__ == "__main__":
    main()
