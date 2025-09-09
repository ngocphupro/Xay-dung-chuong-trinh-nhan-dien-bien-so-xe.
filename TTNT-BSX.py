import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import cv2
import easyocr
import os
import torch
from threading import Thread
import pytesseract
from datetime import datetime
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Biến toàn cục
current_image_path = None
webcam_frame = None
running = True
last_detected = [None]

# Hàm khởi tạo EasyOCR Reader
# - Kiểm tra xem có GPU không để sử dụng CUDA cho tốc độ nhanh hơn.
# - Nếu có GPU, in thông tin GPU và khởi tạo Reader với GPU=True.
# - Nếu không, sử dụng CPU.
def initialize_reader():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory // (1024 ** 2)
        print(f"GPU detected: {gpu_name} with {gpu_memory} MB memory. Using CUDA for processing.")
        return easyocr.Reader(["en", "vi"], gpu=True, model_storage_directory=None, download_enabled=True)
    else:
        print("No GPU detected: Defaulting to CPU.")
        return easyocr.Reader(["en", "vi"], gpu=False, model_storage_directory=None, download_enabled=True)

# Khởi tạo EasyOCR Reader ngay từ đầu
reader = initialize_reader()

# ======================================
# Quy trình 1: Nhận diện biển số (License Plate Detection)
# ======================================

# Hàm phát hiện biển số xe từ ảnh tĩnh (sử dụng cho ảnh tải lên hoặc chụp)
# Thuật toán:
def detect_license_plate(image_path):
    # Bước 1: Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        return None, "Không thể đọc ảnh"

    # Bước 2: Chuyển sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bước 3: Áp dụng GaussianBlur để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 4: Phát hiện biên với Canny
    edged = cv2.Canny(blurred, 50, 200)

    # Bước 5: Tìm contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    plate_contour = None
    for contour in contours:
        # Tính chu vi contour
        peri = cv2.arcLength(contour, True)
        # Xấp xỉ contour để tìm hình 4 cạnh
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Nếu contour có 4 điểm, có thể là biển số
        if len(approx) == 4:
            plate_contour = approx
            break

    # Nếu tìm thấy biển số
    if plate_contour is not None:
        # Vẽ contour lên ảnh gốc
        result_image = image.copy()
        cv2.drawContours(result_image, [plate_contour], -1, (0, 255, 0), 3)

        # Tạo mask để cắt biển số (không sử dụng mask ở đây, mà dùng bounding rect)
        x, y, w, h = cv2.boundingRect(plate_contour)
        plate_image = image[y:y + h, x:x + w]

        return result_image, plate_image
    else:
        return None, None

# Hàm xử lý frame từ webcam để phát hiện biển số (tích hợp detect và OCR cơ bản)
# Thuật toán:
def process_frame(frame, last_detected):
    # Bước 1: Chuyển sang grayscale và tăng cường tương phản
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Bước 2: Lọc nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Bước 3: Áp dụng Canny edge detection
    edges = cv2.Canny(blurred, 100, 200)

    # Bước 4: Tìm các đường viền
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Bước 5: Xấp xỉ đường viền để tìm hình chữ nhật
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # Kiểm tra nếu đường viền là hình chữ nhật (4 đỉnh)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            # Kiểm tra tỷ lệ khung hình phù hợp với biển số
            aspect_ratio = w / float(h)
            if 2.0 <= aspect_ratio <= 4.0 and w >= 50 and h >= 20:
                # Bước 6: Cắt vùng biển số và resize
                plate = gray[y:y + h, x:x + w]
                plate_resized = cv2.resize(plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                # Bước 7: OCR với Pytesseract
                custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
                text = pytesseract.image_to_string(plate_resized, config=custom_config).strip()

                # Bước 8: Kiểm tra tính hợp lệ và vẽ/lưu
                if len(text) < 4:
                    continue

                # Vẽ khung và văn bản trên ảnh gốc
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

                # Lưu ảnh với timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                os.makedirs("images", exist_ok=True)
                plate_filename = f"images/{timestamp}_plate_{text}.jpg"
                frame_filename = f"images/{timestamp}_frame_{text}.jpg"
                cv2.imwrite(plate_filename, plate_resized)
                cv2.imwrite(frame_filename, frame)

                print(f"Biển số: {text}")
                print(f"Ảnh biển lưu tại {plate_filename}, ảnh gốc lưu tại {frame_filename}")
                capture_image()

                last_detected[:] = frame

    return frame

# Hàm phát hiện biển số riêng biệt (gọi từ nút "Phát hiện biển số")
# - Sử dụng detect_license_plate để detect và lưu ảnh annotated/plate.
# - Cập nhật hiển thị và thông báo.
def detect_plate():
    if not current_image_path:
        messagebox.showwarning("Cảnh báo", "Hãy chụp hoặc tải lên một ảnh trước!")
        return

    try:
        status_label.config(text="🔄 Đang phát hiện biển số...", fg="#F39C12")
        root.update()

        annotated_image, plate_image = detect_license_plate(current_image_path)

        if annotated_image is not None:
            annotated_img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            annotated_img.thumbnail((400, 300))
            annotated_display = ImageTk.PhotoImage(annotated_img)
            image_label.config(image=annotated_display)
            image_label.image = annotated_display

            annotated_path = current_image_path.replace(".jpg", "_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)

            if plate_image is not None:
                plate_path = current_image_path.replace(".jpg", "_plate.jpg")
                cv2.imwrite(plate_path, plate_image)

                status_label.config(text="✅ Đã phát hiện và khoanh vùng biển số!", fg="#27AE60")
                messagebox.showinfo("Thành công",
                                    f"Đã phát hiện biển số và lưu ảnh:\n- Ảnh gốc: {current_image_path}\n- Ảnh khoanh vùng: {annotated_path}\n- Ảnh biển số: {plate_path}")
            else:
                status_label.config(text="⚠️ Đã khoanh vùng nhưng không cắt được biển số", fg="#E67E22")
        else:
            status_label.config(text="❌ Không phát hiện được biển số", fg="#E74C3C")
            messagebox.showwarning("Cảnh báo", "Không thể phát hiện biển số trong ảnh này.")

    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể phát hiện biển số: {e}")
        status_label.config(text="❌ Lỗi xử lý", fg="#E74C3C")

# ======================================
# Quy trình 2: Nhận diện ký tự (Character Recognition - OCR)
# ======================================

# Hàm nhận dạng văn bản từ ảnh (sử dụng EasyOCR)
# Thuật toán:
def recognize_text():
    if not current_image_path:
        messagebox.showwarning("Cảnh báo", "Hãy chụp hoặc tải lên một ảnh trước!")
        return

    try:
        # Cập nhật trạng thái
        status_label.config(text="🔄 Đang xử lý...", fg="#F39C12")
        root.update()

        # Phát hiện và khoanh vùng biển số
        annotated_image, plate_image = detect_license_plate(current_image_path)

        if annotated_image is not None:
            # Lưu ảnh đã khoanh vùng
            annotated_path = current_image_path.replace(".jpg", "_annotated.jpg")
            cv2.imwrite(annotated_path, annotated_image)

            # Hiển thị ảnh đã khoanh vùng
            annotated_img = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
            annotated_img.thumbnail((400, 300))
            annotated_display = ImageTk.PhotoImage(annotated_img)
            image_label.config(image=annotated_display)
            image_label.image = annotated_display

            # Nhận dạng văn bản từ biển số
            if plate_image is not None:
                plate_path = current_image_path.replace(".jpg", "_plate.jpg")
                cv2.imwrite(plate_path, plate_image)

                result = reader.readtext(plate_path, detail=0)
            else:
                result = reader.readtext(current_image_path, detail=0)
        else:
            result = reader.readtext(current_image_path, detail=0)

        text_result.delete(1.0, tk.END)

        if result:
            recognized_text = "\n".join(result)

            # Hiển thị kết quả với định dạng đẹp hơn
            text_result.tag_configure("big", font=("Arial", 16, "bold"), foreground="#FFFFFF")
            text_result.tag_configure("plate", font=("Arial", 20, "bold"), foreground="#FFD700",
                                      background="#2C3E50", justify="center")

            # Xóa nội dung cũ và chèn kết quả mới
            text_result.delete(1.0, tk.END)

            # Thêm tiêu đề
            text_result.insert(tk.END, "BIỂN SỐ NHẬN DIỆN ĐƯỢC:\n\n", "big")

            # Thêm kết quả chính với định dạng đặc biệt
            text_result.insert(tk.END, f" {recognized_text} \n", "plate")

            # Thêm phân tách
            text_result.insert(tk.END, "\n" + "═" * 50 + "\n\n", "big")

            # Thêm thông tin phụ
            text_result.insert(tk.END, f"Thời gian: {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}\n", "big")
            text_result.insert(tk.END, f"Ảnh: {os.path.basename(current_image_path)}", "big")

            status_label.config(text="✅ Hoàn thành!", fg="#27AE60")
        else:
            text_result.insert(tk.END, "Không nhận diện được văn bản nào")
            status_label.config(text="❌ Không tìm thấy văn bản", fg="#E74C3C")

    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể nhận dạng văn bản: {e}")
        status_label.config(text="❌ Lỗi xử lý", fg="#E74C3C")

# ======================================
# Các hàm hỗ trợ: Xử lý Webcam, Chụp ảnh, Tải ảnh, GUI
# ======================================

# Hàm xử lý webcam (chạy trong thread riêng)
# - Mở camera, đọc frame liên tục.
# - Hiển thị frame lên label.
# - Gọi process_frame để detect và OCR trong realtime.
def process_webcam():
    global running, webcam_frame
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không thể mở webcam.")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        webcam_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_img = Image.fromarray(webcam_frame)
        frame_img.thumbnail((400, 300))
        frame_display = ImageTk.PhotoImage(frame_img)
        webcam_label.config(image=frame_display)
        webcam_label.image = frame_display
        processed_frame = process_frame(frame, last_detected)

    cap.release()

# Hàm chụp ảnh từ webcam
# - Lưu ảnh vào thư mục captured_images với tên sequential.
# - Hiển thị ảnh chụp lên label.
# - Gọi recognize_text để OCR ngay.
def capture_image():
    global webcam_frame, current_image_path
    if webcam_frame is None:
        messagebox.showwarning("Cảnh báo", "Webcam chưa sẵn sàng!")
        return

    try:
        os.makedirs("captured_images", exist_ok=True)
        file_path = f"captured_images/captured_{len(os.listdir('captured_images')) + 1}.jpg"
        Image.fromarray(webcam_frame).save(file_path)

        img = Image.open(file_path)
        img.thumbnail((400, 300))
        img_display = ImageTk.PhotoImage(img)
        image_label.config(image=img_display)
        image_label.image = img_display

        current_image_path = file_path
        recognize_text()
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể chụp ảnh: {e}")

# Hàm tải ảnh từ tệp
# - Mở dialog chọn file ảnh.
# - Hiển thị ảnh lên label và cập nhật current_image_path.
def upload_image():
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[("Tất cả ảnh", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                   ("JPEG", "*.jpg *.jpeg"),
                   ("PNG", "*.png")]
    )
    if file_path:
        try:
            img = Image.open(file_path)
            img.thumbnail((400, 300))
            img_display = ImageTk.PhotoImage(img)
            image_label.config(image=img_display)
            image_label.image = img_display
            global current_image_path
            current_image_path = file_path
            status_label.config(text="📁 Đã tải ảnh", fg="#3498DB")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể mở ảnh: {e}")

# Hàm xóa kết quả
# - Xóa text result, image label, reset status.
def clear_results():
    text_result.delete(1.0, tk.END)
    image_label.config(image="")
    image_label.text = "Chưa có ảnh nào"
    status_label.config(text="🔄 Sẵn sàng", fg="#7F8C8D")

# Hàm xử lý thoát chương trình
# - Dừng running và destroy root nếu xác nhận.
def on_closing():
    global running
    running = False
    if messagebox.askokcancel("Thoát", "Bạn có muốn thoát ứng dụng?"):
        root.destroy()

# Hàm tạo giao diện GUI
# - Tạo window chính, các frame cho header, webcam, image, controls, results.
# - Định nghĩa styles cho button, label.
# - Gán command cho các button: upload, capture, detect, recognize, clear.
def create_gui():
    global root, webcam_label, image_label, text_result, status_label

    root = tk.Tk()
    root.title("🚗 Nhận Diện Biển Số Xe - License Plate Recognition")
    root.geometry("1200x800")
    root.configure(bg="#2C3E50")

    # Tạo style tùy chỉnh
    style = ttk.Style()
    style.theme_use('clam')

    # Cấu hình style
    style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"),
                    background="#2C3E50", foreground="#ECF0F1")
    style.configure("Subtitle.TLabel", font=("Segoe UI", 12),
                    background="#34495E", foreground="#BDC3C7")
    style.configure("Modern.TButton", font=("Segoe UI", 10, "bold"),
                    padding=(15, 8))
    style.configure("Card.TFrame", background="#34495E", relief="solid", borderwidth=1)

    # Header
    header_frame = tk.Frame(root, bg="#34495E", height=80)
    header_frame.pack(fill=tk.X, padx=10, pady=(10, 0))
    header_frame.pack_propagate(False)

    title_label = tk.Label(header_frame, text="🚗 HỆ THỐNG NHẬN DIỆN BIỂN SỐ XE",
                           font=("Segoe UI", 20, "bold"),
                           bg="#34495E", fg="#ECF0F1")
    title_label.pack(pady=15)

    # Main container
    main_container = tk.Frame(root, bg="#2C3E50")
    main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Top section - Webcam và Captured Image
    top_section = tk.Frame(main_container, bg="#2C3E50")
    top_section.pack(fill=tk.BOTH, expand=True)

    # Webcam frame
    webcam_frame_container = tk.Frame(top_section, bg="#34495E", relief="raised", bd=2)
    webcam_frame_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

    webcam_header = tk.Frame(webcam_frame_container, bg="#E74C3C", height=40)
    webcam_header.pack(fill=tk.X)
    webcam_header.pack_propagate(False)

    tk.Label(webcam_header, text="📹 CAMERA TRỰC TIẾP",
             font=("Segoe UI", 12, "bold"),
             bg="#E74C3C", fg="white").pack(pady=8)

    webcam_label = tk.Label(webcam_frame_container, bg="#2C3E50",
                            text="Camera đang khởi động...",
                            font=("Segoe UI", 12), fg="#BDC3C7")
    webcam_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Captured image frame
    image_frame_container = tk.Frame(top_section, bg="#34495E", relief="raised", bd=2)
    image_frame_container.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

    image_header = tk.Frame(image_frame_container, bg="#3498DB", height=40)
    image_header.pack(fill=tk.X)
    image_header.pack_propagate(False)

    tk.Label(image_header, text="📸 ẢNH ĐÃ CHỤP / BIỂN SỐ PHÁT HIỆN",
             font=("Segoe UI", 12, "bold"),
             bg="#3498DB", fg="white").pack(pady=8)

    image_label = tk.Label(image_frame_container, bg="#2C3E50",
                           text="Chưa có ảnh nào",
                           font=("Segoe UI", 12), fg="#BDC3C7")
    image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    # Middle section - Controls
    middle_section = tk.Frame(main_container, bg="#2C3E50", height=80)
    middle_section.pack(fill=tk.X, pady=10)
    middle_section.pack_propagate(False)

    # Control buttons
    control_frame = tk.Frame(middle_section, bg="#34495E", relief="raised", bd=2)
    control_frame.pack(fill=tk.BOTH, expand=True)

    tk.Label(control_frame, text="ĐIỀU KHIỂN",
             font=("Segoe UI", 10, "bold"),
             bg="#34495E", fg="#ECF0F1").pack(pady=5)

    button_container = tk.Frame(control_frame, bg="#34495E")
    button_container.pack(expand=True, pady=5)

    # Các nút chức năng với màu sắc khác nhau
    upload_btn = tk.Button(button_container, text="📁 Tải ảnh",
                           command=upload_image,
                           font=("Segoe UI", 9, "bold"),
                           bg="#9B59B6", fg="white",
                           relief="flat", padx=15, pady=5,
                           cursor="hand2")
    upload_btn.pack(side=tk.LEFT, padx=5)

    capture_btn = tk.Button(button_container, text="📸 Chụp",
                            command=capture_image,
                            font=("Segoe UI", 9, "bold"),
                            bg="#E67E22", fg="white",
                            relief="flat", padx=15, pady=5,
                            cursor="hand2")
    capture_btn.pack(side=tk.LEFT, padx=5)

    detect_btn = tk.Button(button_container, text="🔍 Phát hiện biển số",
                           command=detect_plate,
                           font=("Segoe UI", 9, "bold"),
                           bg="#F1C40F", fg="white",
                           relief="flat", padx=15, pady=5,
                           cursor="hand2")
    detect_btn.pack(side=tk.LEFT, padx=5)

    recognize_btn = tk.Button(button_container, text="📝 Nhận diện văn bản",
                              command=recognize_text,
                              font=("Segoe UI", 9, "bold"),
                              bg="#2ECC71", fg="white",
                              relief="flat", padx=15, pady=5,
                              cursor="hand2")
    recognize_btn.pack(side=tk.LEFT, padx=5)

    clear_btn = tk.Button(button_container, text="🗑️ Xóa",
                          command=clear_results,
                          font=("Segoe UI", 9, "bold"),
                          bg="#95A5A6", fg="white",
                          relief="flat", padx=15, pady=5,
                          cursor="hand2")
    clear_btn.pack(side=tk.LEFT, padx=5)

    # Bottom section - Results
    bottom_section = tk.Frame(main_container, bg="#34495E", relief="raised", bd=2)
    bottom_section.pack(fill=tk.BOTH, expand=True)

    result_header = tk.Frame(bottom_section, bg="#F39C12", height=40)
    result_header.pack(fill=tk.X)
    result_header.pack_propagate(False)

    tk.Label(result_header, text="📋 KẾT QUẢ NHẬN DIỆN",
             font=("Segoe UI", 12, "bold"),
             bg="#F39C12", fg="white").pack(pady=8)

    # Text result với scrollbar
    text_frame = tk.Frame(bottom_section, bg="#34495E")
    text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    text_result = tk.Text(text_frame, wrap=tk.WORD,
                          font=("Arial", 12),
                          bg="#2C3E50", fg="#ECF0F1",
                          insertbackground="#ECF0F1",
                          relief="flat", bd=0, padx=10, pady=10)

    scrollbar = tk.Scrollbar(text_frame, orient="vertical", command=text_result.yview)
    text_result.configure(yscrollcommand=scrollbar.set)

    text_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # Status bar
    status_frame = tk.Frame(root, bg="#34495E", height=30)
    status_frame.pack(fill=tk.X, side=tk.BOTTOM)
    status_frame.pack_propagate(False)

    status_label = tk.Label(status_frame, text="🔄 Sẵn sàng",
                            font=("Segoe UI", 9),
                            bg="#34495E", fg="#7F8C8D")
    status_label.pack(side=tk.LEFT, padx=10, pady=5)

    exit_btn = tk.Button(status_frame, text="❌ Thoát",
                         command=on_closing,
                         font=("Segoe UI", 9),
                         bg="#E74C3C", fg="white",
                         relief="flat", padx=10, pady=2,
                         cursor="hand2")
    exit_btn.pack(side=tk.RIGHT, padx=10, pady=2)

# Chạy ứng dụng
if __name__ == "__main__":
    create_gui()
    webcam_thread = Thread(target=process_webcam)
    webcam_thread.daemon = True
    webcam_thread.start()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()