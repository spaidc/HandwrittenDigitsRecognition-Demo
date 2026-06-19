#### VI. Triển khai & Đóng gói (Deployment)
Đưa mô hình vào ứng dụng thực tế để demo tương tác.
- Giao diện người dùng: Xây dựng Web App bằng Streamlit.
- Tính năng ứng dụng:
- Cho phép người dùng upload ảnh hoặc vẽ trực tiếp số lên canvas.
- App trả về kết quả dự đoán (Prediction vs Actual).
- Tích hợp toggle để bật/tắt lớp phủ Grad-CAM/Saliency map ngay trên ứng dụng, giúp trực quan hóa "suy nghĩ" của model theo thời gian thực.

#### Idea: 
Tìm cách đóng gói weight của mô hình / mô hình để xử lý và đưa ra output (độ trễ thấp)
+ Đưa ảnh vào xử lý 
	+ Có hai hướng là real time, không cần kích hoạt thủ công
	+ Hoặc là kích hoạt thủ công (ấn nút)
+ Triển khai kích hoạt thủ công:
	+ Canvas vẽ bằng pointer
	+ Upload ảnh và nhận diện (có hiển thị trên UI sau khi upload)
	+ Lấy trực tiếp từ camera, sau khi ấn nút thì chụp lại để giữ ảnh và nhận diện