# App: Backend FastAPI + Dashboard Streamlit

Hai dịch vụ tách rời cho Hệ Thống Dự Báo Nhu Cầu Tiêu Thụ Điện Quốc Gia
(Great Britain, dữ liệu NESO). `streamlit_app.py` không bao giờ tự tải model
trực tiếp; nó chỉ gọi `api.py` qua REST HTTP.

## Cài đặt

```bash
cd app
pip install -r requirements.txt
```

## Chạy cả 2 service (2 cửa sổ terminal)

Terminal 1 — backend:
```bash
cd app
uvicorn api:app --reload --port 8000
```

Terminal 2 — frontend:
```bash
cd app
streamlit run streamlit_app.py
```

Sau đó mở đường link Streamlit in ra ở Terminal 2 (mặc định
`http://localhost:8501`). Dashboard tự động gọi API tại địa chỉ cố định
`http://127.0.0.1:8000`; giao diện không còn ô nhập API URL để cấu hình.

## Cách chạy gộp 1 dòng (API chạy nền + dashboard chạy chính)

```bash
cd app
uvicorn api:app --port 8000 &
streamlit run streamlit_app.py
```

## Triển khai trên Streamlit Community Cloud (nền tảng chỉ chạy 1 service)

Streamlit Community Cloud chỉ chạy được đúng 1 file khởi động
(`streamlit_app.py`) cho mỗi app — nó không tự chạy thêm tiến trình
`uvicorn api:app` cho bạn. Để vẫn triển khai được như một khối duy nhất trên
nền tảng đó, `streamlit_app.py` sẽ **tự động khởi động `api.py` làm tiến
trình nền ngay trong cùng container**, ngay lần đầu phát hiện
`http://127.0.0.1:8000` chưa có gì lắng nghe. Hai file vẫn hoàn toàn tách rời
(2 tiến trình riêng, code riêng, chỉ nói chuyện qua HTTP) — đây chỉ là một
tiện ích để bạn bấm "Deploy" một lần là chạy được ngay.

Danh sách kiểm tra khi triển khai trên Streamlit Community Cloud:
1. Đặt "Main file path" của app là `app/streamlit_app.py`.
2. Đảm bảo Streamlit Cloud đọc được `app/requirements.txt` (mặc định nó tìm
   file `requirements.txt` ngay cạnh file khởi động).
3. Không cần cấu hình gì thêm: app tự gọi `http://127.0.0.1:8000` và tự khởi
   động backend ngay lần tải đầu tiên; việc này có thể mất vài giây trong
   lúc tiến trình API khởi động.
4. Thư mục `data/` và `models/` của repo (khoảng 190 MB gộp lại) phải được
   commit lên thì tiến trình API tự khởi động mới nạp được pipeline đã huấn
   luyện thật, thay vì rơi về bộ mô phỏng tổng hợp; trên các gói miễn phí có
   giới hạn tài nguyên, việc này có thể làm build/khởi động chậm hơn.

Nếu bạn muốn chạy backend FastAPI như một dịch vụ host riêng (Render,
Railway, Fly.io, v.v.), chỉ cần sửa hằng số `DEFAULT_API_URL` ở đầu file
`streamlit_app.py` thành địa chỉ public của dịch vụ đó thay vì `127.0.0.1`
— cơ chế tự khởi động chỉ kích hoạt với địa chỉ local, sẽ tự động bỏ qua với
mọi địa chỉ từ xa.

## Ghi chú

- Nếu các file model đã huấn luyện dưới `../models/*.pkl` và package
  `demandforecast` (`../src/demandforecast`) import được, `api.py` sẽ phục
  vụ dự báo thật từ pipeline nghiên cứu mô tả trong `../paper/main.tex`,
  tính trực tiếp từ chính dataset của bạn
  (`../data/processed/electricity_features_fixed.csv`). Nếu thiếu
  `catboost`/`xgboost`/`lightgbm` hoặc thiếu file model, nó tự động rơi về
  bộ mô phỏng nhu cầu điện tổng hợp (deterministic) để cả 2 service vẫn dùng
  được trọn vẹn — endpoint kiểm tra sức khỏe (`GET /`) và trạng thái kết nối
  trên dashboard đều báo rõ đang chạy ở chế độ nào.
- `GET /models` liệt kê mọi model đã huấn luyện thực sự có sẵn trên server
  (ví dụ CatBoost, LightGBM, ExtraTrees, Random Forest, MLP, ...). Sidebar
  Streamlit đọc danh sách này và cho phép chọn đúng 1 model cụ thể, hoặc để
  ở "Auto" để API tự chọn model có điểm số tốt nhất, đúng logic chọn model
  đã dùng trong pipeline nghiên cứu.
- `/predict/evaluate` được thiết kế cho đúng năm dữ liệu Test đã niêm phong
  (2025-01-01 đến 2025-12-31); bộ chọn ngày trên Streamlit tự giới hạn đúng
  khoảng này ở chế độ "Accuracy Evaluation (Backtesting)" để người dùng
  không chọn lọt vào giai đoạn Train.
- Cả 2 endpoint đều giới hạn tối đa 2.000 điểm dữ liệu mỗi lần gọi, để dung
  lượng phản hồi API và bảng dữ liệu trên Streamlit không phình to vô hạn
  khi chọn khoảng thời gian dài.
