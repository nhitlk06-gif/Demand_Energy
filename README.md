# Dự Báo Nhu Cầu Tiêu Thụ Điện Quốc Gia — Great Britain (Dữ liệu NESO)

Dự án dự báo **National Demand (ND, MW)** cho hệ thống truyền tải điện của
**Great Britain** (Anh, Scotland, Wales), dùng dữ liệu quyết toán nửa giờ
(half-hourly settlement period) do **NESO** (National Energy System
Operator) công bố công khai, trải dài 6 năm (2020–2025).

Repo này gồm 3 phần gắn kết với nhau:
1. **Bài báo khoa học** (`paper/`) viết theo chuẩn Springer LNCS, trình bày
   đầy đủ phương pháp luận, phát hiện chính và các thí nghiệm kiểm chứng.
2. **Pipeline nghiên cứu tái lập được** (`notebooks/` + `src/demandforecast/`)
   — từ làm sạch dữ liệu thô đến huấn luyện, đánh giá, và các thí nghiệm
   giả thuyết dùng trong bài báo.
3. **Ứng dụng minh hoạ** (`app/`) backend FastAPI + dashboard Streamlit,
   dự báo trực tiếp từ chính pipeline/model đã huấn luyện ở trên.

> Đây là dự án nghiên cứu/giáo dục, **không phải** dự báo chính thức của
> NESO và không được dùng cho vận hành lưới điện thật, giao dịch, hay các
> quyết định an toàn-nguy hại. Xem mục Miễn Trừ Trách Nhiệm ở cuối file.

## Phát hiện chính (trên tập Test 2025, chưa từng dùng để huấn luyện)

> **Bài học cốt lõi của dự án**: một hệ số xác định ($R^2$) cao bất thường
> hay một mức cải thiện lớn so với đường cơ sở ngây thơ (naive baseline)
> luôn cần được đối chiếu với khoảng cách thời gian thật sự giữa dữ liệu mà
> mô hình được phép biết và thời điểm nó phải dự báo. Pipeline này từng tự
> phát hiện đúng lỗi "rò rỉ chân trời dự báo" (forecast-horizon leak) của
> chính mình **2 lần liên tiếp** — chi tiết đầy đủ trong `paper/main.tex`,
> mục Discussion.

| Model | R² | MAE (MW) | RMSE (MW) | MAPE (%) | Chân trời gần nhất |
|---|---|---|---|---|---|
| CatBoost Regressor | 0.9272 | 1192.4 | 1665.2 | 4.90 | t−12 (6 giờ) |
| LightGBM Regressor | 0.9229 | 1230.8 | 1714.0 | 5.01 | t−12 (6 giờ) |
| ExtraTrees Regressor | 0.9210 | 1255.4 | 1735.5 | 5.09 | t−12 (6 giờ) |
| HistGradientBoosting Regressor | 0.9197 | 1266.3 | 1749.0 | 5.13 | t−12 (6 giờ) |
| XGBoost Regressor | 0.9179 | 1273.1 | 1769.1 | 5.27 | t−12 (6 giờ) |
| MLP Regressor | 0.9178 | 1315.8 | 1770.3 | 5.33 | t−12 (6 giờ) |
| Random Forest Regressor | 0.9107 | 1332.4 | 1844.7 | 5.41 | t−12 (6 giờ) |
| Linear Regression | 0.9060 | 1403.1 | 1892.4 | 5.57 | t−12 (6 giờ) |
| SNaive Daily | 0.8257 | 1849.7 | 2577.2 | 7.37 | t−48 (24 giờ) |
| SNaive Weekly | 0.7764 | 2177.7 | 2918.9 | 8.44 | t−336 (7 ngày) |

CatBoost thắng mọi chỉ số so với mọi model khác trên đúng cùng chân trời 6
giờ, cắt giảm RMSE khoảng 35,4% so với đường cơ sở ngây thơ mạnh hơn
(SNaive Daily). Số liệu đầy đủ, kiểm định Diebold–Mariano từng cặp model,
và toàn bộ câu chuyện phương pháp luận nằm trong `paper/main.tex`.


## Cấu trúc thư mục

```
Demand-Energy/
├── paper/                  # Bài báo LaTeX (main.tex) + PDF đã compile
├── notebooks/               # 4 notebook nghiên cứu gốc (01 làm sạch, 02 EDA,
│                             # 03 đặc trưng, 04 huấn luyện/đánh giá/giả thuyết)
├── src/demandforecast/      # Package Python sẵn sàng production
│   ├── config.py            #   toàn bộ hằng số/đường dẫn/tham số model
│   ├── cleaning.py           #   làm sạch dữ liệu thô, đồng bộ giờ mùa hè
│   ├── features.py           #   lag, rolling, mã hoá tuần hoàn, lọc MI
│   ├── splits.py             #   chia tập train/valid/test theo thời gian
│   ├── models.py             #   định nghĩa các model (SNaive/Linear/...)
│   ├── metrics.py            #   MAE/RMSE/MAPE/R², MASE, kiểm định DM
│   ├── train.py               #   huấn luyện + tinh chỉnh + lưu model
│   ├── forecast.py            #   suy luận: dự báo đệ quy + backtest thật
│   ├── pipeline.py            #   điều phối cleaning -> features -> train
│   └── cli.py                 #   các lệnh console-script
├── scripts/                  # Script chạy nhanh từng bước của pipeline
├── app/                      # Backend FastAPI + Dashboard Streamlit
│   ├── api.py                  #   xem app/README.md để biết chi tiết endpoint
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── README.md               #   hướng dẫn chạy/triển khai app (tiếng Việt)
├── data/
│   ├── raw/                  # 6 file gốc: demanddata_2020.csv ... 2025.csv
│   ├── processed/             # dữ liệu đã làm sạch/tạo đặc trưng
│   └── external/               # dự báo chính thức của NESO (để đối chứng)
├── models/                   # model .pkl đã huấn luyện + các file chẩn đoán
├── figures/                   # hình minh hoạ dùng trong bài báo
├── docs/ARCHITECTURE.md        # sơ đồ kiến trúc pipeline chi tiết hơn
├── tests/                     # pytest cho cleaning/features/metrics/forecast/API
├── pyproject.toml, Makefile, requirements.txt
└── SUA_DOI_DA_LAM.md            # nhật ký đầy đủ mọi lượt sửa của dự án
```

## Bắt Đầu Nhanh

```bash
git clone <link-repo-cua-ban> Demand-Energy
cd Demand-Energy
python -m venv .venv && source .venv/bin/activate   # không bắt buộc nhưng nên làm
pip install -e ".[dev,app,viz]"
```

### 1. Chạy lại toàn bộ pipeline (làm sạch → đặc trưng → huấn luyện)

Model đã huấn luyện sẵn có trong `models/`, nên bước này **không bắt buộc**
trừ khi bạn muốn huấn luyện lại từ đầu hoặc dữ liệu thô thay đổi:

```bash
make pipeline          # tham số đầy đủ, có tinh chỉnh (vài phút trên CPU)
make pipeline-fast      # tham số nhẹ hơn, ~30 giây, để kiểm tra nhanh
```

Hoặc chạy từng bước:

```bash
python scripts/run_cleaning.py     # data/raw/*.csv -> data/processed/electricity_cleaned.csv
python scripts/run_features.py     # -> data/processed/electricity_features_fixed.csv
python scripts/run_training.py     # -> models/*.pkl + models/metrics_summary.json
```

### 2. Chạy backend API dự báo

```bash
make api
# hoặc: uvicorn app.api:app --reload --port 8000
```

Mở tài liệu tương tác tại **http://localhost:8000/docs**. Các endpoint
chính (xem `app/README.md` để biết đầy đủ chi tiết):

| Method | Đường dẫn | Mô tả |
|---|---|---|
| GET | `/` | Kiểm tra tình trạng server, cho biết đang dùng model thật hay bộ mô phỏng |
| GET | `/models` | Liệt kê các model đã huấn luyện có sẵn, kèm model mặc định (tốt nhất) |
| POST | `/predict/future` | Dự báo về tương lai từ 1 ngày bắt đầu, theo số giờ mong muốn (`horizon_hours`) |
| POST | `/predict/evaluate` | So sánh dự báo với thực tế trên 1 khoảng ngày thuộc tập Test 2025, trả về MAPE/MAE/RMSE |

Nếu không tìm thấy file model đã huấn luyện, API tự động chuyển sang một bộ
mô phỏng nhu cầu điện tổng hợp (có đỉnh tải ban ngày, đáy tải ban đêm, chênh
lệch ngày thường/cuối tuần) để dịch vụ vẫn hoạt động được trọn vẹn.

### 3. Chạy dashboard

```bash
make dashboard
# hoặc: streamlit run app/streamlit_app.py
```

Dashboard tự động gọi API tại `http://127.0.0.1:8000`; nếu API chưa chạy,
`streamlit_app.py` sẽ tự khởi động `api.py` làm tiến trình nền (xem
`app/README.md` để biết chi tiết, kể cả cách triển khai trên Streamlit
Community Cloud).

### 4. Chạy kiểm thử

```bash
make test
# hoặc: pytest
```

## Tóm Tắt Phương Pháp Luận

**Làm sạch dữ liệu** (`cleaning.py`, từ `01_pre_eda_and_cleaning_v2.ipynb`)
- Gộp 6 file CSV theo năm (2020–2025), tổng 105.216 dòng dữ liệu nửa giờ.
- Đồng bộ lại 12 ngày chuyển múi giờ mùa hè (British Summer Time, ngày có
  46 hoặc 50 chu kỳ thay vì 48) về đúng lưới 48 chu kỳ chuẩn, bằng nội suy
  tuyến tính trên trục thời gian tương đối đã chuẩn hoá trong ngày.
- Đánh dấu các giá trị nhu cầu không dương là thiếu và nội suy tuyến tính
  toàn bộ khoảng trống còn lại.

**Xây dựng đặc trưng** (`features.py`, từ
`03_feature_engineering_and_selection_revised.ipynb`)
- Chỉ tạo lag tự hồi quy (autoregressive) cho riêng `ND`, ở 4 độ sâu ngắn
  hạn (6–24 giờ) cộng 1 lag tuần (7 ngày) — **cố tình bỏ mọi lag ngắn hơn 6
  giờ** để tránh biến bài toán dự báo thành một phép nowcast gần như tức
  thời (xem `docs/ARCHITECTURE.md` và mục Methodology trong `paper/main.tex`).
- Thống kê rolling mean/std, luôn dịch lùi đúng 12 chu kỳ (6 giờ) trước khi
  tính, khớp với chân trời tối thiểu áp dụng xuyên suốt toàn pipeline.
- 12 cột ngoại sinh đo tại đúng thời điểm cần dự báo (gió/mặt trời/bơm tích
  năng/9 luồng liên kết) bị loại hẳn vì không quan sát được trước 6 giờ;
  chỉ giữ lại 2 cột công suất lắp đặt vì là số liệu kế hoạch công bố trước.
- Mã hoá tuần hoàn sin/cos cho chu kỳ trong ngày và ngày trong tuần, cộng
  thêm cờ cuối tuần.
- Chọn đặc trưng bằng **thông tin tương hỗ (mutual information)** so với
  `ND`, tính riêng trên phần dữ liệu 2020–2024 (năm 2025 được niêm phong
  hoàn toàn để tránh rò rỉ ngay từ bước chọn đặc trưng).
- Ma trận cuối cùng: 104.880 dòng × 22 đặc trưng.

**Huấn luyện model** (`train.py`, từ
`04_model_training_and_evaluation_revised.ipynb`)
- Chia theo thời gian: train 2020–2023, valid 2024 (dùng để early-stopping
  và tinh chỉnh), test 2025 (niêm phong hoàn toàn, chỉ dùng đúng 1 lần để
  chấm điểm cuối cùng).
- 9 model được so sánh: 2 đường cơ sở SNaive, Linear Regression, Random
  Forest, ExtraTrees, HistGradientBoosting, XGBoost, LightGBM, CatBoost
  (2 model boosting được tinh chỉnh thêm bằng random search).

**Suy luận / dự báo** (`forecast.py`)
- `forecast_horizon()`: vì các model dùng đặc trưng tự hồi quy, dự báo vượt
  quá điểm dữ liệu cuối cùng đã quan sát được thực hiện **đệ quy**: mỗi dự
  báo 30 phút được đưa ngược lại vào lịch sử như thể là một quan sát thật,
  để tính lag/rolling cho bước kế tiếp.
- `backtest_predictions()`: dự báo một bước bình thường, trực tiếp trên các
  dòng lịch sử thật đã quan sát được (train/valid/test) — cách công bằng,
  không có vòng phản hồi, để vẽ đường dự báo so với thực tế.

## Nguồn Dữ Liệu

Dữ liệu nhu cầu lịch sử của NESO (`demanddata_2020.csv` …
`demanddata_2025.csv`), theo chu kỳ quyết toán nửa giờ, do nhà điều độ hệ
thống điện Great Britain công bố công khai. Đối chứng dự báo ngày-trước
chính thức lấy từ bộ dữ liệu công khai "Day Ahead Half Hourly Demand
Forecast Performance" của NESO (`data/external/neso_day_ahead_forecast.csv`).

## Miễn Trừ Trách Nhiệm

Đây là công cụ dự báo phục vụ nghiên cứu/giáo dục. Đây **không phải** dự
báo chính thức của NESO và không được dùng cho vận hành lưới điện thật,
giao dịch, hay các quyết định liên quan đến an toàn.

## Tài Liệu Chi Tiết Hơn

- `paper/main.tex` — bài báo đầy đủ: phương pháp luận, kết quả, thảo luận,
  giới hạn, và toàn bộ số liệu trích dẫn được.
- `app/README.md` — hướng dẫn cài đặt/chạy/triển khai backend + dashboard.
- `docs/ARCHITECTURE.md` — sơ đồ chi tiết luồng dữ liệu và lý do chọn chân
  trời tối thiểu 6 giờ (t−12).
- `SUA_DOI_DA_LAM.md` — nhật ký đầy đủ, theo từng đợt, mọi lỗi đã tìm ra và
  sửa trong suốt quá trình phát triển dự án (kể cả các lần rò rỉ dữ liệu đã
  phát hiện và khắc phục).
