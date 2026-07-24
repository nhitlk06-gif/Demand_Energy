Đây là kết quả CHẠY THẬT của `scripts/run_full_diagnostics.py` trong môi trường sandbox
(không có mạng, không có xgboost/lightgbm/catboost). Ba việc **đã chạy thật** và có số liệu
thật trong thư mục này:

- `five_model_comparison.csv`: HistGradientBoosting, ExtraTrees, MLP (thiếu CatBoost,
  LightGBM, XGBoost vì thiếu thư viện).
- `dm_tests.csv`: kiểm định Diebold-Mariano thật giữa 3 model trên + 2 SNaive.
- `recursive_horizon_48step.csv`: dự báo đệ quy 48 bước thật, backtest trên 20 điểm
  xuất phát ngẫu nhiên trong 2025, so với ground truth thật. MAPE theo h: ~2.6% ở h=1,
  tăng lên ~7.2% quanh h=12, rồi dao động quanh 4.4-4.6% tới h=48.

Chưa chạy được (cần bạn chạy lại toàn bộ script sau khi cài xgboost/lightgbm/catboost):
- `three_config_comparison.csv` (cần LightGBM)
- `quantile_coverage.json` (cần LightGBM)
- `hypothesis4_reread.csv` (cần XGBoost)
- Cột CatBoost/LightGBM/XGBoost trong `five_model_comparison.csv`
- DM test có các model đó

Chạy `python scripts/run_full_diagnostics.py` lại (đè lên các file trong thư mục này) sau
khi cài đủ thư viện để có bộ số liệu đầy đủ.
