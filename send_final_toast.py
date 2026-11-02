import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

toast_code = """
import marimo as mo
mo.status.toast(
    title="Đã triển khai hoàn tất CED-SingleMoE-SplitFFN! 🚀🎯",
    description="Toàn bộ kiến trúc, bảng phân bổ 6.62M tham số, tiến trình tqdm và bảng 9 chỉ số trên 3,064 clips đã sẵn sàng!",
    kind="success"
)
"""

execute_remote(toast_code)
