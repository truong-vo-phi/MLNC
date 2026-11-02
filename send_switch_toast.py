import sys
sys.path.insert(0, r"D:\PROJECT\MLNC\imp\kd-moe-ced")
from deploy_to_sandbox import execute_remote

toast_code = """
import marimo as mo
mo.status.toast(
    title="Đã tích hợp công tắc RUN_TRAINING = True! 🎛️🚀",
    description="Bạn có thể gạt công tắc tại Phần 5 để bật/tắt chế độ Huấn Luyện Thực Tế trực tiếp trên giao diện!",
    kind="success"
)
"""

execute_remote(toast_code)
