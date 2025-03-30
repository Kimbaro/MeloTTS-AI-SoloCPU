import os
import tensorflow as tf
import tensorboard

# 로그 파일이 있는 디렉토리 설정
log_dir = "logs/"  # 실제 로그 파일이 있는 경로로 변경

# TensorBoard 서버 실행
os.system(f"tensorboard --logdir={log_dir} --port=6006")
