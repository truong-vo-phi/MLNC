from deploy_to_sandbox import execute_remote

code = """
import os
p = '/marimo/checkpoints/best_model.pt'
print('best_model.pt size:', os.path.getsize(p) / 1024 / 1024, 'MB')
"""
execute_remote(code)
