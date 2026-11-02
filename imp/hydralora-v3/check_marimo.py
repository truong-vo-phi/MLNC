from deploy_to_sandbox import execute_remote

code = """
import inspect
import marimo
# Let's see what routes exist in marimo server
print("marimo version:", marimo.__version__)
"""
execute_remote(code)
