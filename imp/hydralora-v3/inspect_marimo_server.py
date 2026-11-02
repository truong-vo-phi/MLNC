from deploy_to_sandbox import execute_remote

code = """
import marimo
import sys
# Let's inspect marimo._server
try:
    from marimo._server.api import status
    print("Found marimo._server.api")
except Exception as e:
    print("Error:", e)
"""
execute_remote(code)
