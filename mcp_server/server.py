from fastmcp import FastMCP
import requests
import json
import os

SandboxURL = "http://0.0.0.0:8080/run_code"
mcp = FastMCP(name="SandboxMCPServer")

@mcp.tool
def run_code(code: str = "",
             language: str = "python",
             compile_timeout: float = 10,
             run_timeout: float = 10,
             stdin: str = '',
             files: dict = {},
             fetch_files: list = []
             ) -> str:
    """
    Run code in the sandbox service

    Args:
        code (str): The code to run
        language (str): The programming language of the code
        compile_timeout (float): The compile timeout for compiled languages. Defaults to 10.
        run_timeout (float): The code run timeout. Defaults to 10.
        stdin (str): The string to pass into stdin. Defaults to None.
        files (dict): A dict from file path to base64 encoded file content. Defaults to {}.
        fetch_files (list): A list of file paths to fetch after code execution. Defaults to [].

    Returns:
        dict: The response from the sandbox service
            status (str): The status of the code execution. One of 'Success', 'Failed', 'SandboxError'.
            message (str): The message from the sandbox service.
            compile_result (dict or None): The result of the compilation step, if applicable.
            run_result (dict or None): The result of the execution step.
            executor_pod_name (str or None): The name of the executor pod.
            files (dict): A dict from file path to base64 encoded file content that were fetched
    """

    payload = {
        "code": code,
        "language": language,
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "stdin": stdin,
        "files": files,
        "fetch_files": fetch_files
    }

    try:
        response = requests.post(SandboxURL, json=payload)
        return json.dumps(response.json(), indent=2)
    except Exception as e:
        return f"Error running code: {e}"

if __name__ == "__main__":
    SandboxURL = os.getenv("SANDBOX_URL", SandboxURL)

    mcp.run(transport="http", port="8080")