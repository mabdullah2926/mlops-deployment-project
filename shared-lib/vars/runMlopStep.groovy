def call(String stageName, String scriptPath) {
    echo "=== Running MLOps Stage: ${stageName} ==="
    sh """
        . .venv/bin/activate
        python ${scriptPath}
    """
}
