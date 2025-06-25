import os
import subprocess
import sys

# Get port from environment variable, default to 3000
port = os.environ.get('PORT', '3000')

# Build the gunicorn command
cmd = [
    'gunicorn',
    '--bind', f'0.0.0.0:{port}',
    '-w', '4',
    'src/main:app'
]

print(f"Starting server on port {port}")
print(f"Command: {' '.join(cmd)}")

# Run gunicorn
try:
    subprocess.run(cmd)
except KeyboardInterrupt:
    print("Server stopped")
    sys.exit(0)
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)