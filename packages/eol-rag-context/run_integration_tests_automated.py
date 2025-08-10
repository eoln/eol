#!/usr/bin/env python3
"""
Automated integration test runner for EOL RAG Context.
Handles Redis lifecycle and test execution.
"""

import subprocess
import sys
import time
import os
import signal
import atexit
from pathlib import Path

class IntegrationTestRunner:
    def __init__(self):
        self.redis_process = None
        self.redis_container = None
        self.exit_code = 0
        
    def check_docker(self):
        """Check if Docker is available and running."""
        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def check_redis_native(self):
        """Check if Redis is installed locally."""
        try:
            result = subprocess.run(
                ["redis-server", "--version"],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start_redis_docker(self):
        """Start Redis using Docker."""
        print("Starting Redis with Docker...")
        
        # Stop any existing container
        subprocess.run(
            ["docker", "rm", "-f", "eol-test-redis"],
            capture_output=True,
            stderr=subprocess.DEVNULL
        )
        
        # Start Redis container
        result = subprocess.run(
            [
                "docker", "run", "-d",
                "--name", "eol-test-redis",
                "-p", "6379:6379",
                "redis/redis-stack:latest"
            ],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"Failed to start Redis container: {result.stderr}")
            return False
        
        self.redis_container = result.stdout.strip()
        
        # Wait for Redis to be ready
        for i in range(30):
            try:
                result = subprocess.run(
                    ["docker", "exec", "eol-test-redis", "redis-cli", "ping"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if "PONG" in result.stdout:
                    print("Redis is ready!")
                    return True
            except subprocess.SubprocessError:
                pass
            
            time.sleep(1)
            if i % 5 == 0:
                print(f"Waiting for Redis... ({i}/30)")
        
        print("Redis failed to start within timeout")
        return False
    
    def start_redis_native(self):
        """Start Redis natively."""
        print("Starting Redis natively...")
        
        # Create temp directory for Redis data
        redis_dir = Path("/tmp/eol-test-redis")
        redis_dir.mkdir(exist_ok=True)
        
        # Start Redis server
        self.redis_process = subprocess.Popen(
            [
                "redis-server",
                "--port", "6379",
                "--dir", str(redis_dir),
                "--save", "",  # Disable persistence for tests
                "--appendonly", "no"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for Redis to be ready
        for i in range(30):
            try:
                result = subprocess.run(
                    ["redis-cli", "ping"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if "PONG" in result.stdout:
                    print("Redis is ready!")
                    return True
            except subprocess.SubprocessError:
                pass
            
            time.sleep(0.5)
            if i % 10 == 0:
                print(f"Waiting for Redis... ({i}/30)")
        
        print("Redis failed to start within timeout")
        return False
    
    def stop_redis(self):
        """Stop Redis."""
        print("\nStopping Redis...")
        
        if self.redis_container:
            # Stop Docker container
            subprocess.run(
                ["docker", "rm", "-f", "eol-test-redis"],
                capture_output=True,
                stderr=subprocess.DEVNULL
            )
            print("Redis container stopped")
        
        if self.redis_process:
            # Stop native Redis
            self.redis_process.terminate()
            try:
                self.redis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.redis_process.kill()
                self.redis_process.wait()
            print("Redis process stopped")
    
    def install_dependencies(self):
        """Install required Python dependencies."""
        print("Checking Python dependencies...")
        
        required_packages = [
            "redis",
            "aioredis",
            "pytest",
            "pytest-asyncio",
            "pytest-cov",
            "numpy",
            "pydantic",
            "pydantic-settings",
            "aiofiles",
            "sentence-transformers",
            "beautifulsoup4",
            "markdown",
            "pyyaml",
            "networkx",
            "watchdog",
            "gitignore-parser"
        ]
        
        # Check which packages are missing
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing.append(package)
        
        if missing:
            print(f"Installing missing packages: {', '.join(missing)}")
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q"] + missing,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print(f"Warning: Some packages failed to install: {result.stderr}")
        else:
            print("All dependencies are installed")
    
    def run_tests(self):
        """Run the integration tests."""
        print("\n" + "="*60)
        print("Running Integration Tests")
        print("="*60 + "\n")
        
        # Set environment variables
        env = os.environ.copy()
        env["REDIS_HOST"] = "localhost"
        env["REDIS_PORT"] = "6379"
        env["PYTHONPATH"] = f"{Path.cwd()}/src:{env.get('PYTHONPATH', '')}"
        
        # Run pytest with coverage
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/integration/",
                "-xvs",
                "--cov=eol.rag_context",
                "--cov-report=term",
                "--cov-report=html:coverage/html",
                "--tb=short",
                "-m", "integration"
            ],
            env=env,
            capture_output=False,  # Show output in real-time
            text=True
        )
        
        self.exit_code = result.returncode
        
        # Also run combined unit + integration for total coverage
        print("\n" + "="*60)
        print("Combined Coverage Report (Unit + Integration)")
        print("="*60 + "\n")
        
        result = subprocess.run(
            [
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=eol.rag_context",
                "--cov-report=term:skip-covered",
                "--quiet",
                "--no-header"
            ],
            env=env,
            capture_output=True,
            text=True
        )
        
        # Extract and display coverage summary
        for line in result.stdout.split('\n'):
            if 'TOTAL' in line or 'Name' in line:
                print(line)
        
        return self.exit_code == 0
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_redis()
    
    def run(self):
        """Main execution flow."""
        print("="*60)
        print("EOL RAG Context - Automated Integration Tests")
        print("="*60 + "\n")
        
        # Register cleanup
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, lambda s, f: sys.exit(1))
        signal.signal(signal.SIGTERM, lambda s, f: sys.exit(1))
        
        # Install dependencies
        self.install_dependencies()
        
        # Start Redis
        redis_started = False
        
        if self.check_docker():
            print("Docker is available")
            redis_started = self.start_redis_docker()
        elif self.check_redis_native():
            print("Native Redis is available")
            redis_started = self.start_redis_native()
        else:
            print("Error: Neither Docker nor Redis is available")
            print("Please install either Docker or Redis to run integration tests")
            return 1
        
        if not redis_started:
            print("Failed to start Redis")
            return 1
        
        # Run tests
        success = self.run_tests()
        
        # Clean up
        self.cleanup()
        
        # Report results
        print("\n" + "="*60)
        if success:
            print("✅ All integration tests passed!")
        else:
            print(f"❌ Some tests failed (exit code: {self.exit_code})")
        print("="*60)
        
        return self.exit_code


if __name__ == "__main__":
    runner = IntegrationTestRunner()
    sys.exit(runner.run())