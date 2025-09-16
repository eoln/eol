#!/usr/bin/env python3
"""Debug Redis import issue in tests."""

import os
import sys

# Set PYTHONPATH
sys.path.insert(0, "src")
os.environ["PYTHONPATH"] = "src:" + os.environ.get("PYTHONPATH", "")

print("=== Debug Redis Import ===")
print(f"Python: {sys.executable}")
print(f"Path: {sys.path[:3]}")

# Check what happens when we import
print("\n1. Direct import of redis:")
try:
    from redis import Redis
    from redis.asyncio import Redis as AsyncRedis

    print(f"   Redis type: {type(Redis)}")
    print(f"   AsyncRedis type: {type(AsyncRedis)}")
    print(f"   Is Mock? {'Mock' in str(type(AsyncRedis))}")
except ImportError as e:
    print(f"   Import failed: {e}")

print("\n2. Import redis_client module:")
from eol.rag_context import redis_client

print(f"   redis_client.Redis type: {type(redis_client.Redis)}")
print(f"   redis_client.AsyncRedis type: {type(redis_client.AsyncRedis)}")
print(f"   Is Mock? {'Mock' in str(type(redis_client.AsyncRedis))}")

print("\n3. Check if fallback was triggered:")
# Add debug to redis_client to see what happened
import importlib

import eol.rag_context.redis_client as rc

# Check the actual values
print(f"   Redis is MagicMock? {rc.Redis.__module__ == 'unittest.mock'}")
print(f"   AsyncRedis is MagicMock? {rc.AsyncRedis.__module__ == 'unittest.mock'}")

print("\n4. Try to reload without mocks:")
# Clear any mocks
if "redis" in sys.modules:
    del sys.modules["redis"]
if "redis.asyncio" in sys.modules:
    del sys.modules["redis.asyncio"]

# Reload
importlib.reload(rc)
print(f"   After reload - AsyncRedis type: {type(rc.AsyncRedis)}")
print(f"   After reload - Is Mock? {rc.AsyncRedis.__module__ == 'unittest.mock'}")
