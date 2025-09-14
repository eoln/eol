#!/usr/bin/env python3
"""Sample Python file for testing indexing."""


def hello_world():
    """Print a greeting to the world."""
    print("Hello, World!")
    return "Hello, World!"


class TestClass:
    """A simple test class."""

    def __init__(self, name: str):
        self.name = name

    def greet(self):
        """Return a personalized greeting."""
        return f"Hello, {self.name}!"


if __name__ == "__main__":
    hello_world()
    test = TestClass("Test")
    print(test.greet())
