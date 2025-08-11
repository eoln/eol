"""Sample Python module for testing."""

def calculate_factorial(n: int) -> int:
    """Calculate factorial of n."""
    if n <= 1:
        return 1
    return n * calculate_factorial(n - 1)

class Calculator:
    """Simple calculator class."""
    
    def __init__(self):
        self.result = 0
    
    def add(self, a: float, b: float) -> float:
        """Add two numbers."""
        self.result = a + b
        return self.result
    
    def multiply(self, a: float, b: float) -> float:
        """Multiply two numbers."""
        self.result = a * b
        return self.result
