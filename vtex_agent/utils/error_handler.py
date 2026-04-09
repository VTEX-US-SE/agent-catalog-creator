"""Error handling utilities with retry logic and exponential backoff."""
import time
from typing import Callable, Any, Optional
from functools import wraps


def retry_with_exponential_backoff(
    max_retries: int = 5,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0,
    retryable_errors: Optional[tuple] = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        backoff_factor: Multiplier for exponential backoff
        retryable_errors: Tuple of exception types to retry (default: all exceptions)
        
    Returns:
        Decorated function with retry logic
    """
    if retryable_errors is None:
        retryable_errors = (Exception,)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_exception = e
                    
                    # Check if it's a rate limit error (429)
                    error_str = str(e).lower()
                    is_rate_limit = (
                        "429" in error_str or
                        "rate limit" in error_str or
                        "quota" in error_str or
                        "too many requests" in error_str or
                        "resource exhausted" in error_str
                    )
                    
                    # Also check for HTTP status code if available
                    if hasattr(e, 'status_code') and e.status_code == 429:
                        is_rate_limit = True
                    
                    if not is_rate_limit and attempt == 0:
                        # Not a rate limit error, re-raise immediately on first attempt
                        raise
                    
                    if attempt < max_retries:
                        # Calculate delay with exponential backoff
                        wait_time = min(delay, max_delay)
                        print(f"     ⚠️  Error (attempt {attempt + 1}/{max_retries + 1}): {str(e)[:100]}")
                        if is_rate_limit:
                            print(f"     ⚠️  Rate limit detected. Retrying in {wait_time:.1f}s...")
                        time.sleep(wait_time)
                        delay *= backoff_factor
                    else:
                        # Max retries reached
                        print(f"     ❌ Max retries ({max_retries + 1}) reached. Last error: {str(e)[:200]}")
                        raise
            
            # Should never reach here, but just in case
            if last_exception:
                raise last_exception
                
        return wrapper
    return decorator


def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an error is a rate limit error.
    
    Args:
        error: Exception to check
        
    Returns:
        True if error is rate limit related
    """
    error_str = str(error).lower()
    return (
        "429" in error_str or
        "rate limit" in error_str or
        "quota" in error_str or
        "too many requests" in error_str or
        "resource exhausted" in error_str or
        (hasattr(error, 'status_code') and error.status_code == 429)
    )

