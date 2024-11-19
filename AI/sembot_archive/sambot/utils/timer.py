import time
from functools import wraps

def async_timeit(func):
    """비동기 함수 실행 시간을 측정하는 데코레이터"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        result = await func(*args, **kwargs)  # 비동기 함수 호출
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper

def timeit(func):
    """함수 실행 시간을 측정하는 데코레이터"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 실행 시간: {end_time - start_time:.2f}초")
        return result
    return wrapper