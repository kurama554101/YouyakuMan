import time


class Timer:
    def __init__(self):
        self.p_time_dict = {}
    
    def time_deco(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            p_time = end_time - start_time
            self.p_time_dict[func.__name__] = p_time
            
            return result
        return wrapper
