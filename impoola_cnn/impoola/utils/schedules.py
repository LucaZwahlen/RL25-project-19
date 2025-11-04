def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    if t >= duration:
        return end_e
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)
