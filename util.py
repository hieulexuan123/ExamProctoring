import time

class FPSCounter:
    def __init__(self):
        self.frame_count = 0
        self.start_time = time.time()

    def update(self):
        self.frame_count += 1

    def get_fps(self):
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            fps = self.frame_count / elapsed_time
        else:
            fps = 0
        return fps

    def reset(self):
        self.frame_count = 0
        self.start_time = time.time()