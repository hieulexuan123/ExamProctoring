import cv2
import multiprocessing as mp
import signal
import sys
import time

def capture_frames(frame_queue, stop_event):
    """Capture frames from the webcam and put them in the queue."""
    cap = cv2.VideoCapture(0)
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    cv2.destroyAllWindows()
    frame_queue.cancel_join_thread()
    

def display_frames(frame_queue, stop_event):
    """Display frames from the queue."""
    while not stop_event.is_set() or not frame_queue.empty():
        if not frame_queue.empty():
            frame = frame_queue.get()
            cv2.imshow('Webcam Feed', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
    cv2.destroyAllWindows()

def signal_handler(sig, frame):
    """Handle signals to stop the processes."""
    print('Signal received, stopping...')
    stop_event.set()

def main():
    global stop_event
    stop_event = mp.Event()  # Create an Event for signaling stop
    frame_queue = mp.Queue()  # Queue to hold frames

    print(mp.cpu_count())

    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)

    # Create and start processes
    capture_process = mp.Process(target=capture_frames, args=(frame_queue, stop_event))
    display_process = mp.Process(target=display_frames, args=(frame_queue, stop_event))

    capture_process.start()
    display_process.start()
    
    # time.sleep(20)
    # stop_event.set()
    # Wait for processes to finish
    display_process.join()
    capture_process.join()

if __name__ == "__main__":
    main()