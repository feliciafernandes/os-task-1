# OS Lab Assignment 1: Concurrency and Multithreading

Implementation of the Producer-Consumer synchronization problem in Java and multithreaded Matrix Multiplication in Python where every multiplication operation is executed on a thread.

---

## Files in this Project

| File | Description |
| :--- | :--- |
| `ProducerConsumer.java` | Thread-safe circular bounded buffer demonstrating monitor synchronization (`wait`/`notifyAll`). |
| `matrix_multiplication.py` | Python implementation of 100x100 matrix multiplication using `ThreadPoolExecutor` (10,000 tasks) with animation generator. |
| `matrix_multiplication_100x100.mp4` | High-definition MP4 video animation showing parallel thread distribution across cells. |
| `matrix_multiplication.gif` | High-definition animated GIF showing thread activity and cell resolution in real time. |
| `README.md` | Detailed setup and execution guide. |

---

## 1. Producer-Consumer Synchronization (`ProducerConsumer.java`)

### Overview
- Implemented in Java using `CircularStorage`, maintaining a circular queue of capacity 4.
- Uses synchronized monitor blocks with `wait()` and `notifyAll()` to coordinate between producer and consumer threads.
- Processes 12 consecutive items across independent producer and consumer threads.

### Compiling & Running
```bash
javac ProducerConsumer.java
java ProducerConsumer
```

---

## 2. Multithreaded Matrix Multiplication (`matrix_multiplication.py`)

### Overview
- Two $100 \times 100$ matrices $A$ and $B$ multiplied to produce $C = A \times B$.
- For a $100 \times 100$ matrix, there are **10,000 multiplication operations** ($C[r][c]$).
- Uses Python's `ThreadPoolExecutor` to execute all 10,000 cell tasks concurrently as independent threads.
- Renders real-time visual animations showing row/column scan lines and completed cells.

### Running Animation Generator
```bash
python3 matrix_multiplication.py
```

Outputs generated:
- `matrix_multiplication_100x100.mp4`
- `matrix_multiplication.gif`

---

## 3. Note on TensorRT / TensorFlow

- **CPU Multithreading** (our code): Dispatches matrix operations across concurrent threads using a thread pool.
- **GPU Acceleration** (NVIDIA TensorRT / TensorFlow): Offloads computations to thousands of GPU CUDA cores and hardware Tensor Cores for massive matrix calculations.
