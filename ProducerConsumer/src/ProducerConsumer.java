class CircularStorage {
    private final int[] storage;
    private final int capacity;
    private int occupiedCount = 0;
    private int headIdx = 0;
    private int tailIdx = 0;

    public CircularStorage(int capacity) {
        this.capacity = capacity;
        this.storage = new int[capacity];
    }

    public String dumpQueue() {
        StringBuilder builder = new StringBuilder("[");
        for (int i = 0; i < occupiedCount; i++) {
            int pos = (tailIdx + i) % capacity;
            builder.append(storage[pos]);
            if (i < occupiedCount - 1) {
                builder.append(", ");
            }
        }
        builder.append("]");
        return builder.toString();
    }

    public synchronized void put(int item) {
        while (occupiedCount == capacity) {
            try {
                System.out.println("Buffer full, producer waiting...");
                wait();
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                return;
            }
        }

        storage[headIdx] = item;
        headIdx = (headIdx + 1) % capacity;
        occupiedCount++;

        System.out.println("Produced " + item + " -> " + dumpQueue());
        notifyAll();
    }

    public synchronized int get() {
        while (occupiedCount == 0) {
            try {
                System.out.println("Buffer empty, consumer waiting...");
                wait();
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                return -1;
            }
        }

        int item = storage[tailIdx];
        tailIdx = (tailIdx + 1) % capacity;
        occupiedCount--;

        System.out.println("Consumed " + item + " -> " + dumpQueue());
        notifyAll();
        return item;
    }
}

class DataProducer extends Thread {
    private final CircularStorage channel;
    private final int limit;
    private final int delayMs;

    public DataProducer(CircularStorage channel, int limit, int delayMs) {
        this.channel = channel;
        this.limit = limit;
        this.delayMs = delayMs;
    }

    @Override
    public void run() {
        for (int val = 1; val <= limit; val++) {
            channel.put(val);
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
}

class DataConsumer extends Thread {
    private final CircularStorage channel;
    private final int limit;
    private final int delayMs;

    public DataConsumer(CircularStorage channel, int limit, int delayMs) {
        this.channel = channel;
        this.limit = limit;
        this.delayMs = delayMs;
    }

    @Override
    public void run() {
        for (int i = 0; i < limit; i++) {
            channel.get();
            try {
                Thread.sleep(delayMs);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }
}

public class ProducerConsumer {
    public static void main(String[] args) {
        CircularStorage sharedPipe = new CircularStorage(4);

        DataProducer prod = new DataProducer(sharedPipe, 12, 180);
        DataConsumer cons = new DataConsumer(sharedPipe, 12, 550);

        cons.start();
        prod.start();

        try {
            prod.join();
            cons.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }
}
