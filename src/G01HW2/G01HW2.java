// Draft

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;
import java.util.*;
import java.util.concurrent.Semaphore;


public class G01HW2
{

    public static void main(String[] args) throws Exception
    {
        if (args.length != 7)
            throw new IllegalArgumentException("USAGE: n, phi, epsilon, delta, d, w, portExp");

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


        int n = Integer.parseInt(args[0]);
        System.out.println("Batch size = " + n);
        int phi = Integer.parseInt(args[1]);
        System.out.println("Frequency threshold = " + phi);
        float epsilon = Float.parseFloat(args[2]);
        System.out.println("Accuracy parameter = " + epsilon);
        float delta = Float.parseFloat(args[3]);
        System.out.println("Confidence parameter = " + delta);
        int d = Integer.parseInt(args[4]);
        System.out.println("Batch size = " + d);
        int w = Integer.parseInt(args[5]);
        System.out.println("Frequency threshold = " + w);
        int portExp = Integer.parseInt(args[6]);
        System.out.println("Receiving data from port = " + portExp);

        if (epsilon <= 0 || epsilon >= phi)
            throw new IllegalArgumentException("Accuracy must be 0 < epsilon < phi");

        if (delta <= 0 || delta >= 1)
            throw new IllegalArgumentException("Accuracy must be 0 < delta < 1");

        // IMPORTANT: the master must be set to "local[*]" or "local[n]" with n > 1, otherwise
        // there will be no processor running the streaming computation and your
        // code will crash with an out of memory (because the input keeps accumulating).
        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]") // remove this line if running on the cluster
                .setAppName("DistinctExample");

        // The definition of the streaming spark context  below, specifies the amount of
        // time used for collecting a batch, hence giving some control on the batch size.
        // Beware that the data generator we are using is very fast, so the suggestion is to
        // use batches of less than a second, otherwise you might exhaust the JVM memory.
        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        // TECHNICAL DETAIL:
        // The streaming spark context and our code and the tasks that are spawned all
        // work concurrently. To ensure a clean shut down we use this semaphore. The
        // main thread will first acquire the only permit available, and then it will try
        // to acquire another one right after spinning up the streaming computation.
        // The second attempt at acquiring the semaphore will make the main thread
        // wait on the call. Then, in the `foreachRDD` call, when the stopping condition
        // is met the semaphore is released, basically giving "green light" to the main
        // thread to shut down the computation. We cannot call `sc.stop()` directly in `foreachRDD`
        // because it might lead to deadlocks.

        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // DEFINING THE REQUIRED DATA STRUCTURES TO MAINTAIN THE STATE OF THE STREAM
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Variable streamLength below is used to maintain the number of processed stream items.
        // It must be defined as a 1-element array so that the value stored into the array can be
        // changed within the lambda used in foreachRDD. Using a simple external counter streamLength of type
        // long would not work since the lambda would not be allowed to update it.
        long[] streamLength = new long[1]; // Stream length (an array to be passed by reference)
        streamLength[0]=0L;
        HashMap<Long, Long> histogram = new HashMap<>(); // Hash Table for the distinct elements

        // CODE TO PROCESS AN UNBOUNDED STREAM OF DATA IN BATCHES
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                // For each batch, to the following.
                // BEWARE: the `foreachRDD` method has "at least once semantics", meaning
                // that the same data might be processed multiple times in case of failure.
                .foreachRDD((batch, time) -> {
                    // this is working on the batch at time `time`.
                    if (streamLength[0] < n)
                    {
                        long batchSize = batch.count();
                        streamLength[0] += batchSize;
                        if (batchSize > 0)
                        {
                            System.out.println("Batch size at time [" + time + "] is: " + batchSize);
                            // Extract the distinct items from the batch
                            Map<Long, Long> batchItems = batch
                                    .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L))
                                    .reduceByKey((i1, i2) -> 1L)
                                    .collectAsMap();
                            // Update the streaming state. If the overall count of processed items reaches the
                            // THRESHOLD value (among all batches processed so far), subsequent items of the
                            // current batch are ignored, and no further batches will be processed
                            for (Map.Entry<Long, Long> pair : batchItems.entrySet())
                                if (!histogram.containsKey(pair.getKey()))
                                    histogram.put(pair.getKey(), 1L);
                            // If we wanted, here we could run some additional code on the global histogram
                            if (streamLength[0] >= n)
                                stoppingSemaphore.release(); // Stop receiving and processing further batches

                        }
                    }
                });

        // MANAGING STREAMING SPARK CONTEXT
        System.out.println("Starting streaming engine");
        sc.start();
        System.out.println("Waiting for shutdown condition");
        stoppingSemaphore.acquire();
        System.out.println("Stopping the streaming engine");

        /* The following command stops the execution of the stream. The first boolean, if true, also
           stops the SparkContext, while the second boolean, if true, stops gracefully by waiting for
           the processing of all received data to be completed. You might get some error messages when
           the program ends, but they will not affect the correctness. You may also try to set the second
           parameter to true.
        */

        sc.stop(false, false);
        System.out.println("Streaming engine stopped");

        // COMPUTE AND PRINT FINAL STATISTICS
        System.out.println("Number of items processed = " + streamLength[0]);
        System.out.println("Number of distinct items = " + histogram.size());
        long max = 0L;
        ArrayList<Long> distinctKeys = new ArrayList<>(histogram.keySet());
        Collections.sort(distinctKeys, Collections.reverseOrder());
        System.out.println("Largest item = " + distinctKeys.get(0));

        // Collect true frequent items, FSS, FCM

        int trueFrequentItems, fss, fcm;
    }
}
