package G01HW2;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.StorageLevels;
import org.apache.spark.streaming.Durations;
import org.apache.spark.streaming.api.java.JavaStreamingContext;
import scala.Tuple2;
import java.util.*;
import java.util.concurrent.Semaphore;

public class G01HW2
{
    // Hash-function parameters for Count-Min Sketch
    static final long P = 8191L; // prime p used in the 2-universal family

    // Generate a hash function h(x) = ((a*x + b) mod p) mod C; returns { a, b }
    static long[] generateHashParams(Random rnd)
    {
        long a = 1 + (long)(rnd.nextDouble() * (P - 1));  // a in [1, P-1]
        long b = (long)(rnd.nextDouble() * P);            // b in [0, P-1]
        return new long[]{a, b};
    }

    // Evaluate h_j(x) = ((a*x + b) mod p) mod C
    static int hashValue(long x, long a, long b, int C)
    {
        long val = ((a * x + b) % P + P) % P;
        return (int)(val % C);
    }

    // Update Count-Min Sketch with one item
    static void cmUpdate(long item, long[][] sketch, long[][] hashParams, int d, int w)
    {
        for (int j = 0; j < d; j++)
        {
            int k = hashValue(item, hashParams[j][0], hashParams[j][1], w);
            sketch[j][k]++;
        }
    }

    // Estimate frequency of item using Count-Min Sketch
    static long cmEstimate(long item, long[][] sketch, long[][] hashParams, int d, int w)
    {
        long minVal = Long.MAX_VALUE;
        for (int j = 0; j < d; j++)
        {
            int k = hashValue(item, hashParams[j][0], hashParams[j][1], w);
            minVal = Math.min(minVal, sketch[j][k]);
        }
        return minVal;
    }

    public static void main(String[] args) throws Exception
    {
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // INPUT READING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        if (args.length != 7)
            throw new IllegalArgumentException("USAGE: n phi epsilon delta d w portExp");

        int    n       = Integer.parseInt(args[0]);
        double phi     = Double.parseDouble(args[1]);
        double epsilon = Double.parseDouble(args[2]);
        double delta   = Double.parseDouble(args[3]);
        int    d       = Integer.parseInt(args[4]);
        int    w       = Integer.parseInt(args[5]);
        int    portExp = Integer.parseInt(args[6]);

        // Sticky Sampling rate: r = ceil(ln(1/(delta*phi)) / epsilon)
        long r = (long) Math.ceil(Math.log(1.0 / (delta * phi)) / epsilon);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // DATA STRUCTURES
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        /* Variable streamLength below is used to maintain the number of processed stream items.
        It must be defined as a 1-element array so that the value stored into the array can be
        changed within the lambda used in foreachRDD. Using a simple external counter streamLength of type
        long would not work since the lambda would not be allowed to update it. */
        long[] streamLength = new long[1];
        streamLength[0] = 0L;

        // True frequency histogram
        HashMap<Long, Long> histogram = new HashMap<>(); // Hash Table for the distinct elements

        // Sticky Sampling dictionary: item -> count
        HashMap<Long, Long> stickyDict = new HashMap<>();

        // Count-Min Sketch: d rows x w columns
        long[][] sketch = new long[d][w];
        Random rnd = new Random();
        long[][] hashParams = new long[d][2];
        for (int j = 0; j < d; j++)
        {
            long[] params = generateHashParams(rnd);
            hashParams[j][0] = params[0];
            hashParams[j][1] = params[1];
        }

        // FCM: items added the first time their CM estimated freq >= phi*n
        // We track which items are already in FCM to avoid duplicates
        HashSet<Long> fcmSet = new HashSet<>();

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // STREAM PROCESSING
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        // Spark setup
        SparkConf conf = new SparkConf(true)
                .setMaster("local[*]")
                .setAppName("G01HW2");

        JavaStreamingContext sc = new JavaStreamingContext(conf, Durations.milliseconds(100));
        sc.sparkContext().setLogLevel("ERROR");

        Semaphore stoppingSemaphore = new Semaphore(1);
        stoppingSemaphore.acquire();

        // BEWARE: the foreachRDD method has "at least once semantics", meaning
        // that the same data might be processed multiple times in case of failure.
        sc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevels.MEMORY_AND_DISK)
                .foreachRDD((batch, time) -> {
                     // this is working on the batch at time *time*
                    if (streamLength[0] < n)
                    {
                        long batchSize = batch.count();
                        streamLength[0] += batchSize;
                        if (batchSize > 0)
                        {
                            // Collect distinct items with their frequencies in this batch
                            // (use i1+i2 to count, NOT the deduplication bug i1->1L)
                            Map<Long, Long> batchItems = batch
                                    .mapToPair(s -> new Tuple2<>(Long.parseLong(s), 1L))
                                    .reduceByKey((i1, i2) -> i1 + i2)
                                    .collectAsMap();

                            // Update the streaming state. If the overall count of processed items reaches the
                            // THRESHOLD value (among all batches processed so far), subsequent items of the
                            // current batch are ignored, and no further batches will be processed
                            for (Map.Entry<Long, Long> entry : batchItems.entrySet())
                            {
                                long item  = entry.getKey();
                                long count = entry.getValue();

                                // Update true histogram
                                histogram.merge(item, count, Long::sum);

                                // Update Sticky Sampling
                                if (stickyDict.containsKey(item))
                                    stickyDict.merge(item, count, Long::sum);
                                else
                                {
                                    // Each individual occurrence gets a Bernoulli trial with prob r/n
                                    // (approximation: treat the batch count as individual arrivals)
                                    long added = 0L;
                                    for (long i = 0; i < count; i++)
                                    {
                                        if (rnd.nextDouble() <= (double) r / (double) n)
                                            added++;
                                    }
                                    if (added > 0)
                                        stickyDict.put(item, added);
                                }

                                // Update Count-Min Sketch (once per actual occurrence)
                                for (long i = 0; i < count; i++)
                                    cmUpdate(item, sketch, hashParams, d, w);

                                // Check if item should be added to FCM
                                if (!fcmSet.contains(item))
                                {
                                    long est = cmEstimate(item, sketch, hashParams, d, w);
                                    if (est >= phi * n)
                                        fcmSet.add(item);
                                }
                            }

                            if (streamLength[0] >= n)
                            {
                                stoppingSemaphore.release();
                            }
                        }
                    }
                });

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // MANAGE STREAMING CONTEXT
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        System.out.println("Starting streaming engine");
        sc.start();
        System.out.println("Waiting for shutdown condition");
        stoppingSemaphore.acquire();

        System.out.println("Stopping the streaming engine");
        /* The following command stops the execution of the stream. The first boolean, if true, also
           stops the SparkContext, while the second boolean, if true, stops gracefully by waiting for
           the processing of all received data to be completed. Error messages might be shown when
           the program ends, but they will not affect the correctness. It is also possible to set
           the second parameter to true.
        */
        sc.stop(false, false);
        System.out.println("Streaming engine stopped");

        // Print input parameters
        System.out.println("INPUT PARAMETERS");
        System.out.println("n = " + n);
        System.out.println("phi = " + phi);
        System.out.println("epsilon = " + epsilon);
        System.out.println("delta = " + delta);
        System.out.println("d = " + d);
        System.out.println("w = " + w);
        System.out.println("port = " + portExp);

        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
        // COMPUTE AND PRINT RESULTS
        // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

        long threshold = (long)(phi * n); // Items with freq >= phi*n are true frequent

        // --- TRUE FREQUENT ITEMS ---
        List<Long> trueFrequent = new ArrayList<>();
        for (Map.Entry<Long, Long> e : histogram.entrySet())
        {
            if (e.getValue() >= threshold)
                trueFrequent.add(e.getKey());
        }
        Collections.sort(trueFrequent);

        System.out.println("TRUE FREQUENT ITEMS");
        for (long item : trueFrequent)
            System.out.println("Item = " + item + " True Freq = " + histogram.get(item));

        // --- STICKY SAMPLING ---
        // FSS: items in stickyDict with count >= (phi - epsilon) * n
        double sssThreshold = (phi - epsilon) * n;
        List<Long> fss = new ArrayList<>();
        for (Map.Entry<Long, Long> e : stickyDict.entrySet())
        {
            if (e.getValue() >= sssThreshold)
                fss.add(e.getKey());
        }
        Collections.sort(fss);

        System.out.println("STICKY SAMPLING");
        System.out.println("Size of dictionary = " + stickyDict.size());
        for (long item : fss)
        {
            long trueFreq = histogram.getOrDefault(item, 0L);
            System.out.println("Item = " + item + " True Freq = " + trueFreq);
        }

        // --- COUNT-MIN SKETCH ---
        List<Long> fcmList = new ArrayList<>(fcmSet);
        Collections.sort(fcmList);

        System.out.println("COUNT-MIN SKETCH");
        System.out.println("Size of F_CM = " + fcmList.size());
        for (long item : fcmList)
        {
            long trueFreq = histogram.getOrDefault(item, 0L);
            System.out.println("Item = " + item + " True Freq = " + trueFreq);
        }
    }
}
