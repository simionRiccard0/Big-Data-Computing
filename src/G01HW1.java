import java.util.ArrayList;
import java.util.Random;
import java.util.Iterator;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import scala.Tuple2;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.api.java.JavaRDD;

public class G01HW1
{
    public static ArrayList<Tuple2<Vector,String>> FairFFT(ArrayList<Tuple2<Vector,String>> U, int kA, int kB)
    {
        // k-center clustering where k = kA + kB (kA, kB given as input)

        ArrayList<Tuple2<Vector,String>> S = new ArrayList<>(); //solution set S initialized

        if (U.isEmpty())
            return S;

        int n = U.size();

        boolean[] chosen = new boolean[n]; //tracks already selected points

        Random rand = new Random();

        //select first center randomly
        int firstIndex = rand.nextInt(n);

        Tuple2<Vector,String> c1 = U.get(firstIndex);

        S.add(c1);

        chosen[firstIndex] = true;

        int countA = 0;
        int countB = 0;

        if (c1._2.equals("A")) //_2 gives access to the second member of the tuple
            countA++;
        else
            countB++;

        // Budget check
        while (countA < kA || countB < kB) {

            int bestIndex = -1;

            double maxDist = -1.0;

            for (int i = 0; i < n; i++) {

                if (chosen[i])
                    continue; //skip already selected points

                Tuple2<Vector,String> p = U.get(i);

                String group = p._2;

                //respect fairness constraint, budget check
                if (group.equals("A") && countA >= kA)
                    continue;

                if (group.equals("B") && countB >= kB)
                    continue;

                double minDist =
                        Double.MAX_VALUE;

                //distance from closest center
                for (Tuple2<Vector,String> c : S) {

                    double dist = Vectors.sqdist(p._1, c._1);

                    if (dist < minDist)
                        minDist = dist;
                }

                if (minDist > maxDist)
                {
                    maxDist = minDist;
                    bestIndex = i;
                }
            }

            if (bestIndex == -1)
                break;

            Tuple2<Vector,String> ci = U.get(bestIndex);

            S.add(ci);

            chosen[bestIndex] = true;

            if (ci._2.equals("A"))
                countA++;
            else
                countB++;

        }

        return S;
    }

    /* NOTE: As mentioned in the assignment, the MR-Fair-FFT implementation should be equal or very similar to MRFFT,
    just calling the FairFFT method - no logical changes should be needed. */

    public static ArrayList<Tuple2<Vector,String>> MRFairFFT(JavaRDD<Tuple2<Vector,String>> U, int kA, int kB)
    {
            //  kA + kB = k, as above

            //ROUND 1

                //MapPhase
                // U.mapPartitions() allows to process each partition separately
                JavaRDD<Tuple2<Vector,String>> coresets = U.mapPartitions((Iterator<Tuple2<Vector,String>> iter) -> {

                                    // Convert partition to ArrayList, needed by FairFFT
                                    ArrayList<Tuple2<Vector,String>> partition = new ArrayList<>();

                                    //iterator used to read all the elements of the current partition
                                    //everything is managed by Spark
                                    while (iter.hasNext())
                                        partition.add(iter.next());
                //ReducePhase
                                    // Run FairFFT locally
                                    ArrayList<Tuple2<Vector,String>> localCenters = FairFFT(partition, kA, kB);

                                    return localCenters.iterator();
                                }
                        );

            //ROUND 2

            // Collect coresets to driver
            ArrayList<Tuple2<Vector,String>> collected = new ArrayList<>(coresets.collect());

            // Run FairFFT again
            ArrayList<Tuple2<Vector,String>> finalCenters = FairFFT(collected, kA, kB);

            return finalCenters;
    }

    public static void main(String [] args)
    {
        String inputPath = args[0];
        int kA = Integer.parseInt(args[1]);
        int kB = Integer.parseInt(args[2]);
        int L  = Integer.parseInt(args[3]);

        System.out.println("Input file = " + inputPath);
        System.out.println("kA = " + kA);
        System.out.println("kB = " + kB);
        System.out.println("L = " + L);

        Logger.getLogger("org").setLevel(Level.OFF);
        Logger.getLogger("akka").setLevel(Level.OFF);
        SparkConf conf = new SparkConf(true).setAppName("G01HW1").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        JavaRDD<String> lines = sc.textFile(inputPath);

        // Convert lines into points
        JavaRDD<Tuple2<Vector,String>> inputPoints = lines.map(MapFunctions::mapPoints).repartition(L);

        long N = inputPoints.count();
        long NA = inputPoints.filter(p -> p._2.equals("A")).count();
        long NB = inputPoints.filter(p -> p._2.equals("B")).count();

        System.out.println("N = " + N);
        System.out.println("NA = " + NA);
        System.out.println("NB = " + NB);

        long start = System.currentTimeMillis();

        ArrayList<Tuple2<Vector,String>> S = MRFairFFT(inputPoints, kA, kB);

        long end = System.currentTimeMillis();

        System.out.println("Centers:");

        for (Tuple2<Vector,String> c : S) {

            System.out.println(
                    c._1 + " " + c._2
            );
        }

        double objective = inputPoints.map(p -> {

                    double minDist = Double.MAX_VALUE;

                    for (Tuple2<Vector,String> c : S)
                    {
                        double dist = Vectors.sqdist(p._1, c._1);

                        if (dist < minDist)
                            minDist = dist;
                    }

                    return minDist;

                }).reduce((x,y) -> Math.max(x,y));

        System.out.println("Objective value = " + objective);
        System.out.println("Time MRFairFFT = " + (end - start) + " ms");

        sc.close();
    }
}

class MapFunctions
{
    public static Tuple2<Vector,String> mapPoints(String line)
    {
        String[] tokens = line.split(",");

        int dim = tokens.length - 1;
        double[] coords = new double[dim];

        for (int i = 0; i < dim; i++)
            coords[i] = Double.parseDouble(tokens[i]);

        Vector v = Vectors.dense(coords);

        String group = tokens[dim];

        return new Tuple2<>(v, group);
    }

}