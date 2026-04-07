import java.util.ArrayList;
import java.util.Random;

import scala.Tuple2;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;

public class G01HW1 {

    public static ArrayList<Tuple2<Vector,String>> FairFFT(ArrayList<Tuple2<Vector,String>> P, int kA, int kB){

        int k = kA + kB; //kA,kB given as input

        ArrayList<Tuple2<Vector,String>> S = new ArrayList<>(); //solution set S initialized

        if (P.isEmpty())
            return S;

        int n = P.size();

        boolean[] chosen = new boolean[n]; //tracks already selected points

        Random rand = new Random();

        //select first center randomly
        int firstIndex = rand.nextInt(n);

        Tuple2<Vector,String> c1 = P.get(firstIndex);

        S.add(c1);

        chosen[firstIndex] = true;

        int countA = 0;
        int countB = 0;

        if (c1._2.equals("A")) //_2 gives access to the second term of the tuple
            countA++;
        else
            countB++;

        // We use this nice strategly

        while (countA < kA || countB < kB) {

            int bestIndex = -1;

            double maxDist = -1.0;

            for (int i = 0; i < n; i++) {

                if (chosen[i])
                    continue; //skip already selected points

                Tuple2<Vector,String> p = P.get(i);

                String group = p._2;

                //respect fairness constraint
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

                if (minDist > maxDist) {

                    maxDist = minDist;
                    bestIndex = i;

                }
            }

            if (bestIndex == -1)
                break;

            Tuple2<Vector,String> ci = P.get(bestIndex);

            S.add(ci);

            chosen[bestIndex] = true;

            if (ci._2.equals("A"))
                countA++;
            else
                countB++;

        }

        return S;
    }

}