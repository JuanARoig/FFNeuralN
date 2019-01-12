using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralN
{
    public class FeedForwardNetwork
    {
        //user specification on layers INCLUDES input, output layers.
        private int lyerAmnt;
        private int[] lyerSzes;

        private List<double[]> biases;
        private List<double[,]> weights;

        private List<double[]> biasupdates;
        private List<double[,]> weightsupdates;

        private List<double[]> network;
        //test with different amounts of layers. observe convergence.
        public FeedForwardNetwork(int layerAmount, int[] layerSizes)
        {
            Random random = new Random();

            this.biases = new List<double[]>();
            this.weights = new List<double[,]>();

            this.biasupdates = new List<double[]>();
            this.weightsupdates = new List<double[,]>();

            this.network = new List<double[]>();
 
            //initialize components of instance.
            this.lyerAmnt = layerAmount;
            this.lyerSzes = layerSizes;

            for (int i = 1; i < this.lyerAmnt; i++)
            {
                double[] biaslayer = new double[this.lyerSzes[i]];
                for (int j = 0; j < this.lyerSzes[i]; j++)
                {
                    biaslayer[j] = Helper.Sigmoid(2 * random.NextDouble());
                }
                this.biases.Add(biaslayer);
            }

            for (int i = 1; i < this.lyerAmnt; i++)
            {
                double[,] weightsmatrix = new double[this.lyerSzes[i], this.lyerSzes[i - 1]];
                for (int j = 0; j < this.lyerSzes[i]; j++)
                {
                    for (int k = 0; k < this.lyerSzes[i - 1]; k++)
                    {
                        weightsmatrix[j, k] = Helper.Sigmoid(2 * random.NextDouble());
                    }
                }
                this.weights.Add(weightsmatrix);
            }

            for (int i = 1; i < this.lyerAmnt; i++)
            {
                double[] biasupdatelayer = new double[this.lyerSzes[i]];
                this.biasupdates.Add(biasupdatelayer);
            }

            for (int i = 1; i < this.lyerAmnt; i++)
            {
                double[,] weightsupdatematrix = new double[this.lyerSzes[i], this.lyerSzes[i - 1]];
                this.weightsupdates.Add(weightsupdatematrix);
            }

            for (int i = 0; i < this.lyerAmnt - 1; i++)
            {
                this.network.Add(new double[this.lyerSzes[i + 1]]);
            }

        }

        public int getLayerAmount() { return this.lyerAmnt; }
        public int[] getLayerSizes() { return this.lyerSzes; }

        //training function
        public void Train(double learningRate, int iterationsPerSet, double[] inputSet, double[] outputSet)
        {
            for (int l = 0; l < iterationsPerSet; l++)
            {

                for (int i = 0; i < this.weightsupdates.Count; i++)
                {
                    double[,] mat = this.weightsupdates[i];
                    for (int j = 0; j < mat.GetLength(0); j++)
                    {
                        for (int k = 0; k < mat.GetLength(1); k++)
                        {
                            mat[j, k] = (-1 * learningRate * CostPDerivweights(inputSet, outputSet, i, j, k));
                        }
                    }
                }

                for (int i = 0; i < this.biasupdates.Count; i++)
                {
                    double[] biaslyer = this.biasupdates[i];
                    for (int j = 0; j < biaslyer.Length; j++)
                    {
                        biaslyer[j] = (-1 * learningRate * CostPDerivbiases(inputSet, outputSet, i, j));
                    }
                }
                //updating weights and biases.
                this.biases = this.biasupdates;
                this.weights = this.weightsupdates;

            }
        }
        //Cost Function
        public double Cost(double[] inputSet, double[] outputSet)
        {
            this.network.Insert(0, inputSet);
            for (int i = 1; i < this.lyerAmnt; i++)
            {
                this.network[i] = Helper.Sigmoid(Helper.AddVec(Helper.MultVecMa(this.weights[i - 1], this.network[i - 1]), this.biases[i - 1]));
            }
            double[] errVec = Helper.SubVec(outputSet, this.network[this.lyerAmnt - 1]);
            double error = 0;
            for (int i = 0; i < errVec.Length; i++)
            {
                error += 0.5 * Math.Pow(errVec[i], 2);
            }
            return error;
        }

        private double CostPDerivweights(double[] inputSet, double[] outputSet, int pos1, int pos2, int pos3)
        {
            double r = 0.0001;
            double[,] w = this.weights[pos1];

            w[pos2, pos3] = w[pos2, pos3] + r;

            double affected = Cost(inputSet, outputSet);

            w[pos2, pos3] = w[pos2, pos3] - r;

            double original = Cost(inputSet, outputSet);

            double partiald = (affected - original) / r;

            return partiald;
        }

        private double CostPDerivbiases(double[] inputSet, double[] outputSet, int pos1, int pos2)
        {
            double r = 0.0001;
            double[] b = this.biases[pos1];

            b[pos2] = b[pos2] + r;

            double affected = Cost(inputSet, outputSet);

            b[pos2] = b[pos2] - r;

            double original = Cost(inputSet, outputSet);

            double partiald = (affected - original) / r;

            return partiald;
        }



    }

    public class Helper
    {
        public static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-1 * x));
        }

        public static double[] Sigmoid(double[] xvec)
        {
            double[] final = new double[xvec.Length];
            for (int i = 0; i < xvec.Length; i++)
            {
                final[i] = 1 / (1 + Math.Exp(-1 * xvec[i]));
            }
            return final;
        }

        public static double[] MultVecMa(double[,] matrix, double[] vector)
        {
            double[] final = new double[matrix.GetLength(0)];

            for (int i = 0; i < matrix.GetLength(0); i++)
            {
                final[i] = 0;
                for (int j = 0; j < matrix.GetLength(1); j++)
                {
                    final[i] += matrix[i, j] * vector[j];
                }
            }

            return final;
        }

        public static double[] AddVec(double[] A, double[] B)
        {
            double[] final = new double[A.Length];
            for (int i = 0; i < A.Length; i++)
            {
                final[i] = A[i] + B[i];
            }
            return final;
        }

        public static double[] SubVec(double[] A, double[] B)
        {
            double[] final = new double[A.Length];
            for (int i = 0; i < A.Length; i++)
            {
                final[i] = A[i] - B[i];
            }
            return final;
        }

        public double[] ScalarVectorMult(double scalarA, double[] vectorA)
        {
            double[] finalVector = new double[vectorA.Length];
            for (int i = 0; i < vectorA.Length; i++)
            {
                finalVector[i] = (vectorA[i] * scalarA);
            }
            return finalVector;
        }

    }
}
