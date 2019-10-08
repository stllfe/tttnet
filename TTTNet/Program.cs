using System;
using TTT.Models;
using System.Linq;

namespace TTT
{
    class Program
    {
        static void Main(string[] args)
        {
            float[] inputs = new float[]{0.1f, .34f, 3.1f, .5f, 3f, 2f};
            float[] trueValues = new float[] { 1, 2, 3, 0, 3, 4 };
            int numberOfConnections = inputs.Length;
            // int numberOfConnections = 4 * 4;
            Net net = new Net(
                name: "TikTacToe", 
                inputSize: numberOfConnections, 
                outputSize: numberOfConnections,
                numberOfHiddenLayers: 2,
                hiddenLayerSize: 4);
            Console.WriteLine(net.ToString());

            var results = net.ForwardPass(inputs);
            Console.WriteLine(String.Join(", ", results));

            var loss = new MSELoss();
            Console.WriteLine(loss.Mean(results, inputs));
        }


        static void Train(Net net, Loss lossFn, float[][] dataset, float[][] trueValues, int batchsize=4)
        {
            // FIXME: Add a new entity called Dataset? So that we would have trueValues and examples in one object
            // FIXME: Add checking for arguments. Also need to split dataset to batches
            var numberOfExamples = dataset.GetLength(0);
            var results = new float[numberOfExamples][];
            for (int i = 0; i < numberOfExamples; ++i)
            {   
                results[i] = net.ForwardPass(dataset[i]);
                var loss = lossFn.Mean(results[i], trueValues[i]);
                //net.BackwardPass(loss);
                Console.WriteLine($"[{i}/{numberOfExamples}] l: {loss}");
            }
        }
    }
}
