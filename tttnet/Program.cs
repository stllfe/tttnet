using System;
using TTT.Models;
using System.Linq;

namespace TTT
{
    class Program
    {
        static void Main(string[] args)
        {
            float[] inputs = {.05f, .10f};
            float[] trueValues = { .01f, .99f};
            int numberOfConnections = inputs.Length;
            // int numberOfConnections = 4 * 4;
            Net net = new Net(
                name: "TikTacToe", 
                inputSize: numberOfConnections, 
                outputSize: numberOfConnections,
                numberOfHiddenLayers: 1,
                hiddenLayerSize: 2);
            Console.WriteLine(net);

            // Let's script things out in order to check them
            net.Layers[0].Neurons[0].Weights = new float[] { .15f, .20f };
            net.Layers[0].Neurons[1].Weights = new float[] { .25f, .30f };
            net.Layers[0].Neurons[0].Bias = net.Layers[0].Neurons[1].Bias = 0.35f;

            net.Layers[1].Neurons[0].Weights = new float[] { .40f, .45f };
            net.Layers[1].Neurons[1].Weights = new float[] { .50f, .55f };
            net.Layers[1].Neurons[0].Bias = net.Layers[1].Neurons[1].Bias = 0.60f;

            // Enabling last activations
            net.Layers[1].Neurons[0].Activation = true;
            net.Layers[1].Neurons[1].Activation = true;

            var bareResults = net.ForwardPass(inputs);
            Console.WriteLine("\nFirst forward pass results: " + string.Join(", ", bareResults));

            var loss = new MSELoss();
            Console.WriteLine("Losses per output: " + string.Join(", ", loss.ElementWise(bareResults, trueValues)));
            Console.WriteLine("Total loss: " + loss.Sum(bareResults, trueValues));

            var firsFradient = loss.Derivative(bareResults, trueValues);
            net.BackwardPass(firsFradient);

            var oneRoundResults = net.ForwardPass(inputs);
            Console.WriteLine("\nSecond forward pass results: " + string.Join(", ", oneRoundResults));
            Console.WriteLine("Losses per output: " + string.Join(", ", loss.ElementWise(oneRoundResults, trueValues)));
            Console.WriteLine("Total loss: " + loss.Sum(oneRoundResults, trueValues));

            var epoches = 40000;
            var logEvery = 1000;

            for (int epoch = 0; epoch < epoches; ++epoch)
            {
                var results = net.ForwardPass(inputs);
                var gradient = loss.Derivative(results, trueValues);
                net.BackwardPass(gradient);

                if (epoch % logEvery == 0)
                {
                    var error = loss.Sum(results, trueValues);
                    Console.WriteLine($"[{epoch}/{epoches}] loss: [{error}]");
                }
            }

        }


        static void Train(
            Net net, 
            Loss lossFn, 
            float[][] dataset, 
            float[][] trueValues, 
            int batchsize=4, 
            int epoches=100, 
            int logEvery=10)
        {
            // FIXME: Add a new entity called Dataset? So that we would have trueValues and examples in one object
            // FIXME: Add checking for arguments. Also need to split dataset to batches

            var numberOfExamples = dataset.GetLength(0);

            for (int epoch = 0; epoch < epoches; ++epoch)
            {
                var epochLosses = new float[numberOfExamples];
                for (int e = 0; e < numberOfExamples; ++e)
                {
                    var output = net.ForwardPass(dataset[e]);
                    var gradient = lossFn.Derivative(output, trueValues[e]);

                    epochLosses[e] = lossFn.Mean(output, trueValues[e]);
                    net.BackwardPass(gradient);

                    if (e % logEvery == 0)
                    {
                        Console.WriteLine($"done: [{e}/{numberOfExamples}]");
                    }
                    //net.BackwardPass(loss);
                }
                Console.WriteLine($"epoch: [{epoch}/{epoches}] mean loss: {epochLosses.Sum() / numberOfExamples}");
            }
        }
    }
}
