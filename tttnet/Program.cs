using System;
using TTT.Models;
using System.Linq;
using System.Collections.Generic;
using System.IO;

namespace TTT
{
    class Program
    {
        static void Main()
        {
            string rootFolderPath = @"/Users/olegpavlovich/Projects/tttnet/tttnet";
            string trainingDataPath = $"{rootFolderPath}/trainingData.txt";
            string trainingLabelsPath = $"{rootFolderPath}/trainingLabels.txt";

            int sideSize = 4;
            int inputSize = sideSize * sideSize;
            Net net = new Net(
                name: "TikTacToe",
                inputSize: inputSize,
                outputSize: 2,
                numberOfHiddenLayers: 1,
                hiddenLayerSize: 4,
                outputActivations: false
                );

            net.SetLearningRate(0.1f);
            Console.WriteLine(net);

            string[] rawData = DataManipulator.ReadData(@trainingDataPath);
            if (rawData.Length % sideSize != 0)
            {
                var error = $"Data doesn't satisfy the board size: {sideSize}x{sideSize}";
                throw new Exception(error);
            }
            var numberOfExamples = rawData.Length / sideSize;
            float[] encoded = DataManipulator.LabelEncode(rawData)
                .SelectMany(a => a).ToArray();
            float[][] data = new float[numberOfExamples][];
            for (int i = 0; i < numberOfExamples; ++i)
            {
                data[i] = new float[inputSize];
                for (int j = 0; j < inputSize; ++j)
                {
                    data[i][j] += encoded[i * j + j];
                }
            }

            string[] rawLabels = DataManipulator.ReadData(@trainingLabelsPath);
            float[][] labels = rawLabels
                .Select(l => l.Where(ch => char.IsDigit(ch))
                .Select(ch => float.Parse(ch.ToString()))
                .ToArray()).ToArray();

            Train(net, new MSELoss(), data, labels);
        }

        static void Train(
            Net net,
            Loss lossFn,
            float[][] dataset,
            float[][] trueValues,
            int epoches = 1000,
            int logEvery = 100
            )
        {
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
                }
                if (epoch % logEvery == 0)
                {
                    var output = net.ForwardPass(dataset[2]);
                    var gradient = lossFn.Derivative(output, trueValues[2]);
                    Console.WriteLine($"true: {string.Join(", ", trueValues[2])} predicted: {string.Join(", ", output)}");
                }
                Console.WriteLine($"epoch: [{epoch}/{epoches}] mean loss: {epochLosses.Sum() / numberOfExamples}");
            }
        }
    }
}
