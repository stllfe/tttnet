using System;
using TTT.Models;
using System.Linq;


namespace TTT
{
    class Program
    {
        static void Main()
        {
            string rootFolderPath = @"/Users/olegpavlovich/Projects/tttnet";
            string trainingDataPath = $"{rootFolderPath}/trainingData.txt";
            string trainingLabelsPath = $"{rootFolderPath}/trainingLabels.txt";
            
            var dataset = new Dataset(
                pathToData: trainingDataPath, 
                pathToLabels: trainingLabelsPath, 
                sideSize: 4
                );

            Train(
                net: CreateTTTNet(), 
                lossFn: new MSELoss(), 
                dataset: dataset,
                epoches: 35,
                logEvery: 5
                );
        }

        static Net CreateTTTNet()
        {
            Net net = new Net(
                name: "TikTacToe",
                inputSize: 16,
                outputSize: 2,
                numberOfHiddenLayers: 1,
                hiddenLayerSize: 4,
                outputActivations: false
                );

            net.SetLearningRate(0.05f);
            Console.WriteLine("The following NN has been created:\n");
            Console.WriteLine(net.ToString() + '\n');
            return net;
        }

        static void Train(
            Net net,
            Loss lossFn,
            Dataset dataset,
            int epoches = 100,
            int logEvery = 10
            )
        {
            var numberOfExamples = dataset.Length;
            var separator = new string('-', 60);
            Console.WriteLine("Training starts.");
            Console.WriteLine(new string('=', 60));

            for (int epoch = 0; epoch < epoches; ++epoch)
            {
                var epochLosses = new float[numberOfExamples];

                // Starting from 1, since we took one item for testing.
                for (int i = 0; i < numberOfExamples; ++i)
                {
                    var dataAndAnswer = dataset.GetItem(i);
                    var output = net.ForwardPass(dataAndAnswer.Item1);
                    var outputAndAnswer = new Tuple<float[], float[]>(output, dataAndAnswer.Item2);
                    var gradient = lossFn.Derivative(outputAndAnswer);

                    epochLosses[i] = lossFn.Calculate(outputAndAnswer).Average();
                    net.BackwardPass(gradient);
                }
                if ((epoch + 1) % logEvery == 0)
                {  
                    var dataAndAnswer = dataset.GetItem(0); 
                    var output = net.ForwardPass(dataAndAnswer.Item1);
                    var outputAndAnwer = new Tuple<float[], float[]>(output, dataAndAnswer.Item2);


                    Console.WriteLine(separator);
                    Console.WriteLine($"true: {string.Join(", ", dataAndAnswer.Item2)}\tpredicted: {string.Join(", ", output)}");
                    Console.WriteLine(separator);
                }
                Console.WriteLine($"epoch: [{epoch + 1}/{epoches}]\tmean loss: {epochLosses.Average()}");
            }
        }
    }
}
