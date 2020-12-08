using System;
using System.IO;
using System.Linq;
using TTT.Models;


namespace TTT
{
    class Program
    { 
        const int LINE_LENGTH = 50;
        static string SEPARATOR = new string('-', LINE_LENGTH);
        static string UNDERLINE = new string('=', LINE_LENGTH);
        static string STARLINE = new string('*', LINE_LENGTH);
        static void Main()
        {
            string rootFolderPath = Environment.CurrentDirectory;
            string trainingDataPath = $"{rootFolderPath}/trainingData.txt";
            string trainingLabelsPath = $"{rootFolderPath}/trainingLabels.txt";
            string weightsSavePath = $"{rootFolderPath}/weights.txt";
            
            var dataset = new Dataset(
                pathToData: trainingDataPath, 
                pathToLabels: trainingLabelsPath, 
                sideSize: 4
                );

            var net = CreateTTTNet();

            Train(
                net: net, 
                lossFn: new MSELoss(), 
                dataset: dataset
                );
            
            var weightsString = net.DumpStateToString(precision: Config.WEIGHTS_PRECISION);
            net = CreateTTTNet(weightsString);

            Test(
                net: net,
                lossFn: new MSELoss(),
                dataset: dataset
            );
        }

        static Net CreateTTTNet()
        {
            var rnd = new Random(Config.RANDOM_SEED);
            Net net = new Net(
                name: "TikTacToe",
                inputSize: Config.SIDE_SIZE * Config.SIDE_SIZE,
                outputSize: 2,
                numberOfHiddenLayers: Config.NUM_HIDDEN_LAYERS,
                hiddenLayersSizes: Config.HIDDEN_LAYERS_SIZES,
                outputActivations: false
                );            
            Console.WriteLine("The following NN is created:\n");
            Console.WriteLine(net.ToString() + '\n');
            return net;
        }

        static Net CreateTTTNet(string weightsStringOrPath)
        {
            var net = CreateTTTNet();
            if (File.Exists(weightsStringOrPath))
            {
                net.LoadStateFromFile(weightsStringOrPath);
            }
            else 
            {
                net.LoadStateFromString(weightsStringOrPath);
            }
            Console.WriteLine("NN state is restored.");
            return net;
        }

        static void Train(
            Net net,
            Loss lossFn,
            Dataset dataset,
            int epochs = Config.NUM_EPOCHS,
            int logEvery = Config.LOG_EVERY
            )
        {
            net.SetLearningRate(Config.LEARNING_RATE);
            Console.WriteLine(STARLINE);
            Console.WriteLine("TRAINING");
            Console.WriteLine(UNDERLINE);

            // Reserve the first element for testing.
            var numberOfExamples = dataset.Length - 2;

            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                var epochLosses = new float[numberOfExamples];
                for (int i = 0; i < numberOfExamples; ++i)
                {
                    var dataAndAnswer = dataset.GetItem(i + 2);
                    var output = net.ForwardPass(dataAndAnswer.Item1);
                    var outputAndAnswer = new Tuple<float[], float[]>(output, dataAndAnswer.Item2);
                    var gradient = lossFn.Derivative(outputAndAnswer);

                    epochLosses[i] = lossFn.Calculate(outputAndAnswer).Average();
                    net.BackwardPass(gradient);
                }

                Console.WriteLine($"epoch: [{epoch + 1}/{epochs}]\tmean loss: {epochLosses.Average()}");
                
                // Log results if needed.
                if ((epoch + 1) % logEvery == 0)
                {
                    Console.WriteLine(UNDERLINE);
                    for (int i = 0; i < 4; ++i)
                    {  
                        var dataAndAnswer = dataset.GetItem(i); 
                        var output = net.ForwardPass(dataAndAnswer.Item1);
                        var outputAndAnwer = new Tuple<float[], float[]>(output, dataAndAnswer.Item2);

                        Console.WriteLine($"true: {string.Join(", ", dataAndAnswer.Item2)}\tpredicted: {string.Join(", ", output)}");
                    }
                    Console.WriteLine(SEPARATOR);
                }
            }
            Console.WriteLine("\nFINISHED");
            Console.WriteLine(STARLINE);
        }

        static void Test(
            Net net,
            Loss lossFn,
            Dataset dataset
            )
        {
            Console.WriteLine(STARLINE);
            Console.WriteLine("TESTING");
            Console.WriteLine(UNDERLINE);

            var numberOfExamples = 2;

            var losses = new float[numberOfExamples];
            var indicators = new float[numberOfExamples];
            for (int i = 0; i < numberOfExamples; ++i)
            {
                var dataAndAnswer = dataset.GetItem(i);
                var output = net.ForwardPass(dataAndAnswer.Item1);
                var outputAndAnswer = new Tuple<float[], float[]>(output, dataAndAnswer.Item2);

                losses[i] = lossFn.Calculate(outputAndAnswer).Average();
                indicators[i] = Convert.ToSingle(
                    (MathF.Round(output[0]) == dataAndAnswer.Item2[0]) &&
                    (MathF.Round(output[1]) == dataAndAnswer.Item2[1])
                );
            }
            Console.WriteLine($"accuracy: {indicators.Average()}\tmean loss: {losses.Average()}");
            Console.WriteLine("\nFINISHED");
            Console.WriteLine(STARLINE);
        }
    }
}
