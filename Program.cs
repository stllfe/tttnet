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
            string weightsSavePath = $"{rootFolderPath}/weights.txt";
            
            string trainDataPath = $"{rootFolderPath}/trainData.txt";
            string trainLabelsPath = $"{rootFolderPath}/trainLabels.txt";
            
            string testDataPath = $"{rootFolderPath}/testData.txt";
            string testLabelsPath = $"{rootFolderPath}/testLabels.txt";
            
            var trainDataset = new Dataset(
                pathToData: trainDataPath, 
                pathToLabels: trainLabelsPath, 
                sideSize: 4
                );

            var testDataset = new Dataset(
                pathToData: testDataPath, 
                pathToLabels: testLabelsPath, 
                sideSize: 4
                );

            var net = CreateTTTNet();
            var loss = new MSELoss();

            Train(
                net: net, 
                lossFn: loss, 
                dataset: trainDataset,
                validationRatio: 0.2f
                );
            
            var weightsString = net.DumpStateToString(precision: Config.WEIGHTS_PRECISION);
            net = CreateTTTNet(weightsString);

            Test(
                net: net,
                lossFn: loss,
                dataset: testDataset
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
            int logEvery = Config.LOG_EVERY,
            float validationRatio = 0.2f
        )
        {
            net.SetLearningRate(Config.LEARNING_RATE);

            Console.WriteLine(STARLINE);
            Console.WriteLine("TRAINING");
            Console.WriteLine(UNDERLINE);

            // Reserve some elements for validation
            var validationSize = (int) MathF.Round(dataset.Length * validationRatio);
            var numberOfExamples = dataset.Length - validationSize;
            var validationIndices = Enumerable.Range(0, validationSize).ToArray();

            for (int epoch = 0; epoch < epochs; ++epoch)
            {
                var epochLosses = new float[numberOfExamples];
                for (int i = validationSize; i < numberOfExamples; ++i)
                {
                    var dataAndAnswer = dataset.GetItem(i + 2);
                    var output = net.ForwardPass(dataAndAnswer.Item1);
                    var outputAndAnswer = new Tuple<float[], float[]>(output, dataAndAnswer.Item2);
                    var gradient = lossFn.Derivative(outputAndAnswer);

                    epochLosses[i] = lossFn.Calculate(outputAndAnswer).Average();
                    net.BackwardPass(gradient);
                }

                Console.WriteLine($"epoch: [{epoch + 1}/{epochs}]\tmean loss: {epochLosses.Average()}");
                
                // Log validation results if needed
                if ((epoch + 1) % logEvery == 0)
                {
                    Console.WriteLine(UNDERLINE);
                    Evaluate(
                        net: net,
                        lossFn: lossFn,
                        dataset: dataset,
                        indices: validationIndices
                    );
                    Console.WriteLine(SEPARATOR);
                }
            }
            Console.WriteLine("\nFINISHED");
            Console.WriteLine(STARLINE);
        }

        static private void Evaluate(
            Net net,
            Loss lossFn,
            Dataset dataset,
            int[] indices
        )
        {
            var numberOfExamples = dataset.Length;
            var losses = new float[numberOfExamples];
            var indicators = new float[numberOfExamples];

            foreach (int i in indices)
            {
                var dataAndAnswer = dataset.GetItem(i);
                var output = net.ForwardPass(dataAndAnswer.Item1);
                var outputAndAnswer = new Tuple<float[], float[]>(output, dataAndAnswer.Item2);

                losses[i] = lossFn.Calculate(outputAndAnswer).Average();
                indicators[i] = Convert.ToSingle(
                    (MathF.Round(output[0]) == dataAndAnswer.Item2[0]) &&
                    (MathF.Round(output[1]) == dataAndAnswer.Item2[1])
                );

                Console.WriteLine($"true: {string.Join(", ", dataAndAnswer.Item2)}\tpredicted: {string.Join(", ", output)}");
            }
            Console.WriteLine($"accuracy: {indicators.Average()}\tmean loss: {losses.Average()}");
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

            Evaluate(
                net: net,
                lossFn: lossFn,
                dataset: dataset,
                indices: Enumerable.Range(0, dataset.Length).ToArray()
            );

            Console.WriteLine("\nFINISHED");
            Console.WriteLine(STARLINE);
        }
    }
}
