using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;


namespace TTT.Models
{
    public sealed class Net : Module
    {
        public string Name { get; }
        public int InputSize { get; }
        public int OutputSize { get; }
        public int[] HiddenLayerSizes { get; }
        public int NumberOfHiddenLayers { get; }

        public List<Layer> Layers { get; } = new List<Layer>();
        public Function Activation { get; }

        public Net(
            string name,
            int inputSize,
            int outputSize,
            int numberOfHiddenLayers,
            int[] hiddenLayersSizes,
            Function activation = null,
            bool outputActivations = true)
        {
            Name = name;
            InputSize = inputSize;
            OutputSize = outputSize;

            NumberOfHiddenLayers = numberOfHiddenLayers;
            HiddenLayerSizes = hiddenLayersSizes;

            if (activation == null) {
                activation = new Sigmoid();
            }

            Activation = activation;
            int numberOfConnections = InputSize;

            for (int i = 0; i < NumberOfHiddenLayers; ++i)
            {
                
                Layers.Add(new Layer(hiddenLayersSizes[i], numberOfConnections, Activation));
                numberOfConnections = Layers.Last().NumberOfNeurons;
            }

            Layers.Add(new Layer(outputSize, hiddenLayersSizes[NumberOfHiddenLayers - 1], outputActivations ? Activation : null));
        }

        public override float[] ForwardPass(float[] input)
        {
            ValidateInput(input);
            foreach (Layer layer in Layers)
            {
                input = layer.ForwardPass(input);
            }
            return input;
        }

        public override float[] BackwardPass(float[] gradient)
        {
            ValidateInput(gradient, Direction.Backward);
            foreach (Layer layer in Layers.Reverse<Layer>())
            {
                gradient = layer.BackwardPass(gradient);
            }
            return gradient;
        }

        protected override void ValidateInput(float[] input, Direction direction = Direction.Forward)
        {
           int layerSize = direction == Direction.Forward ? InputSize : OutputSize;
           if (input.Length != layerSize)
            {
                throw new ArgumentException(
                    $"Inputs size: {input.Length} doesn't match " +
                    $"the number of connections: {layerSize}"
                );
            }
        }

        public override string ToString()
        {
            int[] layerSizes = Layers.Select(l => l.Neurons.Count()).ToArray();
            var parameters = new Dictionary<string, string>()
            {
                { "Name", Name },
                { "Depth", (NumberOfHiddenLayers + 2).ToString() },
                { "Sizes", InputSize + " x " + string.Join(" x ", layerSizes) },
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return string.Join(Environment.NewLine, printable);
        }

        public void SetLearningRate(float learningRate)
        {
            foreach (Layer layer in Layers)
            {
                layer.SetLearningRate(learningRate);
            }
        }

        public string DumpStateToString(int precision)
        {
            string format = "{0:G" + precision + "}";
            int neuronIdx = 0;
            var sb = new System.Text.StringBuilder();
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    sb.Append($"{neuronIdx}:");
                    sb.Append(string.Join(',', neuron.Weights.Select(w => String.Format(format, w))));
                    sb.AppendLine(";");
                    neuronIdx++;
                }
            }
            return sb.ToString();
        }

        public void DumpStateToFile(string pathToWeights, int precision)
        {
            var weightsString = DumpStateToString(precision: precision);
            File.WriteAllLines(pathToWeights, new string[]{ weightsString });
        }

        private string[] CorrectWeightsEntries(string[] weightsEntry)
        {
            return weightsEntry.Where(v => !string.IsNullOrWhiteSpace(v)).ToArray();
        }

        public void LoadStateFromString(string weightsString)
        {
            Dictionary<int, Neuron> neuronsMap = new Dictionary<int, Neuron>();
            int neuronIdx = 0;
            foreach (var layer in Layers)
            {
                foreach (var neuron in layer.Neurons)
                {
                    neuronsMap.Add(neuronIdx, neuron);
                    neuronIdx++;
                }
            }
            foreach (var line in CorrectWeightsEntries(weightsString.Split(';')))
            {
                string[] parsedString = line.Split(':');
                int neuronId = int.Parse(parsedString.First());
                var neuron = neuronsMap[neuronId];
                
                float[] weights = parsedString[1].Split(',')
                    .Select(v => float.Parse(v.ToString()))
                    .ToArray();

                if (weights.Length != neuron.Weights.Length)
                {
                    throw new Exception(
                        $"Size mismatch for neuron: {neuronId}. " +
                        $"Expected: {neuron.Weights.Length}; got: {weights.Length} weights."
                    );
                }
                neuron.Weights = weights;
            }
        }

        public void LoadStateFromFile(string path)
        {
            string[] lines = CorrectWeightsEntries(File.ReadAllLines(path));
            string weightsString = string.Join("", lines);
            LoadStateFromString(weightsString);
        }
    }

    public class Layer : Module
    {
        public int NumberOfNeurons { get; }
        public int NumberOfConnections { get; }
        public List<Neuron> Neurons { get; } = new List<Neuron>();

        public Layer(int numberOfNeurons, int numberOfConnections, Function activation = null)
        {
            if ((numberOfNeurons < 1) || (numberOfConnections < 1))
            {
                throw new ArgumentOutOfRangeException();
            }
            NumberOfNeurons = numberOfNeurons;
            NumberOfConnections = numberOfConnections;

            for (int i = 0; i < numberOfNeurons; ++i)
            {
                Neurons.Add(new Neuron(numberOfConnections, activation));
            }
        }

        public override float[] ForwardPass(float[] input)
        {
            var results = new float[NumberOfNeurons];
            for (int i = 0; i < NumberOfNeurons; ++i)
            {
                results[i] = Neurons[i].ForwardPass(input);
            }
            return results;
        }

        protected override void ValidateInput(float[] input, Direction direction)
        {
            if (input.Length != NumberOfNeurons)
            {
                throw new ArgumentException(
                    $"Inputs size: {input.Length} doesn't match " +
                    $"the number of connections: {NumberOfNeurons}"
                );
            }
        }

        public override float[] BackwardPass(float[] gradient)
        {
            // Obtain all the weighted errors from neurons
            // Sum them all per each connection (axis = 0)
            // Profit!
            var neuronsErrors = new float[NumberOfNeurons][];
            for (int i = 0; i < NumberOfNeurons; ++i)
            {
                neuronsErrors[i] = Neurons[i].BackwardPass(gradient[i]);
            }

            var layerGradient = new float[NumberOfConnections];
            for (int i = 0; i < NumberOfNeurons; ++i)
            {
                for (int j = 0; j < NumberOfConnections; ++j)
                {
                    layerGradient[j] += neuronsErrors[i][j];
                }
            }
            return layerGradient;
        }

        public void SetLearningRate(float learningRate)
        {
            foreach (Neuron neuron in Neurons)
                {
                    neuron.LearningRate = learningRate;
                }
        }
    }
}