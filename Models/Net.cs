using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public sealed class Net : Module
    {
        public string Name { get; }
        public int InputSize { get; }
        public int OutputSize { get; }
        public int HiddenLayerSize { get; }
        public int NumberOfHiddenLayers { get; }

        public List<Layer> Layers { get; } = new List<Layer>();

        public Function Activation { get; }

        public Net(
            string name,
            int inputSize,
            int outputSize,
            int numberOfHiddenLayers = 0,
            int hiddenLayerSize = 2,
            Function activation = null,
            bool outputActivations = true)
        {
            Name = name;
            InputSize = inputSize;
            OutputSize = outputSize;

            NumberOfHiddenLayers = numberOfHiddenLayers;
            HiddenLayerSize = hiddenLayerSize;

            if (activation == null) {
                activation = new Sigmoid();
            }

            Activation = activation;

            int numberOfConnections = InputSize;

            for (int i = 0; i < NumberOfHiddenLayers; ++i)
            {
                
                Layers.Add(new Layer(hiddenLayerSize, numberOfConnections, Activation));
                numberOfConnections = Layers.Last().NumberOfNeurons;
            }

            if (outputActivations) 
            {
                Layers.Add(new Layer(outputSize, hiddenLayerSize, Activation));
            }
            else 
            {
                Layers.Add(new Layer(outputSize, hiddenLayerSize)); 
            }            
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
                var error = $"Inputs size: {input.Length} doesn't match " +
                            $"the number of connections: {layerSize}";
                throw new ArgumentException(error);
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

        public void DumpState(string path)
        {

        }

        public void LoadState(string path)
        {

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
                var error = $"Inputs size: {input.Length} doesn't match " +
                            $"the number of connections: {NumberOfNeurons}";
                throw new ArgumentException(error);
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