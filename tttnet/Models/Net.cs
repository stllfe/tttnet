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

        public Net(
            string name,
            int inputSize,
            int outputSize,
            int numberOfHiddenLayers = 0,
            int hiddenLayerSize = 2,
            bool outputActivations = true)
        {
            Name = name;
            InputSize = inputSize;
            OutputSize = outputSize;

            NumberOfHiddenLayers = numberOfHiddenLayers;
            HiddenLayerSize = hiddenLayerSize;

            // TODO: add a value check 
            // Assuming they are all not empty and correct!

            var numberOfConnections = InputSize;

            for (int i = 0; i < NumberOfHiddenLayers; ++i)
            {
                Layers.Add(new Layer(HiddenLayerSize, numberOfConnections));
                numberOfConnections = Layers.Last().Neurons.Count();
            }

            Layers.Add(new Layer(OutputSize, hiddenLayerSize, outputActivations));
        }

        public override float[] ForwardPass(float[] input)
        {
            ValidateInput(input);
            foreach (var layer in Layers)
            {
                input = layer.ForwardPass(input);
            }
            return input;
        }

        public override float[] BackwardPass(float[] gradient)
        {
            ValidateInput(gradient, Direction.Backward);
            foreach (var layer in Layers.Reverse<Layer>())
            {
                gradient = layer.BackwardPass(gradient);
            }
            return gradient;
        }

        protected override void ValidateInput(float[] input, Direction direction = Direction.Forward)
        {
            var comparingLayerSize = direction == Direction.Forward ? InputSize : OutputSize;
            if (input.Length != comparingLayerSize)
            {
                var error = $"Inputs size: {input.Length} doesn't match " +
                            $"the number of connections: {comparingLayerSize}";
                throw new ArgumentException(error);
            }
        }

        public override string ToString()
        {
            var layerSizes = from layer in Layers select layer.Neurons.Count();
            var parameters = new Dictionary<string, string>()
            {
                { "Net", Name },
                { "Depth", (NumberOfHiddenLayers + 2).ToString() },
                { "Sizes", InputSize + " x " + string.Join(" x ", layerSizes)},
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return string.Join(Environment.NewLine, printable);
        }
    }

    public class Layer : Module
    {
        public List<Neuron> Neurons { get; } = new List<Neuron>();
        private readonly int _numberOfNeurons;
        private readonly int _numberOfConnections;

        public Layer(int numberOfNeurons, int numberOfConnections, bool activation = true)
        {
            _numberOfNeurons = numberOfNeurons;
            _numberOfConnections = numberOfConnections;

            for (int i = 0; i < numberOfNeurons; ++i)
            {
                Neurons.Add(new Neuron(numberOfConnections, activation));
            }
        }

        public override float[] ForwardPass(float[] input)
        {
            float[] results = new float[_numberOfNeurons];
            for (int i = 0; i < _numberOfNeurons; ++i)
            {
                results[i] = Neurons[i].ForwardPass(input);
            }
            return results;
        }

        protected override void ValidateInput(float[] input, Direction direction)
        {
            if (input.Length != _numberOfNeurons)
            {
                var error = $"Inputs size: {input.Length} doesn't match " +
                            $"the number of connections: {_numberOfNeurons}";
                throw new ArgumentException(error);
            }
        }

        public override float[] BackwardPass(float[] gradient)
        {
            // Obrain all the weighted errors from neurons
            // Sum all them across all the neurons
            // Profit!
            var neronErrors = new float[_numberOfNeurons][];
            for (int i = 0; i < _numberOfNeurons; ++i)
            {
                neronErrors[i] = Neurons[i].BackwardPass(gradient[i]);
            }

            var layerGradient = new float[_numberOfConnections];
            for (int i = 0; i < _numberOfNeurons; ++i)
            {
                for (int j = 0; j < _numberOfConnections; ++j)
                {
                    layerGradient[j] += neronErrors[i][j];
                }
            }
            return layerGradient;
        }
    }
}