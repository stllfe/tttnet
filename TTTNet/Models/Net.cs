using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public class Net : Module
    {
        public String Name { get; }
        public int InputSize { get => _inputSize; }
        public int OutputSize { get => _outputSize; }
        public List<Layer> Layers { get; } = new List<Layer>();


        private readonly int _hiddenLayerSize;
        private readonly int _numberOfHiddenLayers;
        private readonly int _inputSize;
        private readonly int _outputSize;


        public Net(string name, int inputSize, int outputSize, int numberOfHiddenLayers = 0, int hiddenLayerSize = 2)
        {
            Name = name;

            _inputSize = inputSize;
            _outputSize = outputSize;
            _numberOfHiddenLayers = numberOfHiddenLayers;
            _hiddenLayerSize = hiddenLayerSize;

            // TODO: add a check 
            // Assuming they are all not empty and correct!

            var numberOfConnections = this._inputSize;

            for (int i = 0; i < this._numberOfHiddenLayers; ++i)
            {
                Layers.Add(new Layer(_hiddenLayerSize, numberOfConnections));
                numberOfConnections = Layers.Last().Neurons.Count();
            }

            // the output layers should not have activations
            Layers.Add(new Layer(_outputSize, hiddenLayerSize, false));
        }

        public override string ToString()
        {
            var parameters = new Dictionary<string, string>()
            {
                { "Net", Name },
                { "Depth", (_numberOfHiddenLayers + 2).ToString() },
                { "Sizes",
                    _inputSize + " x " +
                    String.Join(" x ", from layer in Layers select layer.Neurons.Count())
                },
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return String.Join(Environment.NewLine, printable);
        }

        public override float[] ForwardPass(float[] input)
        {

            var output = new float[_inputSize];
            foreach (var layer in Layers)
            {
                output = layer.ForwardPass(input);
                input = output;
            }
            return output;
        }

        public override void BackwardPass(float[] gradient)
        {

        }

        protected override void ValidateInput(float[] input)
        {
            // FIXME: Won't work on backward pass. Change this
            if (input.Length != _inputSize)
            {
                throw new ArgumentException($"Inputs size: {input.Length} doesn't match the number of connections: {_inputSize}");
            }
        }
    }


    public class Layer : Module
    {
        public List<Neuron> Neurons { get; } = new List<Neuron>();
        public Layer(int numberOfNeurons, int numberOfConnections, bool activation = true)
        {
            for (int i = 0; i < numberOfNeurons; ++i)
            {
                Neurons.Add(new Neuron(numberOfConnections, activation));
            }
        }

        public override float[] ForwardPass(float[] input)
        {
            var results = new float[Neurons.Count()];
            for (int i = 0; i < Neurons.Count(); ++i)
            {
                results[i] = Neurons[i].ForwardPass(input);
            }
            return results;
        }

        public override void BackwardPass(float[] gradient)
        {

        }

        protected override void ValidateInput(float[] input)
        {

        }
    }

    public abstract class Atom<T>
    {
        public abstract T ForwardPass(float[] input);
        public abstract void BackwardPass(float[] gradient);
        protected abstract void ValidateInput(float[] input);
    }

    public abstract class Module : Atom<float[]>
    {
        // FIXME: bad structural decision
        // public abstract void BackwardPass(float value);
    }

    public abstract class Unit : Atom<float>
    {

    }
}