using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public sealed class Net : Module
    {
        public string Name { get; }
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

            // TODO: add a value check 
            // Assuming they are all not empty and correct!

            var numberOfConnections = _inputSize;

            for (int i = 0; i < _numberOfHiddenLayers; ++i)
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
            return string.Join(Environment.NewLine, printable);
        }

        public override float[] ForwardPass(float[] input)
        {
            foreach (var layer in Layers)
            {
                input = layer.ForwardPass(input);
            }
            return input;
        }

        public override float[] BackwardPass(float[] gradient)
        { 
            // FIXME: Maybe we should calculate the loss derivative here?
            foreach (var layer in Layers.Reverse<Layer>())
            {
                gradient = layer.BackwardPass(gradient);
            }
            return gradient;
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


        protected override void ValidateInput(float[] input)
        {
            // FIXME: Write it!
        }

        public override float[] BackwardPass(float[] gradient)
        {
            var layerGradient = new float[Neurons.Count()];
            for (int i = 0; i < Neurons.Count(); ++i)
            {
                layerGradient[i] = Neurons[i].BackwardPass(gradient);
            }
            return layerGradient;
        }
    }

    // Basic interface, representing an object that passes signals through itself
    public interface INeural<T>
    {
        T ForwardPass(float[] input);
        T BackwardPass(float[] gradient);
    }

    // Module is a bigger part then Unit, hence the module itself may consist out of units
    // Thus is passes a collection of signals from all its units
    public abstract class Module : INeural<float[]>
    {
        public abstract float[] BackwardPass(float[] gradient);
        public abstract float[] ForwardPass(float[] input);
        protected abstract void ValidateInput(float[] input);
    }

    // Unit is something on its own like a standalone neuron
    // Thus it passes only its own signal back and forth
    public abstract class Unit : INeural<float>
    {
        public abstract float BackwardPass(float[] gradient);
        public abstract float ForwardPass(float[] input);
        protected abstract void ValidateInput(float[] input);
    }
}