using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public class Net : Module
    {
        public String Name { get; }
        private int _hiddenLayerSize;
        private int _numberOfHiddenLayers;
        private int _inputSize;
        private int _outputSize;
        public List<Layer> layers { get; } = new List<Layer>();

        public Net(string name, int inputSize, int outputSize, int numberOfHiddenLayers = 0, int hiddenLayerSize = 2)
        {
            this.Name = name;
            
            this._inputSize = inputSize;
            this._outputSize = outputSize;
            this._numberOfHiddenLayers = numberOfHiddenLayers;
            this._hiddenLayerSize = hiddenLayerSize;

            // TODO: add a check 
            // Assuming they are all not empty and correct!

            var numberOfConnections = this._inputSize;

            for (int i = 0; i < this._numberOfHiddenLayers; ++i)
            {
                layers.Add(new Layer(this._hiddenLayerSize, numberOfConnections));
                numberOfConnections = layers.Last().neurons.Count();
            }

            // the output layers should not have activations
            layers.Add(new Layer(this._outputSize, hiddenLayerSize, false));
        }

        public override string ToString()
        {
            var parameters = new Dictionary<string, String>()
            { 
                { "Net", this.Name }, 
                { "Depth", (this._numberOfHiddenLayers + 2).ToString() }, 
                { "Sizes", 
                    this._inputSize + " x " + 
                    String.Join(" x ", this.layers.Select (l => l.neurons.Count ()))
                },
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return String.Join(Environment.NewLine, printable);
        }

        public override float[] ForwardPass(float[] input)
        {

            var output = new float[_inputSize];
            foreach (var layer in this.layers)
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
            if (input.Length != this._inputSize)
            {
                throw new ArgumentException($"Inputs size: {input.Length} doesn't match the number of connections: {this._inputSize}");
            }
        }
    }


    public class Layer : Module
    {
        public List<Neuron> neurons { get; } = new List<Neuron>();
        public Layer(int numberOfNeurons, int numberOfConnections, bool activation = true)
        {
            for (int i = 0; i < numberOfNeurons; ++i)
            {
                neurons.Add(new Neuron(numberOfConnections, activation));
            }
        }

        public override float[] ForwardPass(float[] input)
        {
            var results = new float[this.neurons.Count()];
            for (int i = 0; i < this.neurons.Count(); ++i)
            {
                results[i] = this.neurons[i].ForwardPass(input);
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

    }

    public abstract class Unit : Atom<float>
    {

    }
}