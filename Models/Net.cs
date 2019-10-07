using System;
using System.Linq;
using System.Collections.Generic;

namespace stupidnet.Models
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

            var numberOfLayers = 2 + this._numberOfHiddenLayers;
            var numberOfConnections = this._inputSize;
            var layerSize = 1;

            for (int i = 0; i < numberOfLayers; ++i)
            {
                layers.Add(new Layer(layerSize, numberOfConnections));
                numberOfConnections = layers.Last().neurons.Count();
            }

            layers.Add(new Layer(this._outputSize, hiddenLayerSize));
        }

        public override string ToString()
        {            
            var parameters = new Dictionary<string, String>()
            {
                {"Net", this.Name},
                {"Depth", (this._numberOfHiddenLayers + 2).ToString()},
                {"Sizes", this._inputSize + " x " 
                        + String.Join(" x ", this.layers.Select(l => l.neurons.Count())) + " x " 
                        + this._outputSize
                },
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return String.Join(Environment.NewLine, printable);
        }

        public float[] forwardPass(float[] input)
        {   
            if (input.Length != this._inputSize){
                throw new ArgumentException($"Inputs size: {input.Length} doesn't match the number of connections: {this._inputSize}");
            }
            var output = new float[_inputSize];
            foreach(var layer in this.layers)
            {
                output = layer.forwardPass(input);
                input = output;
            }
            return output;
        } 

        public void backwardPass(float[] gradient)
        {

        }
    }


    public class Layer : Module
    {
        public List<Neuron> neurons { get; } = new List<Neuron>();
        public Layer(int numberOfNeurons, int numberOfConnections)
        {
            for (int i = 0; i < numberOfNeurons; ++i)
            {
                neurons.Add(new Neuron(numberOfConnections));
            }
        }

        public float[] forwardPass(float[] input)
        {
            var results = new float[this.neurons.Count()];
            for (int i = 0; i < this.neurons.Count(); ++i)
            {
                results[i] = this.neurons[i].forwardPass(input);
            }
            return results;
        }   

        public void backwardPass(float[] gradient)
        {

        }
    }


    public interface Atom<T>
    {
        T forwardPass(float[] input);
        void backwardPass(float[] gradient);
    }

    public interface Module: Atom<float[]>
    {

    }

    public interface Unit: Atom<float>
    {

    }
}

