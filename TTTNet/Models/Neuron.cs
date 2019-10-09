using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public sealed class Neuron : Unit
    {
        private float[] _weights;
        private float _bias;
        private readonly bool _activation;
        private float _learningRate;
        private IFunction _activationFn = new Sigmoid(); // FIXME: Whoops, hardcoded!!!
        private float _lastActivation;
        public override string ToString()
        {            
            var parameters = new Dictionary<string, String>()
            {
                { "Connections", _weights.Length.ToString() },
                { "Weights", String.Join("; ", _weights) },
                { "Bias", _bias.ToString() },
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return String.Join(Environment.NewLine, printable);
        }

        public Neuron(int numberOfConnections, bool activation = true, float learningRate = 0.001f)
        {
            if (numberOfConnections < 1)
            {
                throw new ArgumentException($"Number of connections: {numberOfConnections} can't be less than 1");
            }

            _activation = activation;
            _learningRate = learningRate;

            // Initialize weights and biases
            Random gen = new Random();
            float eps = .001f;

            _weights = new float[numberOfConnections];
            _bias = (float)(gen.NextDouble() + eps) % 1;

            for (int i = 0; i < numberOfConnections; ++i)
            {
                _weights[i] = (float)(gen.NextDouble() + eps) % 1;
            }
        }

        public override float ForwardPass(float[] input)
        {   
            ValidateInput(input);
            var signals = input.Zip(_weights, (i, w) => i * w);
            var signal = signals.Sum() + _bias;
            _lastActivation = _activation ? _activationFn.Calculate(signal) : signal;
            return _lastActivation;
        } 

        public override float BackwardPass(float[] gradient)
        {
            ValidateInput(gradient);
            var weightedGradient = gradient.Zip(_weights, (i, w) => i * w);
            _weights = _weights.Zip(weightedGradient, (w, g) => w +  ;)
        }

        protected override void ValidateInput(float[] input)
        {
            if (input.Length != _weights.Length){
                throw new ArgumentException($"Inputs size: {input.Length} doesn't match the number of connections: {_weights.Length}");
            }
        }

        private float DefLoss()
        {
            return 1f;
        }
    }

}