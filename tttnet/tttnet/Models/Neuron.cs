using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public sealed class Neuron : Unit
    {
        private float[] _weights;
        private float _bias;
        private bool _activation;
        private readonly float _learningRate;
        private IFunction _activationFn = new Sigmoid(); // FIXME: Whoops, hardcoded!!!

        // Two caching attributes
        private float _lastSignal;
        private float[] _lastInput;

        public float[] Weights { get => _weights; set => _weights = value; }
        public float Bias { get => _bias; set => _bias = value; }
        public bool Activation { get => _activation; set => _activation = value; }

        public Neuron(int numberOfConnections, bool activation = true, float learningRate = 0.5f)
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
            _lastInput = input;
            _lastSignal = signal;
            return _activation ? _activationFn.Calculate(signal) : signal;
        } 

        public override float BackwardPass(float[] gradient)
        {
            ValidateInput(gradient);
            // Actually we could cache something like the final output as _o
            // and then dActivation would equal _o * (1 - _o), 
            // but even though it's efficient, I don't want to hardcode stuff
            var weightedGradient = gradient.Zip(_weights, (i, w) => i * w).ToArray();
            var dA = _activation ? _activationFn.Derivative(_lastSignal) : 1;

            // Updating the weights
            for (int i = 0; i < _weights.Length; ++i)
            {
                _weights[i] = _weights[i] - gradient[i] * dA * _learningRate * _lastInput[i];
            }

            return weightedGradient.Sum();
        }

        protected override void ValidateInput(float[] input)
        {
            if (input.Length != _weights.Length){
                throw new ArgumentException($"Inputs size: {input.Length} doesn't match the number of connections: {_weights.Length}");
            }
        }

        public override string ToString()
        {
            var parameters = new Dictionary<string, string>()
            {
                { "Connections", _weights.Length.ToString() },
                { "Weights", String.Join("; ", _weights) },
                { "Bias", _bias.ToString() },
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return String.Join(Environment.NewLine, printable);
        }
    }

}