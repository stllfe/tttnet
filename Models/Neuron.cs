using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public sealed class Neuron : Unit
    {
        private float[] _weights;
        private float _bias;

        private bool _withActivation;
        private float _learningRate;
        public float LearningRate { get => _learningRate; set => _learningRate = value; }
        public float[] Weights { get => _weights; set => _weights = value; }

        private Function _activationFn;

        // Two memorizing attributes
        private float _lastSignal;
        private float[] _lastInput;

        public Neuron(int numberOfConnections, Function activation = null, float learningRate = 0.5f)
        {
            if (numberOfConnections < 1)
            {
                throw new ArgumentException($"Number of connections: {numberOfConnections} can't be less than 1");
            }

            _activationFn = activation;
            _withActivation = activation != null;
            _learningRate = learningRate;

            // Initialize weights and biases
            var rnd = new Random(Config.RANDOM_SEED);

            _weights = new float[numberOfConnections];
            _bias = (float)(rnd.NextDouble() + Config.EPS) % Config.MAX_INIT_WEIGHTS;

            for (int i = 0; i < numberOfConnections; ++i)
            {
                _weights[i] = (float)(rnd.NextDouble() + Config.EPS) % Config.MAX_INIT_WEIGHTS;
            }
        }

        public override float ForwardPass(float[] input)
        {   
            ValidateInput(input);
            var signals = input.Zip(_weights, (i, w) => i * w);
            var signal = signals.Sum() + _bias;
            _lastInput = input;
            _lastSignal = signal;
            return _withActivation ? _activationFn.Calculate(signal) : signal;
        } 

        public override float[] BackwardPass(float neuronError)
        {
            var weightedGradient = _weights.Select(w => neuronError * w).ToArray();
            var dA = _withActivation ? _activationFn.Derivative(_lastSignal) : 1;

            // Updating the weights
            for (int i = 0; i < _weights.Length; ++i)
            {
                _weights[i] = _weights[i] - neuronError * dA * LearningRate * _lastInput[i];
            }

            return weightedGradient;
        }

        protected override void ValidateInput(float[] input)
        {
            var error = $"Inputs size: {input.Length} doesn't match " +
                        $"the number of connections: {_weights.Length}";
            if (input.Length != _weights.Length){
                throw new ArgumentException();
            }
        }

        public override string ToString()
        {
            var parameters = new Dictionary<string, string>()
            {
                { "Connections", _weights.Length.ToString() },
                { "Weights", string.Join("; ", _weights) },
                { "Bias", _bias.ToString() },
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return string.Join(Environment.NewLine, printable);
        }
    }
}