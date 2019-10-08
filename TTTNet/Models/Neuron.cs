using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public class Neuron : Unit
    {
        private float[] _weights;
        private float _bias;
        private bool _activation;

        public override string ToString()
        {            
            var parameters = new Dictionary<string, String>()
            {
                {"Connections", _weights.Length.ToString()},
                {"Weights", String.Join("; ", _weights)},
                {"Bias", _bias.ToString()},
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return String.Join(Environment.NewLine, printable);
        }

        public Neuron(int numberOfConnections, bool activation = true)
        {
            if (numberOfConnections < 1)
            {
                throw new ArgumentException($"Number of connections: {numberOfConnections} can't be less than 1");
            }
            Random gen = new Random();
            float eps = .001f;

            this._weights = new float[numberOfConnections];
            this._bias = (float)(gen.NextDouble() + eps) % 1;
            this._activation = activation;

            for (int i = 0; i < numberOfConnections; ++i)
            {
                _weights[i] = (float)(gen.NextDouble() + eps) % 1;
            }
        }

        public override float ForwardPass(float[] input)
        {   
            ValidateInput(input);
            var signals = new float[input.Length];
            for (int i = 0; i < signals.Length; ++i)
            {
                signals[i] = input[i] * _weights[i];
            }
            var signal = signals.Sum() + _bias;
            return _activation ? Activation(signal) : signal;
        } 

        public override void BackwardPass(float[] gradient)
        {
            ValidateInput(gradient);
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

        private float Activation(float input)
        {
            // pretty much Sigmoid function
            var sigmoid = 1.0f / (1.0f + (float) Math.Exp(-input));
            // var relu = Math.Max(0, input);
            return sigmoid;
        }
    }

}