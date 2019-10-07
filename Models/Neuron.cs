using System;
using System.Collections.Generic;
using System.Linq;

namespace stupidnet.Models
{
    public class Neuron : Unit
    {
        private float[] _weights;
        private float _bias;

        public override string ToString()
        {            
            var parameters = new Dictionary<string, String>()
            {
                {"Connections", this._weights.Length.ToString()},
                {"Weights", String.Join("; ", this._weights)},
                {"Bias", this._bias.ToString()},
            };
            var printable = parameters.Select(p => p.Key + ": " + p.Value);
            return String.Join(Environment.NewLine, printable);
        }

        public Neuron(int numberOfConnections)
        {
            if (numberOfConnections < 1)
            {
                throw new ArgumentException($"Number of connections: {numberOfConnections} can't be less than 1");
            }
            Random gen = new Random();
            float eps = .001f;

            this._weights = new float[numberOfConnections];
            this._bias = (float) (gen.NextDouble() + eps) % 1;

            for (int i = 0; i < numberOfConnections; ++i)
            {
                _weights[i] = (float) (gen.NextDouble() + eps) % 1;
            }
        }

        public float forwardPass(float[] input)
        {   
            if (input.Length != this._weights.Length){
                throw new ArgumentException($"Inputs size: {input.Length} doesn't match the number of connections: {this._weights.Length}");
            }
            var signals = new float[input.Length];
            for (int i = 0; i < signals.Length; ++i)
            {
                signals[i] = input[i] * this._weights[i];
            }
            var signal = this.activation(signals.Sum() + this._bias);
            return signal;
        } 

        public void backwardPass(float[] gradient)
        {
            if (gradient.Length != this._weights.Length){
                throw new ArgumentException($"Inputs size: {gradient.Length} doesn't match the number of connections: {this._weights.Length}");
            }

        }

        private float defLoss()
        {
            return 1f;
        }

        private float activation(float input)
        {
            // pretty much Sigmoid function
            var sigmoid = 1.0f / (1.0f + (float) Math.Exp(-input));
            // var relu = Math.Max(0, input);
            return sigmoid;
        }
    }
}