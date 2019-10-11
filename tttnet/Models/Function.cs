using System;

namespace TTT.Models
{
    public interface IFunction
    {
        float Calculate(float x);
        float Derivative(float x);
    }

    public class Sigmoid : IFunction
    {
        public float Calculate(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public float Derivative(float x)
        {
            x = Calculate(x);
            return x * (1 - x);
        }
    }
}