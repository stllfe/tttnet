using System;

namespace TTT.Models
{
    public interface IFunction<T1, T2>
    {
        T2 Calculate(T1 data);
        T2 Derivative(T1 data);
    }

    public abstract class Function: IFunction<float, float>
    {
        public abstract float Calculate(float x);
        public abstract float Derivative(float x);
    }

    public class Sigmoid : Function
    {
        public override float Calculate(float x)
        {
            return 1.0f / (1.0f + (float)Math.Exp(-x));
        }

        public override float Derivative(float x)
        {
            float f = Calculate(x);
            return f * (1 - f);
        }
    }
}