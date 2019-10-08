using System;
using System.Collections.Generic;

namespace stupidnet.Models
{
    public abstract class Loss
    {
        //public abstract float Calculate(float[] predictions, float[] trueValues);
        protected void ValidateInput(float[] predictions, float[] trueValues)
        {
            if(predictions.Length != trueValues.Length)
            {
                throw  new ArgumentException("Can't calculate loss!\n" +
                $"Size mismatch: {predictions.Length} and {trueValues.Length}");
            }
        }
    }

    public class MSELoss : Loss
    {
        public float Calculate(float[] predictions, float[] trueValues)
        {
            ValidateInput(predictions, trueValues);
            var n = predictions.Length;
            var sum = 0f;
            for (int i = 0; i < n; ++i)
            {
                sum += (float) Math.Pow((trueValues[i] - predictions[i]), 2);
            }
            return sum / n;
            // May be also handy to divide by 2 because of derivative
        }
    }
}