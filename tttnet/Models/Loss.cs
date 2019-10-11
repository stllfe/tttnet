using System;
using System.Linq;

namespace TTT.Models
{
    public abstract class Loss
    {
        // Adding new losses is easy - implement these two guys:
        public abstract float[] ElementWise(float[] predictions, float[] trueValues);
        public abstract float[] Derivative(float[] predictions, float[] trueValues);

        public float Mean(float[] predictions, float[] trueValues)
        {
            var n = predictions.Length;
            return Sum(predictions, trueValues) / n;
        }

        public float Sum(float[] predictions, float[] trueValues)
        {
            return ElementWise(predictions, trueValues).Sum();
        }

        protected void ValidateInput(float[] predictions, float[] trueValues)
        {
            if (predictions.Length != trueValues.Length)
            {
                var error = "Can't calculate loss!\n" +
                            $"Size mismatch: {predictions.Length} " +
                            $"and {trueValues.Length}";
                throw new ArgumentException(error);
            }
        }
    } 

    public class MSELoss : Loss
    {
        public override float[] ElementWise(float[] predictions, float[] trueValues)
        {
            ValidateInput(predictions, trueValues);
            float[] loss = trueValues.Zip(predictions, (y, y_) => 0.5f * ((float) Math.Pow(y - y_, 2))).ToArray();
            return loss;
        }

        public override float[] Derivative(float[] predictions, float[] trueValues)
        {
            ValidateInput(predictions, trueValues);
            return trueValues.Zip(predictions, (y, y_) => -(y - y_)).ToArray();
        }

    }

}