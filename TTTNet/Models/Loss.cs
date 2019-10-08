using System;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public abstract class Loss
    {
        public abstract float Mean(float[] predictions, float[] trueValues);
        public abstract float[] ElementWise(float[] predictions, float[] trueValues);
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
        public override float Mean(float[] predictions, float[] trueValues)
        {
            var n = predictions.Length;
            return ElementWise(predictions, trueValues).Sum() / n;
        }

        public override float[] ElementWise(float[] predictions, float[] trueValues)
        {
            ValidateInput(predictions, trueValues);
            float[] loss = trueValues.Zip(predictions, (y, y_) => 0.5f * ((float) Math.Pow(y - y_, 2))).ToArray();
            return loss; // FIXME: Bad boy! Choose only one collection type!
        }
    }

}