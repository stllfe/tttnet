using System;
using System.Linq;

namespace TTT.Models
{
    public abstract class Loss
    {
        // Adding new losses is easy! Just implement these two guys:
        public abstract float[] ElementWise(float[] outputs, float[] targets);
        public abstract float[] Derivative(float[] outputs, float[] targets);

        public float Mean(float[] outputs, float[] targets)
        {
            var n = outputs.Length;
            return Sum(outputs, targets) / n;
        }

        public float Sum(float[] outputs, float[] targets)
        {
            return ElementWise(outputs, targets).Sum();
        }

        protected void ValidateInput(float[] outputs, float[] targets)
        {
            if (outputs.Length != targets.Length)
            {
                var error = "Can't calculate loss!\nSize mismatch: " +
                            $"{outputs.Length} and {targets.Length}";
                throw new ArgumentException(error);
            }
        }
    }

    public class MSELoss : Loss
    {
        public override float[] ElementWise(float[] outputs, float[] targets)
        {
            ValidateInput(outputs, targets);
            float[] results = targets.Zip(outputs,
                (y, y_) => 0.5f * ((float)Math.Pow(y - y_, 2))).ToArray();
            return results;
        }

        public override float[] Derivative(float[] outputs, float[] targets)
        {
            ValidateInput(outputs, targets);
            float[] results = targets.Zip(outputs,
                (y, y_) => -(y - y_)).ToArray();
            return results;
        }
    }
}