using System;
using System.Linq;

namespace TTT.Models
{
    public abstract class Loss: IFunction<Tuple<float[], float[]>, float[]>
    {
        public abstract float[] Calculate(Tuple<float[], float[]> outputsAndTargets);
        public abstract float[] Derivative(Tuple<float[], float[]> outputsAndTargets);

        protected void ValidateInput(Tuple<float[], float[]> outputsAndTargets)
        {
            var outputs = outputsAndTargets.Item1;
            var targets = outputsAndTargets.Item2;
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
        public override float[] Calculate(Tuple<float[], float[]> outputsAndTargets)
        {
            ValidateInput(outputsAndTargets);
            float[] results = outputsAndTargets.Item2.Zip(
                outputsAndTargets.Item1,
                (y, y_) => 0.5f * ((float)Math.Pow(y - y_, 2))
            ).ToArray();
            return results;
        }

        public override float[] Derivative(Tuple<float[], float[]> outputsAndTargets)
        {
            ValidateInput(outputsAndTargets);
            float[] results = outputsAndTargets.Item2.Zip(
                outputsAndTargets.Item1,
                (y, y_) => -(y - y_)
            ).ToArray();
            return results;
        }
    }
}