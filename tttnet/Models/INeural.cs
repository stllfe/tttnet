using System;
namespace TTT.Models
{
    public interface INeural<T>
    {
        T ForwardPass(float[] input);
        T BackwardPass(float[] gradient);
    }
}
