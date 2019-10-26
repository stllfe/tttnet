namespace TTT.Models
{
    public enum Direction
    {
        Forward,
        Backward
    }

    public interface INeural<T>
    {
        T ForwardPass(float[] input);
        float[] BackwardPass(T gradient);

    }

    // Module consists of units.
    // It always passes a collection of signals between all its units.
    public abstract class Module : INeural<float[]>
    {
        public abstract float[] BackwardPass(float[] gradient);
        public abstract float[] ForwardPass(float[] input);
        protected abstract void ValidateInput(float[] input, Direction direction);
    }

    // Unit is a standalone neuron.
    // It returns its signal on a forward pass, 
    // and sends weighted error through all its connections on a backward pass.
    public abstract class Unit : INeural<float>
    {
        public abstract float[] BackwardPass(float neuronError);
        public abstract float ForwardPass(float[] input);
        protected abstract void ValidateInput(float[] input);
    }
}
