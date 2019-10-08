namespace stupidnet.Models
{
    public interface IFunction<T>
    {
        float Calculate(T args);
    }
}