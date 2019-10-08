using System;
using stupidnet.Models;
using System.Linq;

namespace stupidnet
{
    class Program
    {
        static void Main(string[] args)
        {
            float[] inputs = new float[]{0.1f, .34f, 3.1f, .5f, 3f, 2f};
            int numberOfConnections = inputs.Length;
            Neuron neu = new Neuron(numberOfConnections);
            Console.WriteLine(neu.ForwardPass(inputs));

            Console.WriteLine(Environment.NewLine);
            Net net = new Net(
                name: "TikTacToe", 
                inputSize: numberOfConnections, 
                outputSize: 2,
                numberOfHiddenLayers: 2,
                hiddenLayerSize: 4);
            Console.WriteLine(net.ToString());

            var results = net.ForwardPass(inputs);
            Console.WriteLine(String.Join(", ", results));
        }
    }
}
