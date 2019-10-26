using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public static class DataManipulator
    {
        public static string[] ReadData(string path)
        {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Can't find {path}");
            }
            string[] lines = File.ReadAllLines(path);
            ValidateDataStructure(lines);
            return lines;
        }

        private static void ValidateDataStructure(string[] lines)
        {
            int[] lengths = lines.Select(l => l.Length).ToArray();
            Array.Sort(lengths);

            if (lengths.First() != lengths.Last())
            {
                var error = "Data is not properly structured\n" +
                            "Number of characters per line is different.";
                throw new Exception(error);
            }
        }

        public static float[][] LabelEncode(string[] data)
        {
            ValidateDataStructure(data);
            char[] uniques = string.Concat(data).Distinct().ToArray();
            int[] labels = Enumerable.Range(0, uniques.Length).ToArray();
            Func<char, int> encode = (ch) => labels[Array.IndexOf(uniques, ch)];

            int n = data.GetLength(0);
            int m = data[0].Length;
            float[][] encoded = new float[n][];

            for (int i = 0; i < n; ++i)
            {
                encoded[i] = new float[m];
                encoded[i] = data[i].Select(ch => (float) encode(ch)).ToArray();
            }

            return encoded;
        }
    }
}
