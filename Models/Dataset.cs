using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;

namespace TTT.Models
{
    public class Dataset
    {
        private string[] _rawData;
        private float[][] _data;
        private float[][] _labels;
        private int _sideSize;

        public Dictionary<char, int> EncodingsMap { get; }
        public int Length { get => _data.GetLength(0) / _sideSize; }

        public Dataset(string pathToData, string pathToLabels, int sideSize) {
            _sideSize = sideSize;
            if (sideSize < 1)
            {
                throw new ArgumentOutOfRangeException("sideSize should be > 0!");
            }

            // Read and validate the raw data
            _rawData = ReadData(pathToData);
            if (_rawData.Length % sideSize != 0)
            {
                var error = $"Data doesn't fit the provided board size: {sideSize}x{sideSize}";
                throw new Exception(error);
            }

            EncodingsMap = CreateEncodingsMap(_rawData);

            // Read and validate the labels
            _labels = ConvertLabels(ReadData(pathToLabels));
            _data = EncodeDataset(_rawData, _sideSize, EncodingsMap);
            
        }

        public Tuple<float[], float[]> GetItem(int index)
        {
            return new Tuple<float[], float[]>(_data[index], _labels[index]);
        }

        private static string[] ReadData(string path) {
            if (!File.Exists(path))
            {
                throw new FileNotFoundException($"Can't find {path}");
            }
            string[] lines = File.ReadAllLines(path);
            ValidateDataStructure(lines);
            return lines;
        }

        private static float[][] ConvertLabels(string[] rawLabels)
        {
            return rawLabels
            .Select(l => l.Where(ch => char.IsDigit(ch))
            .Select(ch => float.Parse(ch.ToString()))
            .ToArray()).ToArray();
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

        private static Dictionary<char, int> CreateEncodingsMap(string[] rawData)
        {
            char[] uniques = string.Concat(rawData).Distinct().ToArray();
            Dictionary<char, int> encoded = new Dictionary<char, int>();

            int idx = 0;
            foreach (var ch in uniques) {
                encoded.Add(ch, idx);
                idx++;
            }

            return encoded;
        }

        private static float[][] EncodeDataset(string[] rawData, int sideSize, Dictionary<char, int> encodingsMap)
        {
            var numberOfExamples = rawData.Length / sideSize;
            var inputSize = sideSize * sideSize;

            float[][] data = new float[numberOfExamples][];

            for (int i = 0; i < numberOfExamples; ++i)
            {
                data[i] = new float[inputSize];
                for (int j = 0; j < inputSize; ++j) 
                {
                    var value = encodingsMap[rawData[i + (j / sideSize)][j % sideSize]];
                    data[i][j] = value;
                }
            }

            return data;
        }
    }
}
