using ConsoleTables;
using NaiveBayessClasificator;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace App
{
    internal class Program
    {
        static void Main(string[] args)
        {
            var filePath = @"..\..\..\Data\iris.data";

            var dataLoader = new CsvDataLoader();
            var data = dataLoader.LoadData(filePath);

            var (values, classes) = SplitValuesFromClasses(data);
            var (trainingData, trainingClasses, testingData, testingClasses) = GetTrainingAndTestingData(values, classes, 0.6);

            var (confusionMatrix, classMap) = EvaluatePrediction(trainingData, trainingClasses, testingData, testingClasses);

            PrintConfusionMatrix(classMap, confusionMatrix);
        }

        private static (int[][], Dictionary<string, int>) EvaluatePrediction(double[][] trainingData, string[] trainingClasses, double[][] testingData, string[] testingClasses)
        {
            var bayes = new NaiveBayes();
            bayes.Train(trainingData, trainingClasses);
                                                              
            var classMap = new Dictionary<string, int>();   
            var distinctClasses = trainingClasses.Union(testingClasses).Distinct().ToArray();

            var numberOfDistinctClasses = distinctClasses.Count();
            var confusionMatrix = new int[numberOfDistinctClasses][];
            
            for (int i = 0; i < numberOfDistinctClasses; i++)
            {
                classMap.Add(distinctClasses[i], i);
                confusionMatrix[i] = new int[numberOfDistinctClasses];
            }

            for (int i = 0; i < testingData.Length; i++)
            {
                var testingDataInstance = testingData[i];
                var testingClassInstance = testingClasses[i];

                var predictionProbabilities = bayes.Fit(testingDataInstance);

                var maxPredictionValue = predictionProbabilities.Values.Max();

                var prediction = predictionProbabilities.Where(x => x.Value == maxPredictionValue).Select(x => x.Key).First();

                var testingClassIndex = classMap[testingClassInstance];
                var predictedClassIndex = classMap[prediction];

                confusionMatrix[testingClassIndex][predictedClassIndex] += 1;
            }

            return (confusionMatrix, classMap);
        }

        private static void PrintConfusionMatrix(Dictionary<string, int> classMap, int[][] confusionMatrix)
        {
            var reverseClassMap = new Dictionary<int, string>();
            foreach (var @class in classMap)
            {
                reverseClassMap.Add(@class.Value, @class.Key);
            }

            var classes = new List<string> { "-" };
            classes.AddRange(classMap.OrderBy(x => x.Value).Select(x => x.Key));
            var table = new ConsoleTable(classes.ToArray());

            for (int rowIndex = 0; rowIndex < confusionMatrix.Length; rowIndex++)
            {
                var tableRow = new List<string>
                {
                    reverseClassMap[rowIndex]
                };

                var row = confusionMatrix[rowIndex];
                for (int columnIndex = 0; columnIndex < row.Length; columnIndex++)
                {
                    var value = row[columnIndex];
                    tableRow.Add(value.ToString());
                }

                table.AddRow(tableRow.ToArray());
            }

            table.Write(Format.Alternative);
        }

        private static (double[][], string[], double[][], string[]) GetTrainingAndTestingData(double[][] values, string[] classes, double requiredTrainingDataSize)
        {
            var random = new Random();

            var registeredTrainingInstances = new HashSet<int>() { -1 };

            var trainingData = new List<double[]>();
            var trainingClasses = new List<string>();
            var testingData = new List<double[]>();
            var testingClasses = new List<string>();

            for (int i = 0; i < values.Length * requiredTrainingDataSize; i++)
            {
                var instanceId = -1;
                while (registeredTrainingInstances.Contains(instanceId))
                {
                    instanceId = random.Next(0, values.Length - 1);
                }

                registeredTrainingInstances.Add(instanceId);
                trainingData.Add(values[instanceId]);
                trainingClasses.Add(classes[instanceId]);
            }

            for (int i = 0; i < values.Length; i++)
            {
                if (!registeredTrainingInstances.Contains(i))
                {
                    testingData.Add(values[i]);
                    testingClasses.Add(classes[i]);
                }
            }


            return (trainingData.ToArray(), trainingClasses.ToArray(), testingData.ToArray(), testingClasses.ToArray());
        }

        private static (double[][], string[]) SplitValuesFromClasses(List<List<string>> data)
        {
            var valueData = new double[data.Count][];
            var classes = new string[data.Count];

            for (int instance = 0; instance < data.Count; instance++)
            {
                classes[instance] = data[instance][^1];

                var numberOfValueAttributes = data[instance].Count - 1;
                valueData[instance] = new double[numberOfValueAttributes];

                for (int attribute = 0; attribute < numberOfValueAttributes; attribute++)
                {
                    valueData[instance][attribute] = double.Parse(data[instance][attribute], CultureInfo.InvariantCulture);
                }
            }

            return (valueData, classes);
        }
    }
}
