using System;
using System.Collections.Generic;

namespace NaiveBayessClasificator
{
    public class NaiveBayes
    {
        private double[][] _trainingData;
        private string[] _trainingClasses;

        private Dictionary<string, int> _classMap;
        private int[] _classOccurrenceCount;
        private double[][] _means;
        private double[][] _variances;
        private double[] _classProbabilities;

        private int NumberOfClasses => _classMap.Count;
        private int NumberOfInstances => _trainingData.Length;
        private int NumberOfAttributes => _trainingData[0].Length;

        public NaiveBayes() { }

        public void Train(double[][] triningData, string[] trainingClasses)
        {
            _trainingData = triningData;
            _trainingClasses = trainingClasses;

            FillClassMap();
            FillClassOccurrenceCount();
            ComputeMeans();
            ComputeVariances();
            ComputeClassProbabilities();
        }

        public Dictionary<string, double> Fit(double[] unknownInstance)
        {
            var conditionalProbabilities = ComputeConditionalProbabilities(unknownInstance);
            var evidenceTerms = CopmuteEvidenceTerms(conditionalProbabilities);
            var predictedProbabilities = GetPredictedProbabilities(evidenceTerms);

            return predictedProbabilities;
        }

        private Dictionary<string, double> GetPredictedProbabilities(double[] evidenceTerms)
        {
            double sumEvidence = 0.0;
            foreach (var classMapInstance in _classMap)
            {
                var classIndex = classMapInstance.Value;
                sumEvidence += evidenceTerms[classIndex];
            }

            var predictedProbabilities = new Dictionary<string, double>();

            foreach (var classMapInstance in _classMap)
            {
                var classIndex = classMapInstance.Value;
                var predictedProbability = evidenceTerms[classIndex] / sumEvidence;
                predictedProbabilities.Add(classMapInstance.Key, predictedProbability);
            }

            return predictedProbabilities;
        }

        private void FillClassMap()
        {
            _classMap = new Dictionary<string, int>();
            
                var index = 0;
            foreach (var trainingClass in _trainingClasses)
            {
                if (!_classMap.ContainsKey(trainingClass))
                {
                    _classMap.Add(trainingClass, index);
                    index++;
                }
            }
        }

        private void FillClassOccurrenceCount()
        {
            _classOccurrenceCount = new int[NumberOfClasses];
            
            foreach(var trainingClass in _trainingClasses)
            {
                var classIndex = _classMap[trainingClass];
                _classOccurrenceCount[classIndex] += 1;
            }
        }
        
        private void ComputeMeans()
        {
            _means = new double[NumberOfClasses][];
            
            for (int i = 0; i < NumberOfClasses; i++)
            {
                _means[i] = new double[NumberOfAttributes];
            }

            for (int i = 0; i < NumberOfInstances; i++)
            {
                var @class = _trainingClasses[i];
                int classIndex = _classMap[@class];

                for (int attribute = 0; attribute < NumberOfAttributes; attribute++)
                {
                    _means[classIndex][attribute] += _trainingData[i][attribute];
                }
            }

            foreach (var classMapInstance in _classMap)
            {
                var classIndex = classMapInstance.Value;
                
                for (int attribute = 0; attribute < NumberOfAttributes; attribute++)
                {
                    _means[classIndex][attribute] /= _classOccurrenceCount[classIndex];
                }
            }
        }

        private void ComputeVariances()
        {
            _variances = new double[NumberOfClasses][];

            for (int i = 0; i < NumberOfClasses; i++)
            {
                _variances[i] = new double[NumberOfAttributes];
            }

            for (int i = 0; i < NumberOfInstances; i++)
            {
                var @class = _trainingClasses[i];
                int classIndex = _classMap[@class];

                for (int attribute = 0; attribute < NumberOfAttributes; attribute++)
                {
                    double attributeValue = _trainingData[i][attribute];
                    double attributeMean = _means[classIndex][attribute];
                    
                    _variances[classIndex][attribute] += (attributeValue - attributeMean) * (attributeValue - attributeMean);
                }
            }

            foreach(var classMapInstance in _classMap)
            {
                var classIndex = classMapInstance.Value;
                
                for (int attribute = 0; attribute < NumberOfAttributes; attribute++)
                {
                    _variances[classIndex][attribute] /= _classOccurrenceCount[classIndex] - 1;
                }
            }
        }

        private void ComputeClassProbabilities()
        {
            _classProbabilities = new double[NumberOfClasses];
            foreach(var classMapInstance in _classMap)
            {
                var classIndex = classMapInstance.Value;
                _classProbabilities[classIndex] = ((double)_classOccurrenceCount[classIndex]) / NumberOfInstances;
            }
        }

        private double[][] ComputeConditionalProbabilities(double[] unknownInstance)
        {
            var conditionalProbabilities = new double[NumberOfClasses][];
            for (int i = 0; i < NumberOfClasses; i++)
            {
                conditionalProbabilities[i] = new double[NumberOfAttributes];
            }

            foreach (var classMapInstance in _classMap)
            {
                var classIndex = classMapInstance.Value;

                for (int attribute = 0; attribute < NumberOfAttributes; attribute++)
                {
                    double attributeMean = _means[classIndex][attribute];
                    double attributeVariance = _variances[classIndex][attribute];
                    double unknownInstanceAttributeValue = unknownInstance[attribute];

                    conditionalProbabilities[classIndex][attribute] = ProbabilityDensityFunction(attributeMean, attributeVariance, unknownInstanceAttributeValue);
                }
            }

            return conditionalProbabilities;
        }

        private double[] CopmuteEvidenceTerms(double[][] conditionalClassProbabilities)
        {
            var evidenceTerms = new double[NumberOfClasses];

            foreach(var classMapInstance in _classMap)
            {
                var classIndex = classMapInstance.Value;

                evidenceTerms[classIndex] = _classProbabilities[classIndex];
                for (int attribute = 0; attribute < NumberOfAttributes; attribute++)
                {
                    evidenceTerms[classIndex] *= conditionalClassProbabilities[classIndex][attribute];
                }
            }

            return evidenceTerms;
        }

        private static double ProbabilityDensityFunction(double mean, double variance, double value)
        {
            double left = 1.0 / Math.Sqrt(2 * Math.PI * variance);
            double right = Math.Exp(-(value - mean) * (value - mean) / (2 * variance));
            return left * right;
        }
    }
}
