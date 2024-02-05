using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Newtonsoft.Json;

public class QuestionsRoot
{
    public List<QuestionData> Questions { get; set; }
}

public class IntentData
{
    public string Text { get; set; }
    public string Label { get; set; }
}

public class IntentPrediction
{
    [ColumnName("PredictedLabel")]
    public string Prediction { get; set; }
    public float[] Score { get; set; }
}

public class QuestionData
{
    public string Question { get; set; }
    public List<string> Intentions { get; set; }
    public Dictionary<string, List<string>> Examples { get; set; }
}

class Program
{
    private static readonly float ConfidenceThreshold = 0.6f;

    static void Main(string[] args)
    {
        var mlContext = new MLContext();

        string json = File.ReadAllText("data.json");
        var questionsRoot = JsonConvert.DeserializeObject<QuestionsRoot>(json);
        var questionsData = questionsRoot.Questions;

        // Print available question IDs
        Console.WriteLine("Available question IDs:");
        foreach (var question in questionsData)
        {
            Console.WriteLine(question.Question);
        }

        var trainingData = new List<IntentData>();
        foreach (var question in questionsData)
        {
            foreach (var intent in question.Intentions)
            {
                if (question.Examples.TryGetValue(intent, out var examples))
                {
                    foreach (var example in examples)
                    {
                        trainingData.Add(new IntentData { Text = example, Label = intent });
                    }
                }
            }
        }

        IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);

        var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(IntentData.Label))
            .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(IntentData.Text)))
            .AppendCacheCheckpoint(mlContext);

        var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        var trainingPipeline = dataProcessPipeline.Append(trainer);
        var trainedModel = trainingPipeline.Fit(trainingDataView);

        Console.WriteLine("\nModel is trained. Type 'exit' to quit.");
        Console.WriteLine("Enter the question ID and user statement separated by a comma:");

        string input;
        while ((input = Console.ReadLine()) != null && input.ToLower() != "exit")
        {
            var inputs = input.Split(',');
            if (inputs.Length != 2)
            {
                Console.WriteLine("Invalid input. Please provide the question ID and user statement separated by a comma.");
                continue;
            }

            string questionId = inputs[0].Trim();
            string userStatement = inputs[1].Trim();

            // Find the corresponding question data
            var questionData = questionsData.FirstOrDefault(q => q.Question == questionId);
            if (questionData == null)
            {
                Console.WriteLine("Question ID not found.");
                continue;
            }

            // Filter the training data to include only the examples for the specified question ID
            var filteredTrainingData = trainingData.Where(td => questionData.Intentions.Contains(td.Label)).ToList();
            IDataView filteredTrainingDataView = mlContext.Data.LoadFromEnumerable(filteredTrainingData);

            // Retrain the model using filtered data
            var filteredModel = trainingPipeline.Fit(filteredTrainingDataView);

            // Create prediction engine using the filtered model
            var predEngine = mlContext.Model.CreatePredictionEngine<IntentData, IntentPrediction>(filteredModel);
            var prediction = predEngine.Predict(new IntentData { Text = userStatement });

            // Get the index of the maximum score
            int maxScoreIndex = prediction.Score.ToList().IndexOf(prediction.Score.Max());

            // Check if the predicted label's score is above the threshold
            if (prediction.Score[maxScoreIndex] < ConfidenceThreshold)
            {
                Console.WriteLine("The intent is unclear or confidence is too low.");
            }
            else
            {
                Console.WriteLine($"Predicted Intention: {prediction.Prediction}");
            }
        }
    }
}
