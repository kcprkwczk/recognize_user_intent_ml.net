using ConsoleApp4.Models;
using ConsoleApp4.Services;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
namespace ConsoleApp4
{
    class Program
    {
        static void Main(string[] args)
        {
            var services = new ServiceCollection();
            ConfigureServices(services);
            var serviceProvider = services.BuildServiceProvider();

            var questionService = serviceProvider.GetRequiredService<IQuestionService>();
            var modelTrainer = serviceProvider.GetRequiredService<IModelTrainer>();
            var predictionService = serviceProvider.GetRequiredService<IPredictionService>();
            var mlContext = serviceProvider.GetRequiredService<MLContext>();

            var questionsRoot = questionService.LoadQuestions();

            var trainedModels = new Dictionary<string, ITransformer>();

            while (true)
            {
                Console.Write("Wprowadź swoje dane: ");
                string userInput = Console.ReadLine();

                if (userInput?.ToLower() == ":q")
                    break;

                var inputParts = userInput?.Split(new[] { ',' }, 2);
                if (inputParts == null || inputParts.Length != 2)
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("Nieprawidłowy format danych wejściowych. Proszę użyć 'nazwa_modelu, twoja_wypowiedź'.");
                    Console.ResetColor();
                    continue;
                }

                string modelName = inputParts[0].Trim();
                string userStatement = inputParts[1].Trim();

                if (!trainedModels.TryGetValue(modelName, out ITransformer trainedModel))
                {
                    var modelQuestionData = questionsRoot.Questions.FirstOrDefault(q => q.Question == modelName);
                    if (modelQuestionData == null)
                    {
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.WriteLine($"Nie znaleziono modelu '{modelName}'.");
                        Console.ResetColor();
                        continue;
                    }

                    var trainingData = questionService.GetTrainingData(new QuestionsRoot { Questions = new List<QuestionData> { modelQuestionData } });
                    
                    if (trainingData == null || trainingData.Count == 0)
                    {
                        Console.ForegroundColor = ConsoleColor.Yellow;
                        Console.WriteLine($"Brak danych treningowych dla modelu '{modelName}'.");
                        Console.ResetColor();
                        continue;
                    }

                    IDataView trainingDataView = mlContext.Data.LoadFromEnumerable(trainingData);
                    trainedModel = modelTrainer.TrainModel(mlContext, trainingDataView);

                    trainedModels[modelName] = trainedModel;
                }

                IntentPrediction prediction = predictionService.Predict(trainedModel, userStatement);

                if (prediction != null)
                {
                    Console.ForegroundColor = ConsoleColor.Green;
                    Console.WriteLine($"Zamiar dla '{modelName}': {prediction.Prediction}");
                    Console.ResetColor();
                }
                else
                {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine("-1");
                    Console.ResetColor();
                }
            }
        }

        private static void ConfigureServices(IServiceCollection services)
        {
            services.AddSingleton<MLContext>()
                    .AddSingleton<IQuestionService, QuestionService>(s => new QuestionService("data.json"))
                    .AddSingleton<IModelTrainer, ModelTrainer>()
                    .AddSingleton<IPredictionService, PredictionService>(s =>
                        new PredictionService(s.GetRequiredService<MLContext>(), 0.6f));
        }
    }
}
