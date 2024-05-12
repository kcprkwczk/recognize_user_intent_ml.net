using ConsoleApp4.Models;

namespace ConsoleApp4.Services
{
    using Microsoft.ML;
    
    public class PredictionService : IPredictionService
    {
        private readonly MLContext _mlContext;
        private readonly float _confidenceThreshold;
        
        public PredictionService(MLContext mlContext, float confidenceThreshold)
        {
            _mlContext = mlContext;
            _confidenceThreshold = confidenceThreshold;
        }
        
        public IntentPrediction Predict(ITransformer model, string userStatement)
        {
            var predEngine = _mlContext.Model.CreatePredictionEngine<IntentData, IntentPrediction>(model);
            var prediction = predEngine.Predict(new IntentData { Text = userStatement });
            if (prediction.Score.Max() < _confidenceThreshold)
            {
                return null; 
            }
            return prediction;
        }
    }
}