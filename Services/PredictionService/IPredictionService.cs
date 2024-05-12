using ConsoleApp4.Models;
using Microsoft.ML;

namespace ConsoleApp4.Services
{
    public interface IPredictionService
    {
        IntentPrediction Predict(ITransformer model, string userStatement);
    }
}