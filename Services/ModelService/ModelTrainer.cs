using ConsoleApp4.Models;

namespace ConsoleApp4.Services
{
    using Microsoft.ML;

    public class ModelTrainer : IModelTrainer
    {
        public ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView)
        {
            var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(IntentData.Label))
                .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(IntentData.Text)))
                .AppendCacheCheckpoint(mlContext);

            var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features")
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
            
            var trainingPipeline = dataProcessPipeline.Append(trainer);
            return trainingPipeline.Fit(trainingDataView);
            
            
        }
    }
}