namespace ConsoleApp4.Services
{
    using Microsoft.ML;
    
    public interface IModelTrainer
    {
        ITransformer TrainModel(MLContext mlContext, IDataView trainingDataView);
    }
}