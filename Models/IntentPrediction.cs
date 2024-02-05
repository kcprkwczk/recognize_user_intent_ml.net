namespace ConsoleApp4.Models
{
    using Microsoft.ML.Data;

    public class IntentPrediction
    {
        [ColumnName("PredictedLabel")]
        public string Prediction { get; set; }
        public float[] Score { get; set; }
    }
}