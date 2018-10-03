using Microsoft.ML.Runtime.Api;

namespace LangClassifierCommon
{
    public class LanguagePrediction
    {
        [ColumnName("PredictedLabel")]
        public string Label { get; set; }

        [ColumnName("Score")]
        public float[] Score;
    }
}
