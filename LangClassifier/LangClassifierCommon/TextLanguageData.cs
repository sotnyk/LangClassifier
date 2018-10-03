using Microsoft.ML.Runtime.Api;

namespace LangClassifierCommon
{
    public class TextLanguageData
    {
        [Column("0")]
        public string Label { get; set; }

        [Column("1")]
        public string Text { get; set; }
    }
}
