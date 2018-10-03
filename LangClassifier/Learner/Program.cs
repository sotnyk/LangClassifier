using Microsoft.ML;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Data;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using System;
using System.IO;

namespace Learner
{
    class Program
    {

        private static string AppPath => Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
        private static string DatasetsPath => Path.Combine(AppPath, "datasets");
        private static string TrainDataPath => Path.Combine(DatasetsPath, "languages-train.csv");
        private static string TestDataPath => Path.Combine(DatasetsPath, "languages-test.csv");
        private static string ModelPath => Path.Combine(DatasetsPath, "LanguageModel.zip");


        static void Main(string[] args)
        {
            Train();
            Console.ReadLine();
        }

        public static void Train()
        {
            using (var env = new LocalEnvironment(1974))
            {
                /*env.AddListener((messageSource, message) => 
                    Console.WriteLine($"{messageSource.ShortName}: {message.Message} ({message.Kind})"));*/
                env.AddListener(ConsoleLogger);

                var classification = new MulticlassClassificationContext(env);

                var reader = TextLoader.CreateReader(env, ctx => (
                      Sentence: ctx.LoadText(1),
                      Label: ctx.LoadText(0)
                    ),
                    separator: ',');

                var trainData = reader.Read(new MultiFileSource(TrainDataPath));

                var pipeline = reader.MakeNewEstimator()
                    .Append(r => (
                        Label: r.Label.ToKey(),
                        Features: r.Sentence.FeaturizeText()))
                    .Append(r => (
                        r.Label,
                        Predictions: classification.Trainers.Sdca(r.Label, r.Features)
                        ))
                    .Append(r => r.Predictions.predictedLabel.ToValue());

                Console.WriteLine("=============== Training model ===============");

                var model = pipeline.Fit(trainData).AsDynamic;

                using (var fs = new FileStream(ModelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                    model.SaveTo(env, fs);

                Console.WriteLine("=============== End training ===============");
                Console.WriteLine("The model is saved to {0}", ModelPath);
            }
        }

        private static void ConsoleLogger(IMessageSource source, ChannelMessage message)
        {
            try
            {
                Console.WriteLine($"{source.ShortName}: {message.Message} ({message.Kind})");
            }
            catch { }
        }
    }
}
