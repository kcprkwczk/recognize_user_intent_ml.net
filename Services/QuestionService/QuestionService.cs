using ConsoleApp4.Models;

namespace ConsoleApp4.Services
{
    using Newtonsoft.Json;
    using System.Collections.Generic;
    using System.IO;
    
    public class QuestionService : IQuestionService
    {
        private readonly string _filePath;
        
        public QuestionService(string filePath)
        {
            _filePath = filePath;
        }
        
        public QuestionsRoot LoadQuestions()
        {
            string json = File.ReadAllText(_filePath);
            return JsonConvert.DeserializeObject<QuestionsRoot>(json);
        }
        
        public List<IntentData> GetTrainingData(QuestionsRoot questionsRoot)
        {
            var trainingData = new List<IntentData>();
            foreach (var question in questionsRoot.Questions)
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
            return trainingData;
        }
    }
}