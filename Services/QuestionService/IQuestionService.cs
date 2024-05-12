using ConsoleApp4.Models;

namespace ConsoleApp4.Services
{
    public interface IQuestionService
    {
        QuestionsRoot LoadQuestions();
        List<IntentData> GetTrainingData(QuestionsRoot questionsRoot);
    }
}