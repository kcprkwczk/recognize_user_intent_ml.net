namespace ConsoleApp4.Models
{
    using System.Collections.Generic;

    public class QuestionData
    {
        public string Question { get; set; }
        public List<string> Intentions { get; set; }
        public Dictionary<string, List<string>> Examples { get; set; }
    }
}

