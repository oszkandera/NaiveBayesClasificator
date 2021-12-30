using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace App
{
    public class CsvDataLoader
    {
        private string _determinator;
        
        public CsvDataLoader()
        {
            _determinator = ",";
        }

        public CsvDataLoader(string determinator) : this()
        {
            _determinator = determinator;
        }

        public List<List<string>> LoadData(string path)
        {
            var data = new List<List<string>>();
            using (var file = new StreamReader(path))
            {
                string line;
                while ((line = file.ReadLine()) != null)
                {
                    string[] splitLine = line.Split(_determinator).ToArray();

                    var lineItems = new List<string>(splitLine.Length);
                    lineItems.AddRange(splitLine);
                    data.Add(lineItems);
                }
            }

            return data;
        }
    }
}
