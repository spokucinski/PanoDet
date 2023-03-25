using DatasetParser.Formats.COCO;
using DatasetParser.Formats.WiMR;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;

namespace DatasetParser
{
    internal class Program
    {
        private static void Main(string[] args)
        {
            RunArgs runArgs = LoadRunArgs(args);

            List<AnnotationFile> loadedAnnotationFiles = LoadAnnotationsFiles(runArgs.InputPath);

            foreach (AnnotationFile annotationFile in loadedAnnotationFiles)
            {
                WmirDocument loadedWmirDocument = JsonConvert.DeserializeObject<WmirDocument>(annotationFile.Content);
                CocoDocument parsedCocoDocument = ParseWmirToCoco(loadedWmirDocument);
                string serializedCocoDocumentContent = JsonConvert.SerializeObject(parsedCocoDocument);
                File.WriteAllText($"{runArgs.OutputPath}\\{annotationFile.FileName}", serializedCocoDocumentContent);
            }
        }

        private static CocoDocument ParseWmirToCoco(WmirDocument wmirDocument)
        {
            CocoDocument cocoDocument = new CocoDocument();

            cocoDocument.Info = CreateInfo();
            cocoDocument.Licenses = CreateLicenses();
            cocoDocument.Categories = ParseCategories();
            cocoDocument.Images = ParseImages(wmirDocument);
            cocoDocument.Annotations = ParseAnnotations(wmirDocument);

            return cocoDocument;
        }

        private static List<CocoLicense> CreateLicenses()
        {
            return new List<CocoLicense>()
            {
                new CocoLicense()
                {
                    Id = 0,
                    Url = "",
                    Name = ""
                }
            };
        }

        private static CocoInfo CreateInfo()
        {
            return new CocoInfo()
            {
                Year = DateTime.UtcNow.Year.ToString(),
                Version = "1",
                Description = "Dataset parsed from the 'What's in My Room' dataset structure to the COCO format",
                Contributor = "Sebastian Pokuciński, Silesian University of Technology, Gliwice, Poland",
                Url = "https://universe.roboflow.com/polsl-7aaeg/simpano/dataset/1",
                CreationDate = DateTime.UtcNow.ToLongDateString()
            };
        }

        private static List<CocoAnnotation> ParseAnnotations(WmirDocument wmirDocument)
        {
            List<CocoAnnotation> parsedAnnotations = new List<CocoAnnotation>();
            foreach (WmirAnnotation annotation in wmirDocument.Annotations)
            {
                parsedAnnotations.Add(new CocoAnnotation()
                {
                    Id = Int32.Parse(annotation.Id),
                    ImageId = annotation.ImageId,
                    CategoryId = WmirCategoriesDictionary.WmirCategories.GetValueOrDefault(annotation.Category),
                    Area = annotation.Area,
                    Segmentation = new CocoSegmentation() { Counts = annotation.Segmentation.Counts, Size = annotation.Segmentation.Size },
                    BBox = annotation.Bbox,
                    IsCrowd = 1
                });
            }

            return parsedAnnotations;
        }

        private static List<CocoImage> ParseImages(WmirDocument wmirDocument)
        {
            List<CocoImage> parsedImages = new List<CocoImage>();
            foreach (WmirImage image in wmirDocument.Images)
            {
                parsedImages.Add(new CocoImage()
                {
                    Id = image.Id,
                    Width = image.Width,
                    Height = image.Height,
                    FileName = image.FileName + ".png",
                    DateCaptured = "2023-01-01T00:00:00+00:00",
                    License = 0,
                    CocoUrl = "",
                    FlickrUrl = ""
                });
            }

            return parsedImages;
        }

        private static List<CocoCategory> ParseCategories()
        {
            List<CocoCategory> parsedCategories = new List<CocoCategory>();
            foreach (KeyValuePair<string, int> category in WmirCategoriesDictionary.WmirCategories)
            {
                parsedCategories.Add(new CocoCategory()
                {
                    Id = category.Value,
                    Name = category.Key,
                    SuperCategory = ""
                });
            }

            return parsedCategories;
        }

        private static RunArgs LoadRunArgs(string[] args)
        {
            return new RunArgs()
            {
                InputPath = args[0],
                OutputPath = args[1]
            };
        }

        private static List<AnnotationFile> LoadAnnotationsFiles(string inputFolderPath)
        {
            string[] inputFilePaths = Directory.GetFiles(inputFolderPath);
            List<AnnotationFile> loadedFiles = new List<AnnotationFile>();
            foreach (string inputFilePath in inputFilePaths)
            {
                loadedFiles.Add(new AnnotationFile()
                {
                    FileName = Path.GetFileName(inputFilePath),
                    Content = File.ReadAllText(inputFilePath)
                });
            }

            return loadedFiles;
        }
    }
}