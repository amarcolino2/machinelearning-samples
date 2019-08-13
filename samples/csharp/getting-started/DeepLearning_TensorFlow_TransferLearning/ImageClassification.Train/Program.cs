using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using ImageClassification.DataModels;
using Microsoft.ML;
using Microsoft.ML.Transforms;
using static Microsoft.ML.DataOperationsCatalog;
using System.Linq;
using Microsoft.ML.Data;
using ImageClassification.Train.Utils;

namespace ImageClassification.Train
{
    public class Program
    {
        static void Main(string[] args)
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string imagesDownloadFolder = Path.Combine(assetsPath, "inputs", "images");
            string imagesFolder = Path.Combine(imagesDownloadFolder, "flower_photos_small_set"); // "flower_photos"
            string imagesForPredictions = Path.Combine(assetsPath, "inputs", "images-for-predictions", "FlowersForPredictions");

            try
            {
                DownloadImageSet(imagesDownloadFolder);

                MLContext mlContext = new MLContext(seed: 1);

                //Load all the original images info
                IEnumerable<ImageData> images = LoadImagesFromDirectory(folder: imagesFolder, useFolderNameasLabel: true);
                IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);

                // Split the data 80:20 into train and test sets, train and evaluate.
                TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fullImagesDataset, testFraction: 0.2, seed: 1);
                IDataView trainDataset = trainTestData.TrainSet;
                IDataView testDataset = trainTestData.TestSet;

                var pipeline = mlContext.Transforms.Conversion.MapValueToKey("Label")
                    .Append(mlContext.Transforms.LoadImages("ImageObject", null, "ImagePath"))
                    .Append(mlContext.Transforms.ResizeImages("Image",
                        inputColumnName: "ImageObject", imageWidth: 299,
                        imageHeight: 299))
                    .Append(mlContext.Transforms.ExtractPixels("Image",
                        interleavePixelColors: true))
                    .Append(mlContext.Model.ImageClassification("Image", "Label",
                            arch: DnnEstimator.Architecture.InceptionV3,
                            epoch: 100, //An epoch is one learning cycle where the learner sees the whole training data set.
                            batchSize: 100, // batchSize sets then number of images to feed the model at a time
                            statisticsCallback: (epoch, accuracy, crossEntropy) => Console.WriteLine(
                                                                                        $"Epoch/training-cycle: {epoch}, " +
                                                                                        $"Accuracy: {accuracy * 100}%, " +
                                                                                        $"Cross-Entropy: {crossEntropy}")));

                Console.WriteLine("*** Training the image classification model with DNN Transfer Learning on top of the selected pre-trained model/architecture ***");

                // Measuring training time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                var trainedModel = pipeline.Fit(trainDataset);

                watch.Stop();
                long elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Training with transfer learning took: " + (elapsedMs / 1000).ToString() + " seconds");

                TrySinglePrediction(imagesForPredictions, mlContext, trainedModel);
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
        }

        private static void TrySinglePrediction(string imagesForPredictions, MLContext mlContext, TransformerChain<DnnTransformer> trainedModel)
        {
            // Create prediction function to try one prediction
            var predictionEngine = mlContext.Model
                .CreatePredictionEngine<ImageData, ImagePrediction>(trainedModel);

            IEnumerable<ImageData> testImages = LoadImagesFromDirectory(imagesForPredictions, false);
            ImageData imageToPredict = new ImageData
            {
                ImagePath = testImages.First().ImagePath
            };

            var prediction = predictionEngine.Predict(imageToPredict);

            // Find the original label names.
            VBuffer<ReadOnlyMemory<char>> keys = default;
            predictionEngine.OutputSchema["Label"].GetKeyValues(ref keys);

            var originalLabels = keys.DenseValues().ToArray();
            var index = prediction.PredictedLabel;

            Console.WriteLine($"ImageFile : [{Path.GetFileName(imageToPredict.ImagePath)}], " +
                              $"Scores : [{string.Join(",", prediction.Score)}], " +
                              $"Predicted Label : {originalLabels[index]}");
        }

        //EvaluateModel(mlContext, testDataset, trainedModel);
        private static void EvaluateModel(MLContext mlContext, IDataView testDataset, ITransformer trainedModel)
        {
            Console.WriteLine("Making bulk predictions and evaluating model's quality...");

            // Measuring time
            var watch2 = System.Diagnostics.Stopwatch.StartNew();

            IDataView predictions = trainedModel.Transform(testDataset);
            var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

            Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                              $"macro-accuracy = {metrics.MacroAccuracy}");

            watch2.Stop();
            long elapsed2Ms = watch2.ElapsedMilliseconds;

            Console.WriteLine("Predicting and Evaluation took: " + (elapsed2Ms / 1000).ToString() + " seconds");

            // Find out labels list
            //VBuffer<ReadOnlyMemory<char>> keys = default;
            //predictions.Schema["Label"].GetKeyValues(ref keys);
            //var originalLabels = keys.DenseValues().ToArray();
        }

        public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameasLabel = true)
        {
            var files = Directory.GetFiles(folder, "*",
                searchOption: SearchOption.AllDirectories);

            foreach (var file in files)
            {
                if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                    continue;

                var label = Path.GetFileName(file);
                if (useFolderNameasLabel)
                    label = Directory.GetParent(file).Name;
                else
                {
                    for (int index = 0; index < label.Length; index++)
                    {
                        if (!char.IsLetter(label[index]))
                        {
                            label = label.Substring(0, index);
                            break;
                        }
                    }
                }

                yield return new ImageData()
                {
                    ImagePath = file,
                    Label = label
                };

            }            
        }

        public static void DownloadImageSet(string imagesDownloadFolder)
        {
            // get a set of images to teach the network about the new classes

            //FULL FLOWERS IMAGESET (3,600 files)
            //string fileName = "flower_photos.tgz";
            //string url = $"http://download.tensorflow.org/example_images/{fileName}";
            //Web.Download(url, imagesDownloadFolder, fileName);
            //Compress.ExtractTGZ(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);

            //SMALL FLOWERS IMAGESET (200 files)
            string fileName = "flower_photos_small_set.zip";
            string url = $"https://mlnetfilestorage.file.core.windows.net/imagesets/flower_images/flower_photos_small_set.zip?st=2019-08-07T21%3A27%3A44Z&se=2030-08-08T21%3A27%3A00Z&sp=rl&sv=2018-03-28&sr=f&sig=SZ0UBX47pXD0F1rmrOM%2BfcwbPVob8hlgFtIlN89micM%3D";
            Web.Download(url, imagesDownloadFolder, fileName);
            Compress.UnZip(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);
        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
