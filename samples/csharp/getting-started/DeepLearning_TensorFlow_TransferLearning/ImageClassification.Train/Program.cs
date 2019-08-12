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

namespace ImageClassification.Train
{
    public class Program
    {
        static void Main(string[] args)
        {
            string assetsRelativePath = @"../../../assets";
            string assetsPath = GetAbsolutePath(assetsRelativePath);

            string imagesFolder = Path.Combine(assetsPath, "inputs", "images");
            string imagesForPredictions = Path.Combine(assetsPath, "inputs", "images-for-predictions", "FlowersForPredictions");

            try
            {
                MLContext mlContext = new MLContext(seed: 1);

                //Load all the original images info
                IEnumerable<ImageData> images = LoadImagesFromDirectory(folder:imagesFolder, useFolderNameasLabel:true);
                IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(images);
                IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

                // Split the data 80:20 into train and test sets, train and evaluate.
                TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.2, seed: 1);
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

                // Measuring time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                var trainedModel = pipeline.Fit(trainDataset);

                watch.Stop();
                long elapsedMs = watch.ElapsedMilliseconds;

                Console.WriteLine("Training with transfer learning took: " + (elapsedMs/1000).ToString() + " seconds");

                Console.WriteLine("Predicting and evaluating quality...");

                // Measuring time
                var watch2 = System.Diagnostics.Stopwatch.StartNew();

                var predictions = trainedModel.Transform(testDataset);
                var metrics = mlContext.MulticlassClassification.Evaluate(predictions);

                Console.WriteLine($"Micro-accuracy: {metrics.MicroAccuracy}," +
                                  $"macro-accuracy = {metrics.MacroAccuracy}");

                watch2.Stop();
                long elapsed2Ms = watch2.ElapsedMilliseconds;

                Console.WriteLine("Predicting and Evaluation took: " + (elapsed2Ms / 1000).ToString() + " seconds");


                // Create prediction function and test prediction
                var predictionEngine = mlContext.Model
                    .CreatePredictionEngine<ImageData, ImagePrediction>(trainedModel);

                IEnumerable<ImageData> testImages = LoadImagesFromDirectory(imagesForPredictions, false);
                ImageData imageToPredict = testImages.First();

                var prediction = predictionEngine.Predict(imageToPredict);
                                    
                //**** CESAR --> Need a much better way to show the best prediction labels and their probabilities ******* 
                Console.WriteLine($"ImageFile : [{Path.GetFileName(imageToPredict.ImagePath)}], " +
                                  $"Scores : [{string.Join(",", prediction.Score)}], " +
                                  $"Predicted Label : {prediction.PredictedLabel}");
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }

            Console.WriteLine("Press any key to finish");
            Console.ReadKey();
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

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
